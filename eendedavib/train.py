#!/usr/bin/env python3

# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Copyright 2022 Brno University of Technology (authors: Federico Landini)
# Copyright 2024 Brno University of Technology, National Institute of Informatics (authors: Lin Zhang)
# Licensed under the MIT license.


from backend.models import (
    average_checkpoints,
    get_model,
    load_checkpoint,
    pad_labels,
    pad_sequence,
    save_checkpoint,
)
from backend.updater import setup_optimizer, get_rate
from common_utils.diarization_dataset import (
    KaldiDiarizationDataset,
    PrecomputedDiarizationFeatures)

from common_utils.gpu_utils import use_single_gpu
from common_utils.metrics import (
    calculate_metrics,
    new_metrics,
    reset_metrics,
    update_metrics,
)
from common_utils.pytorchtools import EarlyStopping

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.profiler
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple
import numpy as np
import os
import random
import torch
import logging
import yamlargparse
#torch.autograd.set_detect_anomaly(True)	     


def _init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _convert(
    batch: List[Tuple[torch.Tensor, torch.Tensor, str]]
) -> Dict[str, Any]:
    return {'xs': [x for x, _, _, _, _ in batch],
            'ts': [t for _, t, _, _, _ in batch],
            'names': [r for _, _, r, _, _ in batch]}


#Cyclical Annealing Schedule: A Simple Approach to Mitigating KL Vanishing
#https://arxiv.org/abs/1903.10145
#github.com/haofuml/cyclical_annealing
def frange_cycle_linear(n_iter, start=0.0, stop=1.0,  n_cycle=4, ratio=0.5, order = 'asc'):
    L = np.ones(n_iter) * stop
    period = n_iter/n_cycle
    step = abs(stop-start)/(period*ratio) # linear schedule

    if(order == 'asc'):
        assert stop > start    
        for c in range(n_cycle):
            v, i = start, 0
            while v <= stop and (int(i+c*period) < n_iter):
                L[int(i+c*period)] = v
                v += step
                i += 1
    elif(order == 'des'):
        assert stop < start    
        for c in range(n_cycle):
            v, i = start, 0
            while v >= stop and (int(i+c*period) < n_iter):
                L[int(i+c*period)] = v
                v -= step
                i += 1
    else:
        raise NotImplementedError      
    return L 

def get_KLD_weights(kld_weight_type, kld_loss_weight, max_epochs):
    if(kld_weight_type == 'cyclical_des'):
        kld_L = frange_cycle_linear(max_epochs, start=1.0, stop=0.0,  n_cycle=10, ratio=0.5, order='des')
    elif(kld_weight_type == 'cyclical_asc'):
        kld_L = frange_cycle_linear(max_epochs, start=0.0, stop=1.0,  n_cycle=10, ratio=0.5, order='asc')
    elif(kld_weight_type == 'none'):
        if isinstance(kld_loss_weight, list):  
            kld_L = np.full(max_epochs, kld_loss_weight[-1])
            for ldx, kld in enumerate(kld_loss_weight): kld_L[ldx*10:(ldx+1)*10] = kld	
        else: kld_L = np.full(max_epochs, kld_loss_weight)
    else:
        raise NotImplementedError      
    return kld_L


def compute_loss_and_metrics(
    model: torch.nn.Module,
    labels: torch.Tensor,
    input: torch.Tensor,
    n_speakers: List[int],
    acum_metrics: Dict[str, float],
    vad_loss_weight: float,
    detach_attractor_loss: bool,
    zkld_loss_weight: float,
    ekld_loss_weight: float,
    permutation_type: str,
    average_type: str,
    acfunc: str = 'None',
) -> Tuple[torch.Tensor, Dict[str, float]]:
    if(args.average_type in [ 'loss', 'prob', 'sample']):	
        y_mu_pred, y_pred, attractor_loss, zKLD_loss, eKLD_loss = model(input, labels, n_speakers, args) #[]
        loss, standard_loss = model.get_loss(
            ys_mu = y_mu_pred, ys = y_pred, target = labels, n_speakers = n_speakers, 
	    attractor_loss = attractor_loss, vad_loss_weight = vad_loss_weight,
            zkld_loss = zKLD_loss, zkld_loss_weight = zkld_loss_weight, 
	    ekld_loss = eKLD_loss, ekld_loss_weight = ekld_loss_weight, 
	    detach_attractor_loss = detach_attractor_loss,
            permutation_type = permutation_type,
	    average_type = average_type,
	    acfunc = acfunc)
        	    
    metrics = calculate_metrics(
        labels.detach(), y_mu_pred.detach(), threshold=0.5)
    acum_metrics = update_metrics(acum_metrics, metrics)
    acum_metrics['loss'] += loss.item()
    acum_metrics['loss_standard'] += standard_loss.item()
    acum_metrics['loss_attractor'] += attractor_loss.item()
    acum_metrics['loss_zKLD'] += zKLD_loss.item()
    acum_metrics['loss_eKLD'] += eKLD_loss.item()
    return loss, acum_metrics


def get_training_dataloaders(
    args: SimpleNamespace
) -> Tuple[DataLoader, DataLoader]:

    if args.gpu >= 1:
        train_batchsize = args.train_batchsize * args.gpu
        dev_batchsize = args.dev_batchsize * args.gpu
    else:
        train_batchsize = args.train_batchsize
        dev_batchsize = args.dev_batchsize

    if  args.train_features_dir is None and args.valid_features_dir is None:
        assert not (args.train_data_dir is None) and \
            not (args.valid_data_dir is None), "\
            --train-features-dir and \
            --valid-features-dir or --train-data-dir and --valid-data-dir \
            must be defined"
        # By default use train_data_dir and valid_data_dir
        train_set = KaldiDiarizationDataset(
            args.train_data_dir,
            chunk_size=args.num_frames,
            context_size=args.context_size,
            feature_dim=args.feature_dim,
            frame_shift=args.frame_shift,
            frame_size=args.frame_size,
            input_transform=args.input_transform,
            n_speakers=args.num_speakers,
            sampling_rate=args.sampling_rate,
            shuffle=args.time_shuffle,
            subsampling=args.subsampling,
            use_last_samples=args.use_last_samples,
            min_length=args.min_length,
        )
        train_loader = DataLoader(
            train_set,
            batch_size=args.train_batchsize,
            collate_fn=_convert,
            num_workers=args.num_workers,
            shuffle=True,
            worker_init_fn=_init_fn,
        )

        dev_set = KaldiDiarizationDataset(
            args.valid_data_dir,
            chunk_size=args.num_frames,
            context_size=args.context_size,
            feature_dim=args.feature_dim,
            frame_shift=args.frame_shift,
            frame_size=args.frame_size,
            input_transform=args.input_transform,
            n_speakers=args.num_speakers,
            sampling_rate=args.sampling_rate,
            shuffle=args.time_shuffle,
            subsampling=args.subsampling,
            use_last_samples=args.use_last_samples,
            min_length=args.min_length,
        )
        dev_loader = DataLoader(
            dev_set,
            batch_size=args.dev_batchsize,
            collate_fn=_convert,
            num_workers=1,
            shuffle=False,
            worker_init_fn=_init_fn,
        )

        Y_train, _, _, _, _ = train_set.__getitem__(0)
        Y_dev, _, _, _, _ = dev_set.__getitem__(0)
        assert Y_train.shape[1] == Y_dev.shape[1], \
            f"Train features dimensionality ({Y_train.shape[1]}) and \
            dev features dimensionality ({Y_dev.shape[1]}) differ."
        assert Y_train.shape[1] == (
            args.feature_dim * (1 + 2 * args.context_size)), \
            f"Expected feature dimensionality of {args.feature_dim} \
            but {Y_train.shape[1]} found."
    else:
        assert not (args.train_features_dir is None) and \
            not (args.valid_features_dir is None), "\
            --train-features-dir and \
            --valid-features-dir or --train-data-dir and --valid-data-dir \
            must be defined"
        train_set = PrecomputedDiarizationFeatures(
            features_dir=args.train_features_dir,
            chunk_size=args.num_frames,
            shuffle=args.time_shuffle,
            use_last_samples=args.use_last_samples,
            min_length=args.min_length,
        )

        dev_set = PrecomputedDiarizationFeatures(
            features_dir=args.valid_features_dir,
            chunk_size=args.num_frames,
            shuffle=args.time_shuffle,
            use_last_samples=args.use_last_samples,
            min_length=args.min_length,
        )

        train_loader = DataLoader(
            train_set,
            batch_size=train_batchsize,
            collate_fn=_convert,
            num_workers=args.num_workers,
            shuffle=True,
            worker_init_fn=_init_fn,
        )

        dev_loader = DataLoader(
            dev_set,
            batch_size=dev_batchsize,
            collate_fn=_convert,
            num_workers=1,
            shuffle=False,
            worker_init_fn=_init_fn,
        )

    return train_loader, dev_loader


def parse_arguments() -> SimpleNamespace:
    parser = yamlargparse.ArgumentParser(description='EEND training')
    parser.add_argument('-c', '--config', help='config file path',
                        action=yamlargparse.ActionConfigFile)
    parser.add_argument('--context-size', default=0, type=int)
    parser.add_argument('--dev-batchsize', default=1, type=int,
                        help='number of utterances in one development batch')
    parser.add_argument('--encoder-units', type=int,
                        help='number of units in the encoder')
    parser.add_argument('--feature-dim', type=int)
    parser.add_argument('--frame-shift', type=int)
    parser.add_argument('--frame-size', type=int)
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--gradclip', default=-1, type=int,
                        help='gradient clipping. if < 0, no clipping')
    parser.add_argument('--hidden-size', type=int,
                        help='number of units in SA blocks')
    parser.add_argument('--init-epochs', type=str, default='',
                        help='Initialize model with average of epochs \
                        separated by commas or - for intervals.')
    parser.add_argument('--init-model-path', type=str, default='',
                        help='Initialize the model from the given directory')
    parser.add_argument('--input-transform', default='',
                        choices=['logmel', 'logmel_meannorm',
                                 'logmel_meanvarnorm'],
                        help='input normalization transform')
    parser.add_argument('--log-report-batches-num', default=1, type=float)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--max-epochs', type=int,
                        help='Max. number of epochs to train')
    parser.add_argument('--min-length', default=0, type=int,
                        help='Minimum number of frames for the sequences'
                             ' after downsampling.')
    parser.add_argument('--model-type', default='TransformerEDA',
                        help='Type of model (for now only TransformerEDA)')
    parser.add_argument('--noam-warmup-steps', default=100000, type=float)
    parser.add_argument('--num-frames', default=500, type=int,
                        help='number of frames in one utterance')
    parser.add_argument('--num-speakers', type=int,
                        help='maximum number of speakers allowed')
    parser.add_argument('--num-workers', default=1, type=int,
                        help='number of workers in train DataLoader')
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--output-path', type=str)
    parser.add_argument('--sampling-rate', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--subsampling', default=10, type=int)
    parser.add_argument('--train-batchsize', default=1, type=int,
                        help='number of utterances in one train batch')
    parser.add_argument('--train-data-dir',
                        help='kaldi-style data dir used for training.')
    parser.add_argument('--train-features-dir', default=None,
                        help='directory with pre-computed training features')
    parser.add_argument('--transformer-encoder-dropout', type=float)
    parser.add_argument('--transformer-encoder-n-heads', type=int)
    parser.add_argument('--transformer-encoder-n-layers', type=int)
    parser.add_argument('--use-last-samples', default=True, type=bool)
    parser.add_argument('--vad-loss-weight', default=0.0, type=float)
    parser.add_argument('--valid-data-dir',
                        help='kaldi-style data dir used for validation.')
    parser.add_argument('--valid-features-dir', default=None,
                        help='directory with pre-computed validation features')

    parser.add_argument('--early-stop-step', default=0, type=int)
    parser.add_argument(
        '--use-mu-z', type=bool, default=False,
        help='If True, use mu for spknum-wise branch.')
    parser.add_argument(
        '--use-mu-e', type=bool, default=False,
        help='If True, use mu for frame-wise branch.')
    parser.add_argument(
        '--zKLD-invalid', type=bool, default=False,
        help='If True, we will include KLD for the invalid spk.')
    
    parser.add_argument('--zkld-loss-weight', default=[0.000001], type=float, nargs='+')
    parser.add_argument('--ekld-loss-weight', default=[0.000001], type=float, nargs='+')
    parser.add_argument(
        '--kld-weight-type', type=str, default = 'none', choices = ['none', 'cyclical_des', 'cyclical_asc', 'un', 'dwa'],
        help='how will you adjust value of kld_loss_weight?, ')
    parser.add_argument(
        '--num-esamples', type=int, default = 0,
        help='how many sampling we want to do')
    parser.add_argument('--permutation-type', default='mu', choices=['mu', 'sample'],
                        help='decide permutation based on mean or sample. Note this is only used to find the best map, \
					we will still use sampled vector to calculate loss.')
    parser.add_argument('--temp', default='none',type=str,
                        help='temporary parameter for debug or maintainment. ')
    parser.add_argument('--acfunc', default='sigmoid',type=str, choices = ['sigmoid' ],
                        help='active function for LLR.')
    parser.add_argument('--average-type', type=str, default = 'sample', 
                    choices = ['loss', 'prob', 'sample' ],
        help='how will you treat samples? average samples or average loss? ')

    parser.add_argument('--KLD-cal', type=str, default = 'sum', 
		    choices = ['sum', 'mean'],
        help='how to calculate KLD for sample? sum for sumup all samples, mean for average')

    attractor_args = parser.add_argument_group('attractor')
    attractor_args.add_argument(
        '--time-shuffle', action='store_true',
        help='Shuffle time-axis order before input to the network')
    attractor_args.add_argument(
        '--attractor-loss-ratio', default=1.0, type=float,
        help='weighting parameter')
    attractor_args.add_argument(
        '--attractor-encoder-dropout', type=float)
    attractor_args.add_argument(
        '--attractor-decoder-dropout', type=float)
    attractor_args.add_argument(
        '--detach-attractor-loss', type=bool,
        help='If True, avoid backpropagation on attractor loss')
    attractor_args.add_argument(
        '--num-zsamples', type=int, default = 0,
        help='how many sampling we want to do for z')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arguments()

    # For reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    np.random.seed(args.seed)  # Numpy module.
    random.seed(args.seed)  # Python random module.
    torch.manual_seed(args.seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(args.seed)

    logging.info(args)

    writer = SummaryWriter(f"{args.output_path}/tensorboard")


    train_loader, dev_loader = get_training_dataloaders(args)

    if args.gpu >= 1:
        gpuid = use_single_gpu(args.gpu)
        logging.info('GPU device {} is used'.format(gpuid))
        args.device = torch.device("cuda")
    else:
        gpuid = -1
        args.device = torch.device("cpu")

    if args.init_model_path == '':
        model = get_model(args)
        optimizer = setup_optimizer(args, model)
    else:
        model = get_model(args)
        model = average_checkpoints(
            args.device, model, args.init_model_path, args.init_epochs)
        optimizer = setup_optimizer(args, model)

    train_batches_qty = len(train_loader)
    dev_batches_qty = len(dev_loader)
    logging.info(f"#batches quantity for train: {train_batches_qty}")
    logging.info(f"#batches quantity for dev: {dev_batches_qty}")

    acum_train_metrics = new_metrics()
    acum_dev_metrics = new_metrics()

    if os.path.isfile(os.path.join(
            args.output_path, 'models', 'checkpoint_0.tar')):
        # Load latest model and continue from there
        directory = os.path.join(args.output_path, 'models')
        checkpoints = os.listdir(directory)
        paths = [os.path.join(directory, basename) for
                 basename in checkpoints if basename.startswith("checkpoint_")]
        latest = max(paths, key=os.path.getctime)
        epoch, model, optimizer, _ = load_checkpoint(args, latest)
        init_epoch = epoch
    else:
        init_epoch = 0
        # Save initial model
        save_checkpoint(args, init_epoch, model, optimizer, 0)

    ## initialize the early_stopping object
    early_stopping = EarlyStopping(patience=args.early_stop_step, verbose=True)

    zkld_L = get_KLD_weights(args.kld_weight_type, args.zkld_loss_weight, args.max_epochs)
    ekld_L = get_KLD_weights(args.kld_weight_type, args.ekld_loss_weight, args.max_epochs)

    for epoch in range(init_epoch, args.max_epochs):
        zkld_loss_weight = zkld_L[epoch]
        ekld_loss_weight = ekld_L[epoch]

        model.train()
        for i, batch in enumerate(train_loader):
            features = batch['xs'] #List<[T, D]>, D=345
            labels = batch['ts'] #List<[T, S]>
            n_speakers = np.asarray([max(torch.where(t.sum(0) != 0)[0]) + 1
                                     if t.sum() > 0 else 0 for t in labels])
            max_n_speakers = max(n_speakers)
            features, labels = pad_sequence(features, labels, args.num_frames)
            labels = pad_labels(labels, max_n_speakers)
            features = torch.stack(features).to(args.device)
            labels = torch.stack(labels).to(args.device)
            #DEBUG speed
            #model.eda.test_speed_sampling(10000)	    
            loss, acum_train_metrics = compute_loss_and_metrics(
                model, labels, features, n_speakers, acum_train_metrics,
                args.vad_loss_weight,
                args.detach_attractor_loss, 
		zkld_loss_weight, ekld_loss_weight,
		args.permutation_type, args.average_type,
		args.acfunc)
            if i % args.log_report_batches_num == \
                    (args.log_report_batches_num-1):
                for k in acum_train_metrics.keys():
                    writer.add_scalar(
                        f"train_{k}",
                        acum_train_metrics[k] / args.log_report_batches_num,
                        epoch * train_batches_qty + i)
                writer.add_scalar(
                    "lrate",
                    get_rate(optimizer),
                    epoch * train_batches_qty + i)
                writer.add_scalar(
                    "zkld_weight",
                    zkld_loss_weight,
                    epoch * train_batches_qty + i)
                writer.add_scalar(
                    "ekld_weight",
                    ekld_loss_weight,
                    epoch * train_batches_qty + i)
                acum_train_metrics = reset_metrics(acum_train_metrics)
            optimizer.zero_grad()
            #print("%s-%s" %(epoch, i))	    
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradclip)
            optimizer.step()

            #prof.step()

        save_checkpoint(args, epoch+1, model, optimizer, loss)

        with torch.no_grad():
            model.eval()
            for i, batch in enumerate(dev_loader):
                features = batch['xs']
                labels = batch['ts']
		#calculate max spk num for each file. 
                n_speakers = np.asarray([max(torch.where(t.sum(0) != 0)[0]) + 1
                                        if t.sum() > 0 else 0 for t in labels])
                max_n_speakers = max(n_speakers)
                features, labels = pad_sequence(
                    features, labels, args.num_frames)
                labels = pad_labels(labels, max_n_speakers)
                features = torch.stack(features).to(args.device)
                labels = torch.stack(labels).to(args.device)
                zkld_loss_weight = args.zkld_loss_weight[-1] if isinstance(args.zkld_loss_weight, list) else args.zkld_loss_weight 		    
                ekld_loss_weight = args.ekld_loss_weight[-1] if isinstance(args.ekld_loss_weight, list) else args.ekld_loss_weight 		    
                _, acum_dev_metrics = compute_loss_and_metrics(
                model, labels, features, n_speakers, acum_dev_metrics,
                args.vad_loss_weight,
                args.detach_attractor_loss, 
		zkld_loss_weight, ekld_loss_weight,
		args.permutation_type, args.average_type,
		args.acfunc)
        for k in acum_dev_metrics.keys():
            writer.add_scalar(
                f"dev_{k}", acum_dev_metrics[k] / dev_batches_qty,
                epoch * dev_batches_qty + i)
        #early stop
        if(args.early_stop_step >0):
            early_stopping(acum_dev_metrics['loss'])
            if(early_stopping.early_stop):
                print("Early stopping on %s" %(epoch))
                break	    

        acum_dev_metrics = reset_metrics(acum_dev_metrics)
        torch.cuda.empty_cache()

