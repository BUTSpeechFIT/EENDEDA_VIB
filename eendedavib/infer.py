#!/usr/bin/env python3

# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Copyright 2022 Brno University of Technology (author: Federico Landini)
# Copyright 2024 Brno University of Technology, National Institute of Informatics (authors: Lin Zhang)
# Licensed under the MIT license.

from backend.models import (
    average_checkpoints,
    get_model,
)
from common_utils.diarization_dataset import (
    KaldiDiarizationDataset,
    PrecomputedDiarizationFeatures)
from common_utils.gpu_utils import use_single_gpu
from os.path import join
from pathlib import Path
from scipy.signal import medfilt
from torch.utils.data import DataLoader
from train import _convert
from types import SimpleNamespace
from typing import TextIO, Tuple, List
import logging
import numpy as np
import os
import random
import torch
import yamlargparse
import matplotlib.pyplot as plt
import pickle

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_infer_dataloader(args: SimpleNamespace) -> DataLoader:
    infer_set = KaldiDiarizationDataset(
        args.infer_data_dir,
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
        use_last_samples=True,
        min_length=0,
    )
    infer_loader = DataLoader(
        infer_set,
        batch_size=1,
        collate_fn=_convert,
        num_workers=0,
        shuffle=False,
        worker_init_fn=_init_fn,
    )

    Y, _, _, _, _ = infer_set.__getitem__(0)
    assert Y.shape[1] == \
        (args.feature_dim * (1 + 2 * args.context_size)), \
        f"Expected feature dimensionality of \
        {args.feature_dim} but {Y.shape[1]} found."
    return infer_loader

def rttm_to_hard_labels(
    rttm_path: str,
    precision: float,
    length: float = -1
) -> Tuple[np.ndarray, List[str]]:
    """
        reads the rttm and returns a NfxNs matrix encoding the segments in
        which each speaker is present (labels 1/0) at the given precision.
        Ns is the number of speakers and Nf is the resulting number of frames,
        according to the parameters given.
        Nf might be shorter than the real length of the utterance, as final
        silence parts cannot be recovered from the rttm.
        If length is defined (s), it is to account for that extra silence.
        In case of silence all speakers are labeled with 0.
        In case of overlap all speakers involved are marked with 1.
        The function assumes that the rttm only contains speaker turns (no
        silence segments).
        The overlaps are extracted from the speaker turn collisions.
    """
    # each row is a turn, columns denote beginning (s) and duration (s) of turn
    data = np.loadtxt(rttm_path, usecols=[3, 4])
    # speaker id of each turn
    spks = np.loadtxt(rttm_path, usecols=[7], dtype='str')
    spk_ids = np.unique(spks)
    Ns = len(spk_ids)
    if data.shape[0] == 2 and len(data.shape) < 2:  # if only one segment
        data = np.asarray([data])
        spks = np.asarray([spks])
    # length of the file (s) that can be recovered from the rttm,
    # there might be extra silence at the end
    if len(data) == 0:
        len_file = 0
    else:
        len_file = data[-1][0]+data[-1][1]
    if length > len_file:
        len_file = length

    # matrix in given precision
    matrix = np.zeros([int(round(len_file*precision)), Ns])
    if len(data) > 0:
        # ranges to mark each turn
        ranges = np.around((np.array([data[:, 0],
                            data[:, 0]+data[:, 1]]).T*precision)).astype(int)

        for s in range(Ns):  # loop over speakers
            # loop all the turns of the speaker
            for init_end in ranges[spks == spk_ids[s], :]:
                matrix[init_end[0]:init_end[1], s] = 1  # mark the spk
    return matrix, spk_ids

def hard_labels_to_rttm(
    labels: np.ndarray,
    id_file: str,
    rttm_file: TextIO,
    frameshift: float = 10
) -> None:
    """
    Transform NfxNs matrix to an rttm file
    Nf is the number of frames
    Ns is the number of speakers
    The frameshift (in ms) determines how to interpret the frames in the array
    """
    if len(labels.shape) > 1:
        # Remove speakers that do not speak
        non_empty_speakers = np.where(labels.sum(axis=0) != 0)[0]
        labels = labels[:, non_empty_speakers]

    # Add 0's before first frame to use diff
    if len(labels.shape) > 1:
        labels = np.vstack([np.zeros((1, labels.shape[1])), labels])
    else:
        labels = np.vstack([np.zeros(1), labels])
    d = np.diff(labels, axis=0)

    spk_list = []
    ini_list = []
    end_list = []
    if len(labels.shape) > 1:
        n_spks = labels.shape[1]
    else:
        n_spks = 1
    for spk in range(n_spks):
        if n_spks > 1:
            ini_indices = np.where(d[:, spk] == 1)[0]
            end_indices = np.where(d[:, spk] == -1)[0]
        else:
            ini_indices = np.where(d[:] == 1)[0]
            end_indices = np.where(d[:] == -1)[0]
        # Add final mark if needed
        if len(ini_indices) == len(end_indices) + 1:
            end_indices = np.hstack([
                end_indices,
                labels.shape[0] - 1])
        assert len(ini_indices) == len(end_indices), \
            "Quantities of start and end of segments mismatch. \
            Are speaker labels correct?"
        n_segments = len(ini_indices)
        for index in range(n_segments):
            spk_list.append(spk)
            ini_list.append(ini_indices[index])
            end_list.append(end_indices[index])
    for ini, end, spk in sorted(zip(ini_list, end_list, spk_list)):
        rttm_file.write(
            f"SPEAKER {id_file} 1 " +
            f"{round(ini * frameshift / 1000, 3)} " +
            f"{round((end - ini) * frameshift / 1000, 3)} " +
            f"<NA> <NA> spk{spk} <NA> <NA>\n")


def _init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def postprocess_output(
    probabilities,
    subsampling: int,
    threshold: float,
    median_window_length: int
) -> np.ndarray:
    thresholded = probabilities > threshold
    filtered = np.zeros(thresholded.shape)
    for spk in range(filtered.shape[1]):
        filtered[:, spk] = medfilt(
            thresholded[:, spk].int(),
            kernel_size=median_window_length)
    probs_extended = np.repeat(filtered, subsampling, axis=0)
    return probs_extended


def parse_arguments() -> SimpleNamespace:
    parser = yamlargparse.ArgumentParser(description='EEND inference')
    parser.add_argument('-c', '--config', help='config file path',
                        action=yamlargparse.ActionConfigFile)
    parser.add_argument('--context-size', default=0, type=int)
    parser.add_argument('--encoder-units', type=int,
                        help='number of units in the encoder')
    parser.add_argument('--epochs', type=str,
                        help='epochs to average separated by commas \
                        or - for intervals.')
    parser.add_argument('--feature-dim', type=int)
    parser.add_argument('--frame-size', type=int)
    parser.add_argument('--frame-shift', type=int)
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--hidden-size', type=int,
                        help='number of units in SA blocks')
    parser.add_argument('--infer-data-dir', help='inference data directory.')
    parser.add_argument('--input-transform', default='',
                        choices=['logmel', 'logmel_meannorm',
                                 'logmel_meanvarnorm'],
                        help='input normalization transform')
    parser.add_argument('--log-report-batches-num', default=1, type=float)
    parser.add_argument('--median-window-length', default=11, type=int)
    parser.add_argument('--model-type', default='TransformerEDA',
                        help='Type of model (for now only TransformerEDA)')
    parser.add_argument('--models-path', type=str,
                        help='directory with model(s) to evaluate')
    parser.add_argument('--num-frames', default=-1, type=int,
                        help='number of frames in one utterance')
    parser.add_argument('--num-speakers', type=int)
    parser.add_argument('--rttms-dir', type=str,
                        help='output directory for rttm files.')
    parser.add_argument('--sampling-rate', type=int)
    parser.add_argument('--seed', default=3, type=int)
    parser.add_argument('--subsampling', default=10, type=int)
    parser.add_argument('--threshold', default=0.5, type=float)
    parser.add_argument('--transformer-encoder-n-heads', type=int)
    parser.add_argument('--transformer-encoder-n-layers', type=int)
    parser.add_argument('--transformer-encoder-dropout', type=float)
    parser.add_argument('--vad-loss-weight', default=0.0, type=float)
    parser.add_argument('-r','--ref-rttms-dir', help='reference rttm directory.')

    # for vib
    parser.add_argument(
        '--use-mu-z', type=bool, default=False,
        help='If True, use mu for spknum-wise branch.')
    parser.add_argument(
        '--use-mu-e', type=bool, default=False,
        help='If True, use mu for frame-wise branch.')
    parser.add_argument('--zkld-loss-weight', default=[0.000001], type=float, nargs='+')
    parser.add_argument('--ekld-loss-weight', default=[0.000001], type=float, nargs='+')
    parser.add_argument(
        '--kld-weight-type', type=str, default = 'none', choices = ['none', 'cyclical_des', 'cyclical_asc', 'un', 'dwa'],
        help='how will you adjust value of kld_loss_weight?, ')
    parser.add_argument(
        '--num-esamples', type=int, default = 0,
        help='how many sampling we want to do')
    parser.add_argument('--temp', default='none',type=str,
                        help='temporary parameter for debug or maintainment. ')
    parser.add_argument('--average-type', type=str, default = 'sample', 
                    choices = ['loss', 'prob', 'sample', ],
        help='how will you treat samples? average samples or average loss? ')

    attractor_args = parser.add_argument_group('attractor')
    attractor_args.add_argument(
        '--time-shuffle', action='store_true',
        help='Shuffle time-axis order before input to the network')
    attractor_args.add_argument('--attractor-loss-ratio', default=1.0,
                                type=float, help='weighting parameter')
    attractor_args.add_argument('--attractor-encoder-dropout',
                                default=0.1, type=float)
    attractor_args.add_argument('--attractor-decoder-dropout',
                                default=0.1, type=float)
    attractor_args.add_argument('--estimate-spk-qty', default=-1, type=int)
    attractor_args.add_argument('--estimate-spk-qty-thr',
                                default=-1, type=float)
    attractor_args.add_argument(
        '--detach-attractor-loss', default=False, type=bool,
        help='If True, avoid backpropagation on attractor loss')

    attractor_args.add_argument('--plot-output', default=False, type=bool)
    attractor_args.add_argument(
        '--num-zsamples', type=int, default = 0,
        help='how many sampling we want to do for z')
    parser.add_argument('--infer-type', default='sample',
        choices=['mu', 'sample', 'logprob', 'avgprob'],
        help='how to calculate samples\' posterior for inference? use mu directly or average samples.\
	this one has different options with average-type')
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
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(args.seed)

    logging.info(args)

    infer_loader = get_infer_dataloader(args)

    if args.gpu >= 1:
        gpuid = use_single_gpu(args.gpu)
        logging.info('GPU device {} is used'.format(gpuid))
        args.device = torch.device("cuda")
    else:
        gpuid = -1
        args.device = torch.device("cpu")

    assert args.estimate_spk_qty_thr != -1 or \
        args.estimate_spk_qty != -1, \
        ("Either 'estimate_spk_qty_thr' or 'estimate_spk_qty' "
         "arguments have to be defined.")
    if args.estimate_spk_qty != -1:
        out_dir = join(args.rttms_dir, f"spkqty{args.estimate_spk_qty}_\
            thr{args.threshold}_median{args.median_window_length}")
    elif args.estimate_spk_qty_thr != -1:
        out_dir = join(args.rttms_dir, f"spkqtythr{args.estimate_spk_qty_thr}_\
            thr{args.threshold}_median{args.median_window_length}")

    model = get_model(args)

    model = average_checkpoints(
        args.device, model, args.models_path, args.epochs)
    model.eval()
    print("Parameters are: %.4f" %count_parameters(model))

    app_name=""	    
    if(args.use_mu_z):
        app_name+="nzmu"		
    else:		
        assert(args.num_zsamples > 0)		
        app_name+="nz{}".format(args.num_zsamples)
    if(args.use_mu_e):
        app_name+="_nemu"		
    else:		
        assert(args.num_esamples > 0)		
        app_name+="_ne{}".format(args.num_esamples)
    app_name+="_t{}".format(args.infer_type)

    out_dir = join(
        args.rttms_dir,
        f"epochs{args.epochs}",
        f"{app_name}",
        f"timeshuffle{args.time_shuffle}",
        (f"spk_qty{args.estimate_spk_qty}_"
            f"spk_qty_thr{args.estimate_spk_qty_thr}"),
        f"detection_thr{args.threshold}",
        f"median{args.median_window_length}",
        "rttms"
    )
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    zmu_out_dir = join(
        args.rttms_dir,
        f"epochs{args.epochs}",
        f"{app_name}",
    )

    prob_out_dir = join(
        args.rttms_dir,
        f"epochs{args.epochs}",
        f"{app_name}",
        f"timeshuffle{args.time_shuffle}",
    )

    attprob_dict={}
    zmu_dict={}
    zlogvar_dict={}
    zmu_1dim_dict={}
    zlogvar_1dim_dict={}
    emu_dict={}
    elogvar_dict={}
    emu_1dim_dict={}
    elogvar_1dim_dict={}
    for i, batch in enumerate(infer_loader):
        input = torch.stack(batch['xs']).to(args.device)
        name = batch['names'][0]
        with torch.no_grad():
            y_pred, y_probs, att_probs, z_mu, z_logvar, e_mu, e_logvar = model.estimate_sequential(input, args, return_ys=True)
        # Each one has a single sequence
        y_pred = y_pred[0] #[T, 2]
        y_probs = y_probs  #[T, S+1]
        post_y = postprocess_output(
            y_pred, args.subsampling,
            args.threshold, args.median_window_length)
        rttm_filename = join(out_dir, f"{name}.rttm")
        with open(rttm_filename, 'w') as rttm_file:
            hard_labels_to_rttm(post_y, name, rttm_file)

        attprob_dict[name] = att_probs.tolist() #np.array(y_probs.data)
        zmu_dict[name] = z_mu[0] #[K+1, D]
        zmu_1dim_dict[name] = torch.cat([model.eda.counter.to(model.device)(z) for z in z_mu[0]]).tolist() #[K+1]
        zlogvar_dict[name] = z_logvar[0] #[B=1, K+!, D] -> [K+1, D]
        zlogvar_1dim_dict[name] = torch.cat([model.eda.counter.to(model.device)(z) for z in z_logvar[0]]).tolist() #[K+1]

        emu_dict[name] = e_mu[0] #[K+1, D]
        emu_1dim_dict[name] = torch.cat([model.eda.counter.to(model.device)(e) for e in e_mu[0]]).tolist() #[K+1]
        elogvar_dict[name] = e_logvar[0] #[B=1, K+!, D] -> [K+1, D]
        elogvar_1dim_dict[name] = torch.cat([model.eda.counter.to(model.device)(e) for e in e_logvar[0]]).tolist() #[K+1]

    with open(join(prob_out_dir, "att_probs.pkl"), 'wb') as f:
        pickle.dump(attprob_dict, f)

    with open(join(prob_out_dir, "att_probs.txt"), 'w') as f:
        for k,v in attprob_dict.items():
            f.write("%s %s \n" % (k, np.array(v).tolist()))
    os.system("sed -i 's/ \[/, /g' %s" %(os.path.join(prob_out_dir, "att_probs.txt")))
    os.system("sed -i 's/\]//g' %s " %(os.path.join(prob_out_dir, "att_probs.txt")))

    np.savez(join(zmu_out_dir, "z"), zmu_dict = zmu_dict, zmu_1dim_dict= zmu_1dim_dict,
		  zlogvar_dict = zlogvar_dict,  zlogvar_1dim_dict = zlogvar_1dim_dict)
    np.savez(join(zmu_out_dir, "e"), emu_dict = emu_dict, emu_1dim_dict= emu_1dim_dict,
		  elogvar_dict = elogvar_dict,  elogvar_1dim_dict = elogvar_1dim_dict)
