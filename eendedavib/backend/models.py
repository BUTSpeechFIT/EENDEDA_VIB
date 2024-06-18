#!/usr/bin/env python3

# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Copyright 2022 Brno University of Technology (author: Federico Landini)
# Copyright 2024 Brno University of Technology (authors: Lin Zhang, Lukas Burget)
# Licensed under the MIT license.

from os.path import isfile, join

from backend.losses import (
    pit_loss_multispk_ori,
    pit_loss_multispk,
    vad_loss,
)
from backend.updater import (
    NoamOpt,
    setup_optimizer,
)
from pathlib import Path
from torch.nn import Module, ModuleList
from types import SimpleNamespace
from typing import Dict, List, Tuple
import copy
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import logging
import gc


"""
T: number of frames
C: number of speakers (classes)
D: dimension of embedding (for deep clustering loss)
B: mini-batch size
"""


def get_KLD_loss(mu, log_var, cal='mean'):
    if(cal == 'mean'):	 
        KLD_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    else:    
        KLD_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return KLD_loss

class EncoderDecoderAttractor(Module):
    def __init__(
        self,
        device: torch.device,
        n_units: int,
        encoder_dropout: float,
        decoder_dropout: float,
        detach_attractor_loss: bool,
        average_type: str = 'sample',
    ) -> None:
        """ 
        num_zsamples: how many sample you want to take from distribution of z.
        average_type: sample for average samples, loss for average loss.
        """
        super(EncoderDecoderAttractor, self).__init__()
        self.device = device

        #encoder:
        self.encoder = torch.nn.LSTM(
            input_size=n_units,
            hidden_size=n_units,
            num_layers=1,
            dropout=encoder_dropout,
            batch_first=True,
            device=self.device)


        #decoder:
        self.decoder = torch.nn.LSTM(
            input_size=n_units,
            hidden_size=n_units,
            num_layers=1,
            dropout=decoder_dropout,
            batch_first=True,
            device=self.device)

        self.n_units = n_units
        self.detach_attractor_loss = detach_attractor_loss

        self.linearmu = torch.nn.Linear(n_units, n_units, device=self.device)
        self.linearvar = torch.nn.Linear(n_units, n_units, device=self.device)
        self.counter = torch.nn.Linear(n_units, 1, device=self.device)
        
        self.average_type = average_type

        self.softplus = torch.nn.Softplus()	

    def forward(self, xs: torch.Tensor, zeros: torch.Tensor, temp = 'None') -> torch.Tensor:
        #enc.	    
        _, (hx, cx) = self.encoder.to(self.device)(xs.to(self.device))

        #dec.	
        hiddens, (_, _) = self.decoder.to(self.device)(
            zeros.to(self.device),
            (hx.to(self.device), cx.to(self.device))
        ) #return [B, K+1, D]
        mu = self.linearmu(hiddens)  			#[B, K+1, n_unit]
        log_var = self.linearvar(hiddens)		#[B, K+1, n_unit]

        KLD_loss = get_KLD_loss(mu, log_var) # NOTE: this KLD considered invlid spk. 

        if(temp == 'softplus'):
            return mu, -self.softplus(log_var), KLD_loss 
        return mu, log_var, KLD_loss 

    def sampling(self,
        mu, log_var):
        """
        log_var = log(std^2)
        """    
        std = torch.exp(0.5 * log_var) 
        esp = torch.randn_like(std)  #[B, S, D]
        return esp.mul(std).add_(mu) # [B, S, D]

    def mul_sampling(self,
        mu, log_var, num_sample, use_mu=False):
        assert( (use_mu) + (num_sample>0) == 1 )
        if(num_sample < 1 and use_mu):
            return mu.unsqueeze(0)		
        std = torch.exp(0.5 * log_var)
        esp = torch.randn_like(std.unsqueeze(0).repeat(num_sample, 1, 1, 1)) # [M, B, S, D]
        return esp.mul(std).add_(mu) # [M, B, S, D]

    def estimate(
        self,
        xs: torch.Tensor,
        max_n_speakers: int = 15,
        infer_type: str = 'sample',
        temp = 'None',
	num_zsamples: int = 1,
	use_mu : bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate attractors from embedding sequences
         without prior knowledge of the number of speakers
        Args:
          xs: List of (T,D)-shaped embeddings
          max_n_speakers (int)
	  infer_type: [mu|sample|logprob|avgprob]
	  num_zsamples: int
	  use_mu: bool
        Returns:
          attractors: List of (N,D)-shaped attractors
          probs: List of attractor existence probabilities
	  z_mu, z_logvar: mu and logvar of z (attractors)
	  z_samples: sampled attractors
        """
        zeros = torch.zeros((xs.shape[0], max_n_speakers+1, self.n_units))
        z_mu, z_logvar, _ = self.forward(xs, zeros, temp) #xs: [B, T, D], zeros:[B, K+1, D], z_mu/logvar [B, K+1, D]
        if(infer_type == 'mu'):
            attractors = z_mu
            z_samples = attractors
            probs = torch.sigmoid(
                torch.flatten(self.counter.to(self.device)(attractors)))
        else:
            z_samples = self.mul_sampling(z_mu, z_logvar, num_zsamples, use_mu) #[M, B, S+1, D]
            if(infer_type == 'sample'):
                attractors = z_samples.mean(axis=0) 
                #attractors: [1, K+1, D], z_samples: [M, B, K+1, D] average folloeing the sampling axis.  K=15 as default here.   
                probs = torch.sigmoid(
                    torch.flatten(self.counter.to(self.device)(attractors)))  #
            elif(infer_type == 'logprob'):
                attractors = z_samples
                tmp_probs = torch.sigmoid(
                    torch.flatten(self.counter.to(self.device)(z_samples))).reshape(z_samples.shape[0], z_samples.shape[2])
                probs = torch.exp(torch.log(tmp_probs).mean(axis=0))
            elif(infer_type == 'avgprob'):
                attractors = z_samples
                tmp_probs = torch.sigmoid(
                    torch.flatten(self.counter.to(self.device)(z_samples))).reshape(z_samples.shape[0], z_samples.shape[2])
                probs = tmp_probs.mean(axis=0)
        return attractors, probs, z_mu, z_logvar, z_samples

    def __call__(
        self,
        xs: torch.Tensor,
        n_speakers: List[int],
	num_zsamples: int,
        use_mu: bool = False,
        temp:str = 'None',
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate attractors and loss from embedding sequences
        with given number of speakers
        Args:
          xs: List of (T,D)-shaped embeddings
          n_speakers: List of number of speakers, or None if the number
                                of speakers is unknown (ex. test phase)
	  num_zsamples: int
	  use_mu: bool
        Returns:
          loss: Attractor existence loss
          attractors: List of (N,D)-shaped attractors
        """

        max_n_speakers = max(n_speakers)
        if self.device == torch.device("cpu"):
            zeros = torch.zeros(
                (xs.shape[0], max_n_speakers + 1, self.n_units))
            labels = torch.from_numpy(np.asarray([
                [1.0] * n_spk + [0.0] * (1 + max_n_speakers - n_spk)
                for n_spk in n_speakers]))
        else:
            zeros = torch.zeros(
                (xs.shape[0], max_n_speakers + 1, self.n_units),
                device=torch.device("cuda"))
            labels = torch.from_numpy(np.asarray([
                [1.0] * n_spk + [0.0] * (1 + max_n_speakers - n_spk)
                for n_spk in n_speakers])).to(torch.device("cuda")) #[B, S+1]

        z_mu, z_logvar, z_KLD_loss = self.forward(xs, zeros, temp) #xs: [B, T, D], mu,logvar,zeros:[B, K+1, D]

        z_samples = self.mul_sampling(z_mu, z_logvar, num_zsamples, use_mu) #[M, B, S+1, D]
        if (z_samples.ndim == 3): z_samples = z_samples.unsqueeze(0) 
        #[M, B, S+1, D]

        #average sample
        if(self.average_type == 'sample'):
            raise ValueError('no mearningful, average sample is not use anymore.')
            z_samples = torch.mean(z_samples, axis=0).unsqueeze(0) #keep [M=1, B, S+1, D] 

        logit = self.counter(z_samples).squeeze(-1) #[M, B, S+1]	
        label_samples = labels.unsqueeze(0).repeat(logit.shape[0], 1, 1)	
        if(torch.sum(torch.isnan(logit))): print("find nan in L283: %s" %logit)    
        if(self.average_type in ['loss']):
            loss = F.binary_cross_entropy_with_logits(logit,
                label_samples, reduction='mean') 
        elif(self.average_type == 'prob'):
            m = torch.nn.Sigmoid()            		
            loss = F.binary_cross_entropy(torch.mean(m(logit), axis=0),
               labels.float(), reduction='mean') #[B, S+1]  


        return loss, z_mu[:,:-1,:], z_logvar[:,:-1,:], z_samples[:,:,:-1,:], z_KLD_loss 
        ## The final attractor does not correspond to a speaker so remove it
        #attractors = attractors[:, :-1, :]
        # z_mu [B, S+1, D], z_logvar [B, S+1, D], z_samples: ``[M, B, S+1, D]

    def test_speed_sampling(self, nz: int):
        import time	    
        z_mu = torch.randn(32, 3, 256).to(self.device)	    
        z_logvar = torch.randn(32, 3, 256).to(self.device)	    

        st=time.time()	    
        z_samples = torch.cat([
            self.sampling(z_mu, z_logvar).unsqueeze(0)
            for i in range(nz)
            ])  
        print("before: %s s" %(time.time() - st ))	
        print(z_samples.shape)	

        st=time.time()	    
        z_samples = self.mul_sampling(z_mu, z_logvar, nz) #[M, B, S+1, D]
        print("after: %s s" %(time.time() - st ))	
        print(z_samples.shape)	

class MultiHeadSelfAttention(Module):
    """ Multi head self-attention layer
    """
    def __init__(
        self,
        device: torch.device,
        n_units: int,
        h: int,
        dropout: float
    ) -> None:
        super(MultiHeadSelfAttention, self).__init__()
        self.device = device
        self.linearQ = torch.nn.Linear(n_units, n_units, device=self.device)
        self.linearK = torch.nn.Linear(n_units, n_units, device=self.device)
        self.linearV = torch.nn.Linear(n_units, n_units, device=self.device)
        self.linearO = torch.nn.Linear(n_units, n_units, device=self.device)
        self.d_k = n_units // h
        self.h = h
        self.dropout = dropout
        self.att = None  # attention for plot

    def __call__(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        # x: (BT, F)
        q = self.linearQ(x).reshape(batch_size, -1, self.h, self.d_k)
        k = self.linearK(x).reshape(batch_size, -1, self.h, self.d_k)
        v = self.linearV(x).reshape(batch_size, -1, self.h, self.d_k)
        scores = torch.matmul(q.permute(0, 2, 1, 3), k.permute(0, 2, 3, 1)) \
            / np.sqrt(self.d_k)
        # scores: (B, h, T, T)
        self.att = F.softmax(scores, dim=3)
        p_att = F.dropout(self.att, self.dropout)
        x = torch.matmul(p_att, v.permute(0, 2, 1, 3))
        x = x.permute(0, 2, 1, 3).reshape(-1, self.h * self.d_k)
        return self.linearO(x)


class PositionwiseFeedForward(Module):
    """ Positionwise feed-forward layer
    """
    def __init__(
        self,
        device: torch.device,
        n_units: int,
        d_units: int,
        dropout: float,
    ) -> None:
        super(PositionwiseFeedForward, self).__init__()
        self.device = device
        self.linear1 = torch.nn.Linear(n_units, d_units, device=self.device)
        self.linear2 = torch.nn.Linear(d_units, n_units, device=self.device)
        self.dropout = dropout

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(F.dropout(F.relu(self.linear1(x)), self.dropout))


class TransformerEncoder(Module):
    def __init__(
        self,
        device: torch.device,
        idim: int,
        n_layers: int,
        n_units: int,
        e_units: int,
        h: int,
        dropout: float
    ) -> None:
        super(TransformerEncoder, self).__init__()
        self.device = device
        self.linear_in = torch.nn.Linear(idim, n_units, device=self.device)
        self.lnorm_in = torch.nn.LayerNorm(n_units, device=self.device)
        self.n_layers = n_layers
        self.dropout = dropout
        last_double = False
        for i in range(n_layers):
            last_double = (i == (n_layers-1))	
            setattr(
                self,
                '{}{:d}'.format("lnorm1_", i),
                torch.nn.LayerNorm(n_units, device=self.device)
            )
            setattr(
                self,
                '{}{:d}'.format("self_att_", i),
                MultiHeadSelfAttention(self.device, n_units, h, dropout)
            )
            setattr(
                self,
                '{}{:d}'.format("lnorm2_", i),
                torch.nn.LayerNorm(n_units, device=self.device)
            )
            setattr(
                self,
                '{}{:d}'.format("ff_", i),
                PositionwiseFeedForward(self.device, n_units, e_units, dropout)
            )
        self.lnorm_out = torch.nn.LayerNorm(n_units, device=self.device)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F) ... batch, time, (mel)freq
        BT_size = x.shape[0] * x.shape[1]
        # e: (BT, F)
        e = self.linear_in(x.reshape(BT_size, -1))
        # Encoder stack
        for i in range(self.n_layers):
            # layer normalization
            e = getattr(self, '{}{:d}'.format("lnorm1_", i))(e)
            # self-attention
            s = getattr(self, '{}{:d}'.format("self_att_", i))(e, x.shape[0])
            # residual
            e = e + F.dropout(s, self.dropout)
            # layer normalization
            e = getattr(self, '{}{:d}'.format("lnorm2_", i))(e)
            # positionwise feed-forward
            s = getattr(self, '{}{:d}'.format("ff_", i))(e)
            # residual
            e = e + F.dropout(s, self.dropout)
        # final layer normalization
        # output: (BT, F)
        return self.lnorm_out(e)


class TransformerEDADiarization(Module):

    def __init__(
        self,
        device: torch.device,
        in_size: int,
        n_units: int,
        e_units: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
        vad_loss_weight: float,
        attractor_loss_ratio: float,
        attractor_encoder_dropout: float,
        attractor_decoder_dropout: float,
        detach_attractor_loss: bool,
        zkld_loss_weight: list,
        ekld_loss_weight: list,
        average_type: str = 'sample',
    ) -> None:
        """ Self-attention-based diarization model.
        Args:
          in_size (int): Dimension of input feature vector
          n_units (int): Number of units in a self-attention block
          n_heads (int): Number of attention heads
          n_layers (int): Number of transformer-encoder layers
          dropout (float): dropout ratio
          vad_loss_weight (float) : weight for vad_loss
          attractor_loss_ratio (float)
          attractor_encoder_dropout (float)
          attractor_decoder_dropout (float)
          zkld_loss_weight: kld weight for spk-wise latent space of attractor
          ekld_loss_weight: kld weight for frame-wise latent space of emb.
          num_zsamples: (int)
          average_type: (str)
        """
        self.device = device
        super(TransformerEDADiarization, self).__init__()
        self.enc = TransformerEncoder(
            self.device, in_size, n_layers, n_units, e_units, n_heads, dropout
        )
        #zlin modify units -> n_units/2 as we split the emb
        self.eda = EncoderDecoderAttractor(
            self.device,
            n_units,
            attractor_encoder_dropout,
            attractor_decoder_dropout,
            detach_attractor_loss,
            average_type,
        )
        self.attractor_loss_ratio = attractor_loss_ratio
        self.vad_loss_weight = vad_loss_weight
        self.zkld_loss_weight = zkld_loss_weight	
        self.ekld_loss_weight = ekld_loss_weight	


        self.llr_alpha = torch.nn.Parameter(torch.randn(1))
        self.llr_beta = torch.nn.Parameter(torch.randn(1))
        self.linearmu = torch.nn.Linear(n_units, n_units, device=self.device)
        self.linearlogvar = torch.nn.Linear(n_units, n_units, device=self.device)
        self.softplus = torch.nn.Softplus()	

    def mul_sampling(self,
        mu, log_var, num_sample, use_mu=False):
        assert( (use_mu) + (num_sample>0) == 1 )
        if(num_sample < 1 and use_mu):
            return mu.unsqueeze(0)		
        std = torch.exp(0.5 * log_var)
        esp = torch.randn_like(std.unsqueeze(0).repeat(num_sample, 1, 1, 1)) # [M, B, S, D]
        return esp.mul(std).add_(mu) # [M, B, S, D]


    def get_embeddings(self, xs: torch.Tensor, temp='None') -> torch.Tensor:
        ilens = [x.shape[0] for x in xs]
        # xs: (B, T, F)
        pad_shape = xs.shape
        # emb: (B*T, E)
        emb = self.enc(xs)
        # emb: [(T, E), ...]
        emb = emb.reshape(pad_shape[0], pad_shape[1], -1)

        e_mu = self.linearmu(emb)
        e_logvar = self.linearlogvar(emb)
        if(temp == 'softplus'):
            return e_mu, -self.softplus(e_logvar)
        return e_mu, e_logvar

    def estimate_sequential(
        self,
        xs: torch.Tensor,
        args: SimpleNamespace,
        return_ys: bool = False,
    ) -> List[torch.Tensor]:
        assert args.estimate_spk_qty_thr != -1 or \
            args.estimate_spk_qty != -1, \
            "Either 'estimate_spk_qty_thr' or 'estimate_spk_qty' \
            arguments have to be defined."
        e_mu, e_logvar = self.get_embeddings(xs, args.temp) #[B, T, D]

        e_samples = self.mul_sampling(e_mu, e_logvar, args.num_esamples, args.use_mu_e) #[M, B, S+1, D]
        ys_active = []
        if args.time_shuffle:
            orders = [np.arange(e.shape[0]) for e in e_mu]
            for order in orders:
                np.random.shuffle(order)
            attractors, probs, z_mu, z_logvar, z_samples = self.eda.estimate(
                torch.stack([e[order] for e, order in zip(e_mu, orders)]),
                infer_type = args.infer_type,
                temp = args.temp,
		num_zsamples= args.num_zsamples,
		use_mu = args.use_mu_z)
        else:
            attractors, probs, z_mu, z_logvar, z_samples = self.eda.estimate(
                e_mu,
                infer_type = args.infer_type, 
                temp = args.temp,
		num_zsamples= args.num_zsamples,
		use_mu = args.use_mu_z)


        if(args.infer_type == 'logprob'):
            assert attractors.ndim == 4		
            assert e_samples.shape[0] == attractors.shape[0]	    
            ys = torch.exp(torch.log(
                  torch.sigmoid(torch.matmul(e_samples, attractors.permute(0,1,3,2)))).mean(axis=0))[0]
        elif(args.infer_type == 'avgprob'):
            assert attractors.ndim == 4		
            assert e_samples.shape[0] == attractors.shape[0]	    
            ys = torch.sigmoid(torch.matmul(e_samples, attractors.permute(0,1,3,2))).mean(axis=0)[0]
        elif(args.infer_type == 'llr'):
            llr = LLR(z_mu, z_logvar, e_mu, e_logvar, args.temp) #[1, T, S+1]
            if(torch.sum(torch.isnan(llr))): print("find nan: %s" %llr)    
            ys = llr[0]    
        else:		
            ys = torch.matmul(emb, attractors.permute(0, 2, 1))[0]
        y=ys; p=probs
        if args.estimate_spk_qty != -1:
            sorted_p, order = torch.sort(p, descending=True)
            ys_active.append(y[:, order[:args.estimate_spk_qty]])
        elif args.estimate_spk_qty_thr != -1:
            silence = np.where(
                p.data.to("cpu") < args.estimate_spk_qty_thr)[0]
            n_spk = silence[0] if silence.size else None
            ys_active.append(y[:, :n_spk])
        else:
            NotImplementedError(
                'Possible attribute is estimate_spk_qty or estimate_spk_qty_thr.')

        if(return_ys):		
            return ys_active, ys, probs, z_mu, z_logvar, e_mu, e_logvar
        else:
            return ys_active

    def forward(
        self,
        xs: torch.Tensor,
        ts: torch.Tensor,
        n_speakers: List[int],
        args: SimpleNamespace
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        e_mu, e_logvar = self.get_embeddings(xs, args.temp) #[B, T, D]

        if args.time_shuffle:
            orders = [np.arange(e.shape[0]) for e in e_mu]
            for order in orders:
                np.random.shuffle(order)
            attractor_loss, z_mu, z_logvar, z_samples, KLD_loss = self.eda(
                torch.stack([e[order] for e, order in zip(e_mu, orders)]),
                n_speakers, 
		num_zsamples = args.num_zsamples, 
		use_mu = args.use_mu_z,
                temp = args.temp)
        else:
            attractor_loss, z_mu, z_logvar, z_samples, KLD_loss = self.eda(
                e_mu, n_speakers,
		num_zsamples = args.num_zsamples, 
		use_mu = args.use_mu_z,
                temp = args.temp)
    
        e_KLD_loss = get_KLD_loss(e_mu, e_logvar, args.KLD_cal)        
        if(args.zKLD_invalid):
            z_KLD_loss = KLD_loss
        else:
            z_KLD_loss = get_KLD_loss(z_mu, z_logvar, args.KLD_cal)       

        #a = z_samples  #[M, B, S, D]
        # 
        # How to get diarization res. from pairs of attactors and emb.
        if(args.average_type in ['loss', 'sample', 'prob']):
            # For 'loss' and 'sample', sampling is needed.
            # Then same as in the original eend-eda, 
            # the diarization results are calculated using dot products between all pairs of attractors and embeddings
            emb_samples = self.mul_sampling(e_mu, e_logvar, args.num_esamples, args.use_mu_e) #[M, B, T, D] = [3, 32, 500, 256]

            #instead sue dot-product cross all frame and attractors, we can try use the same sample num and get the dot-product. 
            ys_mu = torch.matmul(e_mu, z_mu.permute(0, 2, 1)) # [B, T, D] , [B, D, S] -> [B, T, S]
            #assert emb_samples.shape[0] == z_samples.shape[0]
            assert(emb_samples.shape[0]==1 or z_samples.shape[0] == 1 or emb_samples.shape[0] == z_samples.shape[0])
	    #torch.matmul support boardcast, so we kept the shape[0]==1

            ys = torch.matmul(emb_samples, z_samples.permute(0, 1, 3, 2)) #[N, B, T, D] [M, B, D, S] -> [M, B, T, S]
            return ys_mu, ys, attractor_loss, z_KLD_loss, e_KLD_loss
        else:
            raise NotImplementedError

    def get_loss(
        self,
        ys_mu: torch.Tensor,
        ys: torch.Tensor,
        target: torch.Tensor,
        n_speakers: List[int],
        attractor_loss: torch.Tensor,
        vad_loss_weight: float,
        zkld_loss: torch.Tensor,
        zkld_loss_weight:float,
        ekld_loss: torch.Tensor,
        ekld_loss_weight:float,
        detach_attractor_loss: bool = False,
        permutation_type:str = 'sample',
        average_type: str = 'loss',        
        acfunc: str = 'None',
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        #ys [M, B, T, S]            
        max_n_speakers = max(n_speakers)
        ts_padded = pad_labels(target, max_n_speakers)                        #
        ts_padded = torch.stack(ts_padded)                                #[B, T, S]

        if(average_type == 'llr'):
            ys_padded = pad_labels(ys, max_n_speakers)                 #
            ys_padded = torch.stack(ys_padded)                                #[M, B, T, S]
            loss = pit_loss_multispk_ori(
                ys_padded, ts_padded, 
                n_speakers, detach_attractor_loss,
                acfunc,		
    	    )
        else:
            ys_padded = pad_labels_samples(ys, max_n_speakers) 		#
            ys_padded = torch.stack(ys_padded)				#[M, B, T, S]
            ys_mu_padded = pad_labels(ys_mu, max_n_speakers) 	#
            ys_mu_padded = torch.stack(ys_mu_padded)			#[M, B, T, S]
            loss = pit_loss_multispk(
                ys_mu_padded,			
                ys_padded, ts_padded, 
                n_speakers, detach_attractor_loss,
    	        permutation_type)
        return loss + \
            attractor_loss * self.attractor_loss_ratio +\
            zkld_loss * zkld_loss_weight + ekld_loss * ekld_loss_weight, loss


def pad_labels(ts: torch.Tensor, out_size: int) -> torch.Tensor:
    # pad label's speaker-dim to be model's n_speakers
    ts_padded = []
    for _, t in enumerate(ts):
        if t.shape[1] < out_size:
            # padding
            ts_padded.append(torch.cat((t, -1 * torch.ones((
                t.shape[0], out_size - t.shape[1]))), dim=1))
        elif t.shape[1] > out_size:
            # truncate
            ts_padded.append(t[:, :out_size].float())
        else:
            ts_padded.append(t.float())
    return ts_padded

def pad_labels_samples(ts_samples: torch.Tensor, out_size: int) -> torch.Tensor:
    # pad label's speaker-dim to be model's n_speakers
    ts_samples_padded = []
    if(ts_samples.ndim == 3): ts_samples = ts_samples.unsqueeze(0) #for target [B, T, S] -> [M, B, T, S]

    for idx, ts in enumerate(ts_samples):
        ts_padded = []
        for _, t in enumerate(ts):
            if t.shape[1] < out_size:
                # padding
                ts_padded.append(torch.cat((t, -1 * torch.ones((
                    t.shape[0], out_size - t.shape[1]))), dim=1))
            elif t.shape[1] > out_size:
                # truncate
                ts_padded.append(t[:, :out_size].float())
            else:
                ts_padded.append(t.float())
        ts_samples_padded.append(torch.stack(ts_padded))		
    return ts_samples_padded

def pad_sequence(
    features: List[torch.Tensor],
    labels: List[torch.Tensor],
    seq_len: int
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    features_padded = []
    labels_padded = []
    assert len(features) == len(labels), (
        f"Features and labels in batch were expected to match but got "
        "{len(features)} features and {len(labels)} labels.")
    for i, _ in enumerate(features):
        assert features[i].shape[0] == labels[i].shape[0], (
            f"Length of features and labels were expected to match but got "
            "{features[i].shape[0]} and {labels[i].shape[0]}")
        length = features[i].shape[0]
        if length < seq_len:
            extend = seq_len - length
            features_padded.append(torch.cat((features[i], -torch.ones((
                extend, features[i].shape[1]))), dim=0))
            labels_padded.append(torch.cat((labels[i], -torch.ones((
                extend, labels[i].shape[1]))), dim=0))
        elif length > seq_len:
            raise (f"Sequence of length {length} was received but only "
                   "{seq_len} was expected.")
        else:
            features_padded.append(features[i])
            labels_padded.append(labels[i])
    return features_padded, labels_padded


def save_checkpoint(
    args,
    epoch: int,
    model: Module,
    optimizer: NoamOpt,
    loss: torch.Tensor
) -> None:
    Path(f"{args.output_path}/models").mkdir(parents=True, exist_ok=True)

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss},
        f"{args.output_path}/models/checkpoint_{epoch}.tar"
    )


def load_checkpoint(args: SimpleNamespace, filename: str):
    model = get_model(args)
    optimizer = setup_optimizer(args, model)

    assert isfile(filename), \
        f"File {filename} does not exist."
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return epoch, model, optimizer, loss


def load_initmodel(args: SimpleNamespace):
    return load_checkpoint(args, args.initmodel)


def get_model(args: SimpleNamespace) -> Module:
    if args.model_type == 'TransformerEDA':
        model = TransformerEDADiarization(
            device=args.device,
            in_size=args.feature_dim * (1 + 2 * args.context_size),
            n_units=args.hidden_size,
            e_units=args.encoder_units,
            n_heads=args.transformer_encoder_n_heads,
            n_layers=args.transformer_encoder_n_layers,
            dropout=args.transformer_encoder_dropout,
            attractor_loss_ratio=args.attractor_loss_ratio,
            attractor_encoder_dropout=args.attractor_encoder_dropout,
            attractor_decoder_dropout=args.attractor_decoder_dropout,
            detach_attractor_loss=args.detach_attractor_loss,
            vad_loss_weight=args.vad_loss_weight,
	    zkld_loss_weight=args.zkld_loss_weight,
            ekld_loss_weight=args.ekld_loss_weight,
            average_type = args.average_type,
        )
    else:
        raise ValueError('Possible model_type is "TransformerEDA"')
    return model


def average_checkpoints(
    device: torch.device,
    model: Module,
    models_path: str,
    epochs: str
) -> Module:
    epochs = parse_epochs(epochs)
    states_dict_list = []
    for e in epochs:
        copy_model = copy.deepcopy(model)
        checkpoint = torch.load(join(
            models_path,
            f"checkpoint_{e}.tar"), map_location=device)
        copy_model.load_state_dict(checkpoint['model_state_dict'])
        states_dict_list.append(copy_model.state_dict())
    avg_state_dict = average_states(states_dict_list, device)
    avg_model = copy.deepcopy(model)
    avg_model.load_state_dict(avg_state_dict)
    return avg_model


def average_states(
    states_list: List[Dict[str, torch.Tensor]],
    device: torch.device,
) -> List[Dict[str, torch.Tensor]]:
    qty = len(states_list)
    avg_state = states_list[0]
    for i in range(1, qty):
        for key in avg_state:
            avg_state[key] = avg_state[key].to(device)		
            avg_state[key] += states_list[i][key].to(device)

    for key in avg_state:
        avg_state[key] = avg_state[key] / qty
    return avg_state


def parse_epochs(string: str) -> List[int]:
    parts = string.split(',')
    res = []
    for p in parts:
        if '-' in p:
            interval = p.split('-')
            res.extend(range(int(interval[0])+1, int(interval[1])+1))
        else:
            res.append(int(p))
    return res


    	
