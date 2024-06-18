#!/usr/bin/env python3

# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Copyright 2022 Brno University of Technology (authors: Federico Landini, Lukas Burget, Mireia Diez)
# Copyright 2022 AUDIAS Universidad Autonoma de Madrid (author: Alicia Lozano-Diez)
# Copyright 2024 Brno University of Technology, National Institute of Informatics (authors: Lin Zhang)
# Licensed under the MIT license.

from itertools import permutations
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple
from torch.nn.functional import logsigmoid
from scipy.optimize import linear_sum_assignment


def pit_loss_multispk_ori(
        logits: List[torch.Tensor], target: List[torch.Tensor],
        n_speakers: np.ndarray, detach_attractor_loss: bool,
	acfunc: str = 'sigmoid'):
    if detach_attractor_loss:
        # -1's for speakers that do not have valid attractor
        for i in range(target.shape[0]):
            target[i, :, n_speakers[i]:] = -1 * torch.ones(
                          target.shape[1], target.shape[2]-n_speakers[i])

    logits_t = logits.detach().transpose(1, 2)
    cost_mxs = -logsigmoid(logits_t).bmm(target) - logsigmoid(-logits_t).bmm(1-target)

    max_n_speakers = max(n_speakers)

    for i, cost_mx in enumerate(cost_mxs.cpu().numpy()):
        if max_n_speakers > n_speakers[i]:
            max_value = np.absolute(cost_mx).sum()
            cost_mx[-(max_n_speakers-n_speakers[i]):] = max_value
            cost_mx[:, -(max_n_speakers-n_speakers[i]):] = max_value
        pred_alig, ref_alig = linear_sum_assignment(cost_mx)
        assert (np.all(pred_alig == np.arange(logits.shape[-1])))
        target[i, :] = target[i, :, ref_alig]
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
         logits, target, reduction='none')
    loss[torch.where(target == -1)] = 0
    # normalize by sequence length
    loss = torch.sum(loss, axis=1) / (target != -1).sum(axis=1)
    for i in range(target.shape[0]):
        loss[i, n_speakers[i]:] = torch.zeros(loss.shape[1]-n_speakers[i])

    # normalize in batch for all speakers
    loss = torch.mean(loss)
    return loss

def pit_loss_multispk(
        logits_mu: List[torch.Tensor],		
        logits_samples: List[torch.Tensor], 
	target: List[torch.Tensor],
        n_speakers: np.ndarray, detach_attractor_loss: bool,
	permutation_type: str):
    """    
    logits_mu: [B, T, S], dot-product between emb and mu.
    logits_samples: [M, B, T, S], dot-product between emb and samples.
    target: [B, T, S]
    permutation_type: use mu to find the best mapping or use samples find best map for each.
    """
    max_n_speakers = max(n_speakers)
    #loss_samples=[]
    if detach_attractor_loss:
        # -1 for speakers that do not have valid attractor
        for i in range(target.shape[0]):
            target[i, :, n_speakers[i]:] = -1 * torch.ones(
                          target.shape[1], target.shape[2]-n_speakers[i])

    if(permutation_type == 'mu'):	    
        #1. utilize mu to find best mapping (permutation)	    
        logits_t = logits_mu.detach().transpose(1, 2)
        cost_mxs = -logsigmoid(logits_t).bmm(target) - logsigmoid(-logits_t).bmm(1-target) # [B, max_S, max_S]

        for i, cost_mx in enumerate(cost_mxs.cpu().numpy()):
            if max_n_speakers > n_speakers[i]:
                max_value = np.absolute(cost_mx).sum()
                cost_mx[-(max_n_speakers-n_speakers[i]):] = max_value
                cost_mx[:, -(max_n_speakers-n_speakers[i]):] = max_value
            pred_alig, ref_alig = linear_sum_assignment(cost_mx)
            assert (np.all(pred_alig == np.arange(logits_mu.shape[-1])))
            target[i, :] = target[i, :, ref_alig]

        #2. based on the above mapping, calculate pit loss for multiple samples.
        with torch.no_grad():
            target_ = target.unsqueeze(0).repeat(logits_samples.shape[0], 1, 1, 1)	 #[M, B, T, S]    

    elif(permutation_type == 'sample'):
        cost_mxs_samples = torch.einsum('mbds, bdk -> mbsk', -logsigmoid(logits_samples), target) - \
                           torch.einsum('mbds, bdk -> mbsk', logsigmoid(-logits_samples), 1-target) 
        #[M, B, max_S, max_S]		       

        target_ = torch.zeros(logits_samples.permute(1,2,0,3).shape, device = logits_samples.device) #[B, T, M, S]    
        for i, cost_mx_oneb in enumerate(cost_mxs_samples.permute(1,0,2,3).cpu().detach().numpy()):
            if max_n_speakers > n_speakers[i]:
                max_value = np.absolute(cost_mx_oneb.mean(axis=0)).sum()
                cost_mx_oneb[:, -(max_n_speakers-n_speakers[i]):,:] = max_value
                cost_mx_oneb[:, :, -(max_n_speakers-n_speakers[i]):] = max_value
            
            best_map=np.stack([linear_sum_assignment(cost_mx) for cost_mx in cost_mx_oneb])
            pred_alig = best_map[:,0,:] #[M, max_spk]
            ref_alig = best_map[:,1,:] #[M, max_spk]

            assert (np.all(pred_alig == np.arange(logits_samples[:,i,:,:].shape[-1])))
            with torch.no_grad():
                target_[i, :, :] = target[i, :, ref_alig].clone()
        with torch.no_grad():
            target_ = target_.permute(2, 0, 1, 3) #back to [M, B, T, S]


    loss_samples = torch.nn.functional.binary_cross_entropy_with_logits(
         logits_samples, target_, reduction='none')  #[M, B, T, S]

    loss_samples[torch.where(target_ == -1)] = 0

    # normalize by sequence length
    loss_samples = torch.sum(loss_samples, axis=2) / (target_ != -1).sum(axis=2) #do on T-axis [M, B, S]

    #do not use broadcast, which will produce wrong res.

    for i in range(target_.shape[1]):
        loss_samples[:, i, n_speakers[i]:] = torch.zeros(loss_samples.shape[0], loss_samples.shape[2]-n_speakers[i])

    # normalize in batch and samples (if have) for all speakers
    #return torch.mean(loss_samples)  #
    return torch.mean(loss_samples[loss_samples!=0])  



def vad_loss(ys: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
    # Take from reference ts only the speakers that do not correspond to -1
    # (-1 are padded frames), if the sum of their values is >0 there is speech
    vad_ts = (torch.sum((ts != -1)*ts, 2, keepdim=True) > 0).float()
    # We work on the probability space, not logits. We use silence probabilities
    ys_silence_probs = 1-torch.sigmoid(ys)
    # The probability of silence in the frame is the product of the
    # probability that each speaker is silent
    silence_prob = torch.prod(ys_silence_probs, 2, keepdim=True)
    # Estimate the loss. size=[batch_size, num_frames, 1]
    loss = F.binary_cross_entropy(silence_prob, 1-vad_ts, reduction='none')
    # "torch.max(ts, 2, keepdim=True)[0]" keeps the maximum along speaker dim
    # Invalid frames in the sequence (padding) will be -1, replace those
    # invalid positions by 0 so that those losses do not count
    loss[torch.where(torch.max(ts, 2, keepdim=True)[0] < 0)] = 0
    # normalize by sequence length
    # "torch.sum(loss, axis=1)" gives a value per batch
    # if torch.mean(ts,axis=2)==-1 then all speakers were invalid in the frame,
    # therefore we should not account for it
    # ts is size [batch_size, num_frames, num_spks]
    loss = torch.sum(loss, axis=1) / (torch.mean(ts, axis=2) != -1).sum(axis=1, keepdims=True)
    # normalize in batch for all speakers
    loss = torch.mean(loss)
    return loss

