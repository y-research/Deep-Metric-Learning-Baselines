#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Haitao Yu on 10/07/2018

"""Description

"""

import torch

#####################
# Global Attributes
#####################

""" Seed """
l2r_seed = 137

""" A Small Value """
epsilon  = 1e-8


""" GPU Setting If Expected """

#global_gpu, global_device, gpu_id = False, 'cpu', None
global_gpu, global_device, gpu_id = True, 'cuda:0', 0
#global_gpu, global_device, gpu_id = True, 'cuda:1', 1

#
if global_gpu: torch.cuda.set_device(gpu_id)

# a uniform tensor type
tensor      = torch.cuda.FloatTensor if global_gpu else torch.FloatTensor
byte_tensor = torch.cuda.ByteTensor  if global_gpu else torch.ByteTensor
long_tensor = torch.cuda.LongTensor  if global_gpu else torch.LongTensor

# uniform constants
torch_one, torch_half, torch_zero = tensor([1.0]), tensor([0.5]), tensor([0.0])
torch_two = tensor([2.0])

torch_minus_one = tensor([-1.0])



