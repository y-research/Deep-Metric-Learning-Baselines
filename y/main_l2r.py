#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Created by Hai-Tao Yu | 2020/02/13 | https://y-research.github.io

# Copyright 2019 Karsten Roth and Biagio Brattoli
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


#################### LIBRARIES ########################

import warnings
warnings.filterwarnings("ignore")

import os, sys, numpy as np, argparse, imp, datetime, time, pickle as pkl, random, json
os.chdir(os.path.dirname(os.path.realpath(__file__)))

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from tqdm import tqdm
import pandas as pd

import torch, torch.nn as nn
import auxiliaries as aux
import datasets as data

import netlib as netlib
import losses as losses
import evaluate as eval

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from y.l2r_global import global_device as device, gpu_id

"""============================================================================"""
#################### TRAINER FUNCTION ############################
def train_one_epoch(train_dataloader, model, optimizer, criterion, opt, epoch, grad_measure, LOG):
    """
    This function is called every epoch to perform training of the network over one full
    (randomized) iteration of the dataset.

    Args:
        train_dataloader: torch.utils.data.DataLoader, returns (augmented) training data.
        model:            Network to train.
        optimizer:        Optimizer to use for training.
        criterion:        criterion to use during training.
        opt:              argparse.Namespace, Contains all relevant parameters.
        epoch:            int, Current epoch.

    Returns:
        Nothing!
    """
    loss_collect = []

    start = time.time()

    data_iterator = tqdm(train_dataloader, desc='Epoch {} Training...'.format(epoch))
    for i,(class_labels, input) in enumerate(data_iterator):
        # print(i, class_labels)
        #Compute embeddings for input batch.
        # print('input', input)
        features  = model(input.to(opt.device))
        # print('feature', features)
        #Compute loss.
        loss      = criterion(features, class_labels)

        #Ensure gradients are set to zero at beginning
        optimizer.zero_grad()
        #Compute gradients.
        loss.backward()

        if opt.grad_measure:
            #If desired, save computed gradients.
            grad_measure.include(model.model.last_linear)

        #Update weights using comp. gradients.
        optimizer.step()

        #Store loss per iteration.
        loss_collect.append(loss.item())
        if i==len(train_dataloader)-1: data_iterator.set_description('Epoch (Train) {0}: Mean Loss [{1:.4f}]'.format(epoch, np.mean(loss_collect)))

    #Save metrics
    LOG.log('train', LOG.metrics_to_log['train'], [epoch, np.round(time.time()-start,4), np.mean(loss_collect)])

    if opt.grad_measure:
        #Dump stored gradients to Pickle-File.
        grad_measure.dump(epoch)





def run(opt=None):

    if opt.dataset=='online_products': # recall cutoff values
        opt.k_vals = [1,10,100,1000]
    elif opt.dataset=='in-shop':
        opt.k_vals = [1,10,20,30,50]
    elif opt.dataset=='vehicle_id':
        opt.k_vals = [1,5]

    if opt.loss == 'fastap':
        if opt.dataset=='online_products':
            opt.batches_per_super_pair = 10
        elif opt.dataset=='in-shop':
            opt.batches_per_super_pair = 4

    if opt.loss == 'proxynca':
        opt.samples_per_class = 1

    if opt.samples_per_class == 0:
        print("samples_per_class=0, not enforced (few-shot mode)")

    # if opt.loss == 'BetaMinusOriginSwapPrecision':
    #     opt.samples_per_class = 5

    else:
        assert opt.samples_per_class > 0
        assert not opt.bs%opt.samples_per_class, 'Batchsize needs to fit number of samples per class for distance sampling and margin/triplet loss!'

    if opt.loss == 'npair' or opt.loss == 'proxynca' or opt.loss == 'fastap':
        opt.sampling = 'None'

    opt.pretrained = not opt.not_pretrained


    """============================================================================"""
    ################### GPU SETTINGS ###########################
    os.environ["CUDA_DEVICE_ORDER"]   ="PCI_BUS_ID"
    #os.environ["CUDA_VISIBLE_DEVICES"]= str(opt.gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)


    """============================================================================"""
    #################### SEEDS FOR REPROD. #####################
    torch.backends.cudnn.deterministic=True
    np.random.seed(opt.seed); random.seed(opt.seed)
    torch.manual_seed(opt.seed); torch.cuda.manual_seed(opt.seed); torch.cuda.manual_seed_all(opt.seed)


    """============================================================================"""
    ##################### NETWORK SETUP ##################
    #opt.device = torch.device('cuda')
    #opt.device = device
    #Depending on the choice opt.arch, networkselect() returns the respective network model
    model      = netlib.networkselect(opt)

    print('{} Setup for {} with {} sampling on {} complete with #weights: {}'.format(opt.loss.upper(), opt.arch.upper(), opt.sampling.upper(), opt.dataset.upper(), aux.gimme_params(model)))

    #Push to Device
    _          = model.to(opt.device)
    #Place trainable parameter in list of parameters to train:
    #if 'fc_lr_mul' in vars(opt).keys() and opt.fc_lr_mul!=0:
    #    all_but_fc_params = filter(lambda x: 'last_linear' not in x[0],model.named_parameters())
    #    fc_params         = model.model.last_linear.parameters()
    #    to_optim          = [{'params':all_but_fc_params,'lr':opt.lr,'weight_decay':opt.decay},
    #                         {'params':fc_params,'lr':opt.lr*opt.fc_lr_mul,'weight_decay':opt.decay}]
    #    print(to_optim)
    #    pdb.set_trace()
    #else:
    #    to_optim   = [{'params':model.parameters(),'lr':opt.lr,'weight_decay':opt.decay}]
    to_optim = model.to_optim(opt)




    """============================================================================"""
    #################### DATALOADER SETUPS ##################
    #Returns a dictionary containing 'training', 'testing', and 'evaluation' dataloaders.
    #The 'testing'-dataloader corresponds to the validation set, and the 'evaluation'-dataloader
    #Is simply using the training set, however running under the same rules as 'testing' dataloader,
    #i.e. no shuffling and no random cropping.
    dataloaders      = data.give_dataloaders(opt.dataset, opt)
    #Because the number of supervised classes is dataset dependent, we store them after
    #initializing the dataloader
    opt.num_classes  = len(dataloaders['training'].dataset.avail_classes)


    """============================================================================"""
    #################### LOSS SETUP ####################
    # Depending on opt.loss and opt.sampling, the respective criterion is returned,
    # and if the loss has trainable parameters, to_optim is appended.
    criterion, to_optim = losses.loss_select(opt.loss, opt, to_optim)

    if opt.loss in ['SwapPrecision', 'RankAwareSwapPrecision', 'RankAwareMargin', 'BetaMinusOriginSwapPrecision' , 'BetaKMeanSepSwapPrecision' , 'BetaMinusMeanSwapPrecision', 'SwapPrecisionPaper']:
        opt.save_path = os.path.join(opt.save_path, criterion.get_para_str())

    print('opt.save_path', opt.save_path)

    _ = criterion.to(opt.device)


    """============================================================================"""
    #################### CREATE LOGGING FILES ###############
    #Each dataset usually has a set of standard metrics to log. aux.metrics_to_examine()
    #returns a dict which lists metrics to log for training ('train') and validation/testing ('val')

    metrics_to_log = aux.metrics_to_examine(opt.dataset, opt.k_vals)
    # example output: {'train': ['Epochs', 'Time', 'Train Loss', 'Time'],
    #                  'val': ['Epochs','Time','NMI','F1', 'Recall @ 1','Recall @ 2','Recall @ 4','Recall @ 8']}

    #Using the provided metrics of interest, we generate a LOGGER instance.
    #Note that 'start_new' denotes that a new folder should be made in which everything will be stored.
    #This includes network weights as well.
    LOG = aux.LOGGER(opt, metrics_to_log, name='Base', start_new=True)
    #If graphviz is installed on the system, a computational graph of the underlying
    #network will be made as well.
    try:
        aux.save_graph(opt, model)
    except:
        print('Cannot generate network graph!')



    """============================================================================"""
    ##################### OPTIONAL EVALUATIONS #####################
    #Store the averaged gradients returned from the embedding to the last conv. layer.
    if opt.grad_measure:
        grad_measure = eval.GradientMeasure(opt, name='baseline')
    else:
        grad_measure = None

    #Store the relative distances between average intra- and inter-class distance.
    if opt.dist_measure:
        #Add a distance measure for training distance ratios
        distance_measure = eval.DistanceMeasure(dataloaders['evaluation'], opt, name='Train', update_epochs=1)
        # #If uncommented: Do the same for the test set
        # distance_measure_test = eval.DistanceMeasure(dataloaders['testing'], opt, name='Train', update_epochs=1)


    """============================================================================"""
    #################### OPTIM SETUP ####################
    #As optimizer, Adam with standard parameters is used.
    optimizer    = torch.optim.Adam(to_optim)

    if opt.scheduler=='exp':
        scheduler    = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=opt.gamma)
    elif opt.scheduler=='step':
        scheduler    = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.tau, gamma=opt.gamma)
    elif opt.scheduler=='none':
        print('No scheduling used!')
    else:
        raise Exception('No scheduling option for input: {}'.format(opt.scheduler))


    """============================================================================"""
    """========================== MAIN TRAINING PART =============================="""
    """============================================================================"""
    ################### SCRIPT MAIN ##########################
    print('\n-----\n')
    if opt.eval_only:
        opt.n_epochs = 1
        print("Evaluation-only mode!")

    for epoch in range(opt.n_epochs):
        if epoch%3 == 0:
            print(f"GPU:{opt.gpu}, dataset:{opt.dataset}, arch:{opt.arch}, embed_dim:{opt.embed_dim}, embed_init:{opt.embed_init}")
            print(f"loss:{opt.loss}, sampling:{opt.sampling}, samples_per_class:{opt.samples_per_class}, resize256:{opt.resize256}")
            print(f"bs:{opt.bs}, lr:{opt.lr}, fc_lr_mul:{opt.fc_lr_mul}, decay:{opt.decay}, gamma:{opt.gamma}, tau:{opt.tau}, bnft:{opt.ft_batchnorm}")

        ### Print current learning rates for all parameters
        if not opt.eval_only and opt.scheduler!='none':
            print('Running with learning rates {}...'.format(' | '.join('{}'.format(x) for x in scheduler.get_lr())))

        ### Train one epoch
        if not opt.eval_only:
            _ = model.train()
            train_one_epoch(dataloaders['training'], model, optimizer, criterion, opt, epoch, grad_measure, LOG)

        ### Evaluate
        _ = model.eval()
        #Each dataset requires slightly different dataloaders.
        if opt.dataset in ['cars196', 'CUB_200_2011', 'online_products']:
            eval_params = {'dataloader':dataloaders['testing'], 'model':model, 'opt':opt, 'epoch':epoch}
        elif opt.dataset=='in-shop':
            eval_params = {'query_dataloader':dataloaders['testing_query'], 'gallery_dataloader':dataloaders['testing_gallery'], 'model':model, 'opt':opt, 'epoch':epoch}
        elif opt.dataset=='vehicle_id':
            eval_params = {'dataloaders':[dataloaders['testing_set1'], dataloaders['testing_set2'], dataloaders['testing_set3']], 'model':model, 'opt':opt, 'epoch':epoch}

        #Compute Evaluation metrics, print them and store in LOG.
        eval.evaluate(opt.dataset, LOG, save=True, **eval_params)

        #Update the Metric Plot and save it.
        LOG.update_info_plot()

        #(optional) compute ratio of intra- to interdistances.
        if opt.dist_measure:
            distance_measure.measure(model, epoch)
            # distance_measure_test.measure(model, epoch)

        ### Learning Rate Scheduling Step
        if not opt.eval_only and opt.scheduler != 'none':
            scheduler.step()

        print('\n-----\n')



if __name__ == '__main__':

    # for k in [5, 10]:# k=5, bs=60 RuntimeError: CUDA out of memory , k=5 bs=70 mada
    #     for bs in [100]:
    for bs in [30]:#, 30]:
        for ds in ['CUB_200_2011', 'cars196', 'online_products']:#
            for lossf in ['ListNet']: #RankNet']:#,
                if ds == 'cars196' or ds == 'online_products':
                    root_dataset = '/media/dl-box/f12286fd-f13c-4fe0-a92d-9f935d6a7dbd/CVPR' #'/home/dl-box/WorkBench/Datasets/CVPR/'
                elif ds == 'CUB_200_2011':
                    root_dataset = '/home/dl-box/WorkBench/Pool/CUB200-2011/'
                root_output  = '/media/dl-box/f12286fd-f13c-4fe0-a92d-9f935d6a7dbd/Project_output/Out_img_metric/DML_L2R/'#ResNet_norm/'
                ################### INPUT ARGUMENTS ###################
                parser = argparse.ArgumentParser()
                parser.add_argument('--seed', default=135, type=int, help='Random seed for reproducibility.')

                ##### In/Out Parameters
                parser.add_argument('--gpu', default=0, type=int, help='GPU-id for GPU to use.') # now controlled by l2r_global
                parser.add_argument('--init_pth', default=None, type=str)
                parser.add_argument('--savename', default='', type=str, help='Save folder name if any special information is to be included.')
                ### Paths to datasets and storage folder
                parser.add_argument('--source_path', default=os.getcwd() + '/Datasets', type=str, help='Path to training data.')
                parser.add_argument('--save_path', default=os.getcwd() + '/Training_Results', type=str, help='Where to save everything.')

                ##### Evaluation Settings
                parser.add_argument('--k_vals', nargs='+', default=[1, 2, 4, 8], type=int, help='Recall @ Values.')
                parser.add_argument('--eval_only', action='store_true', help='If added, only evaluate model.')

                ##### Network parameters
                parser.add_argument('--embed_dim', default=128, type=int,
                                    help='Embedding dimensionality of the network. Note: in literature, dim=128 is used for ResNet50 and dim=512 for GoogLeNet.')
                parser.add_argument('--embed_init', default='default', type=str,
                                    help='Embedding layer initialization method: {default, kaiming_normal, kaiming_uniform, normal}')
                parser.add_argument('--arch', default='resnet50', type=str, help='Network backend choice: resnet50, googlenet.')
                parser.add_argument('--resize256', action='store_true', help='If added, resize training images to 256x256 first.')
                parser.add_argument('--ft_batchnorm', action='store_true',
                                    help='If added, BatchNorm layers will be un-frozen for finetuning.')
                parser.add_argument('--not_pretrained', action='store_true',
                                    help='If added, the network will be trained WITHOUT ImageNet-pretrained weights.')
                parser.add_argument('--grad_measure', action='store_true',
                                    help='If added, gradients passed from embedding layer to the last conv-layer are stored in each iteration.')
                parser.add_argument('--dist_measure', action='store_true',
                                    help='If added, the ratio between intra- and interclass distances is stored after each epoch.')

                ####### Main Parameter: Dataset to use for Training
                parser.add_argument('--dataset', default=ds, type=str, help='Dataset to use.') #online_products, cars196, CUB_200_2011

                ### General Training Parameters

                parser.add_argument('--fc_lr_mul', default=0, type=float, help='OPTIONAL: Multiply the embedding layer learning rate by this value. If set to 0, the embedding layer shares the same learning rate.')

                parser.add_argument('--n_epochs', default=100, type=int, help='Number of training epochs.')
                parser.add_argument('--bs', default=bs, type=int, help='Mini-Batchsize to use. default=100')#100

                parser.add_argument('--samples_per_class', default=10, type=int, help='Number of samples in one class drawn before choosing the next class. Set to >1 for losses other than ProxyNCA.')

                parser.add_argument('--kernels', default=8, type=int, help='Number of workers for pytorch dataloader.')

                parser.add_argument('--lr', default=0.00001, type=float, help='Learning Rate for network parameters.')
                parser.add_argument('--scheduler', default='step', type=str, help='Type of learning rate scheduling. Currently: step & exp.')
                parser.add_argument('--decay', default=0.0004, type=float, help='Weight decay for optimizer.')
                parser.add_argument('--gamma', default=0.3, type=float, help='Learning rate reduction after tau epochs.')
                parser.add_argument('--tau', default=[40, 80], nargs='+', type=int, help='Stepsize(s) before reducing learning rate.')

                ##### Loss-specific Settings
                #parser.add_argument('--loss', default='marginloss', type=str, help='loss options: marginloss, triplet, npair, proxynca')
                # test 'SwapPrecision', 'RankAwareSwapPrecision', 'RankAwareMargin' , 'BetaMinusOriginSwapPrecision' , 'BetaKMeanSepSwapPrecision' , 'BetaMinusMeanSwapPrecision' , 'SwapPrecisionPaper'
                parser.add_argument('--loss', default=lossf, type=str, help='loss options: marginloss, triplet, npair, proxynca, RankAwareMargin')

                parser.add_argument('--anchor_id', default='Anchor', type=str, help='anchor-wise list or class-wise list')
                parser.add_argument('--sampling', default='distance', type=str, help='For triplet-based losses: Modes of Sampling: random, semihard, distance.')

                ### MarginLoss
                parser.add_argument('--margin', default=0.1, type=float, help='TRIPLET/MARGIN: Margin for Triplet-based Losses')
                parser.add_argument('--beta_lr', default=0.0005, type=float, help='MARGIN: Learning Rate for class margin parameters in MarginLoss')
                parser.add_argument('--beta', default=1.2, type=float, help='MARGIN: Initial Class Margin Parameter in Margin Loss')
                parser.add_argument('--nu', default=0, type=float, help='MARGIN: Regularisation value on betas in Margin Loss.')
                parser.add_argument('--beta_constant', action='store_true', help='MARGIN: Use constant, un-trained beta.')

                ### ProxyNCA
                parser.add_argument('--proxy_lr', default=0.00001, type=float, help='PROXYNCA: Learning Rate for Proxies in ProxyNCALoss.')
                ### NPair L2 Penalty
                parser.add_argument('--l2npair', default=0.02, type=float, help='NPAIR: Penalty-value for non-normalized N-PAIR embeddings.')

                ### FastAP
                parser.add_argument('--histbins', default=10, type=int, help='FASTAP: Number of histogram bins in distance quantization.')
                parser.add_argument('--batches_per_super_pair', default=5, type=int, help='FASTAP: Number sampled batches per pair of super-labels.')

                ### Precision
                # parser.add_argument('--pk', default=k, type=int, help='Cutoff value w.r.t. precision optimization.') #5, 10


                ##### Read in parameters
                opt = parser.parse_args()

                """============================================================================"""
                #opt.source_path += '/' + opt.dataset
                #opt.save_path += '/' + opt.dataset

                opt.source_path = os.path.join(root_dataset, opt.dataset)
                opt.save_path = os.path.join(root_output, opt.dataset)

                opt.gpu, opt.device = gpu_id, device

                if opt.loss == 'listloss':
                    debug = False
                    pass

                elif opt.loss == 'ppp':
                    pass

                else:
                    run(opt=opt)
