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


############################ LIBRARIES ######################################
import torch, os, numpy as np

import torch.nn as nn
import pretrainedmodels as ptm

import pretrainedmodels.utils as utils
import torchvision.models as models

import googlenet
import math



"""============================================================="""
def initialize_weights(model):
    """
    Function to initialize network weights.
    NOTE: NOT USED IN MAIN SCRIPT.

    Args:
        model: PyTorch Network
    Returns:
        Nothing!
    """
    for idx,module in enumerate(model.modules()):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(0,0.01)
            module.bias.data.zero_()



"""=================================================================================================================================="""
### ATTRIBUTE CHANGE HELPER
def rename_attr(model, attr, name):
    """
    Rename attribute in a class. Simply helper function.

    Args:
        model:  General Class for which attributes should be renamed.
        attr:   str, Name of target attribute.
        name:   str, New attribute name.
    """
    setattr(model, name, getattr(model, attr))
    delattr(model, attr)


"""=================================================================================================================================="""
### NETWORK SELECTION FUNCTION
def networkselect(opt):
    """
    Selection function for available networks.

    Args:
        opt: argparse.Namespace, contains all training-specific training parameters.
    Returns:
        Network of choice
    """
    if opt.arch == 'googlenet':
        network =  GoogLeNet(opt)
    elif opt.arch == 'resnet50':
        network =  ResNet50(opt)
    elif opt.arch == 'resnet50_mcn':
        network =  ResNet50_mcn(opt)
    else:
        raise Exception('Network {} not available!'.format(opt.arch))

    if opt.init_pth:
        network.load_pth(opt.init_pth)
        print("Loaded: ", opt.init_pth)
        return network

    # initialize embedding layer
    if opt.embed_init == 'kaiming_normal':
        torch.nn.init.kaiming_normal_(network.model.last_linear.weight, mode='fan_in', nonlinearity='relu')
    elif opt.embed_init == 'kaiming_uniform':
        torch.nn.init.kaiming_uniform_(network.model.last_linear.weight, mode='fan_in', nonlinearity='relu')
    elif opt.embed_init == 'normal':
        network.model.last_linear.weight.data.normal_(0, 0.01)
        network.model.last_linear.bias.data.zero_()
    else:
        # do nothing, already intialized
        assert opt.embed_init == 'default'

    print(f"{opt.arch.upper()}: Embedding layer (last_linear) initialized with {opt.embed_init}")

    # finetune BatchNorm layers?
    if opt.ft_batchnorm:
        print(f"{opt.arch.upper()}: BatchNorm layers will be finetuned.")
    else:
        print(f"{opt.arch.upper()}: BatchNorm layers will NOT be finetuned.")
        for module in filter(lambda m: type(m) == nn.BatchNorm2d, network.model.modules()):
            module.eval()
            module.train = lambda _: None

    return network




"""=================================================================================================================================="""
class GoogLeNet(nn.Module):
    """
    Container for GoogLeNet s.t. it can be used for metric learning.
    The Network has been broken down to allow for higher modularity, if one wishes
    to target specific layers/blocks directly.
    """
    def __init__(self, opt):
        """
        Args:
            opt: argparse.Namespace, contains all training-specific parameters.
        Returns:
            Nothing!
        """
        super(GoogLeNet, self).__init__()

        self.pars       = opt

        self.model = googlenet.googlenet(num_classes=1000, pretrained='imagenet' if not opt.not_pretrained else False)

        rename_attr(self.model, 'fc', 'last_linear')

        self.layer_blocks = nn.ModuleList([self.model.inception3a, self.model.inception3b, self.model.maxpool3,
                                           self.model.inception4a, self.model.inception4b, self.model.inception4c,
                                           self.model.inception4d, self.model.inception4e, self.model.maxpool4,
                                           self.model.inception5a, self.model.inception5b, self.model.avgpool])

        self.model.last_linear = torch.nn.Linear(self.model.last_linear.in_features, opt.embed_dim)


    def forward(self, x):
        ### Initial Conv Layers
        x = self.model.conv3(self.model.conv2(self.model.maxpool1(self.model.conv1(x))))
        x = self.model.maxpool2(x)

        ### Inception Blocks
        for layerblock in self.layer_blocks:
            x = layerblock(x)

        x = x.view(x.size(0), -1)
        x = self.model.dropout(x)

        mod_x = self.model.last_linear(x)

        #No Normalization is used if N-Pair Loss is the target criterion.
        return mod_x if self.pars.loss=='npair' else torch.nn.functional.normalize(mod_x, dim=-1)


    def to_optim(self, opt):
        # TODO
        return [{'params':model.parameters(),'lr':opt.lr,'weight_decay':opt.decay}]


"""============================================================="""
class ResNet50(nn.Module):
    """
    Container for ResNet50 s.t. it can be used for metric learning.
    The Network has been broken down to allow for higher modularity, if one wishes
    to target specific layers/blocks directly.
    """
    def __init__(self, opt, list_style=False, no_norm=False):
        super(ResNet50, self).__init__()

        self.pars = opt

        if not opt.not_pretrained:
            print('ResNet50: Getting pretrained weights...')
            self.model = ptm.__dict__['resnet50'](num_classes=1000, pretrained='imagenet')
            print('ResNet50: Done.')
        else:
            print('ResNet50: Not utilizing pretrained weights!')
            self.model = ptm.__dict__['resnet50'](num_classes=1000, pretrained=None)

        self.layer_blocks = nn.ModuleList([self.model.layer1, self.model.layer2, self.model.layer3, self.model.layer4])

        self.model.last_linear = torch.nn.Linear(self.model.last_linear.in_features, opt.embed_dim)


    def forward(self, x, is_init_cluster_generation=False):
        x = self.model.maxpool(self.model.relu(self.model.bn1(self.model.conv1(x))))

        for layerblock in self.layer_blocks:
            x = layerblock(x)

        x = self.model.avgpool(x)
        x = x.view(x.size(0),-1)

        mod_x = self.model.last_linear(x)
        #No Normalization is used if N-Pair Loss is the target criterion.
        # return mod_x if self.pars.loss=='npair' else torch.nn.functional.normalize(mod_x, dim=-1)
        return mod_x if self.pars.loss == 'Lambdarank' else torch.nn.functional.normalize(mod_x, dim=-1)


    def to_optim(self, opt):
        if 'fc_lr_mul' in vars(opt).keys() and opt.fc_lr_mul!=0:
            return [{'params':self.model.conv1.parameters(),'lr':opt.lr,'weight_decay':opt.decay},
                    {'params':self.model.bn1.parameters(),'lr':opt.lr,'weight_decay':opt.decay},
                    {'params':self.model.layer1.parameters(),'lr':opt.lr,'weight_decay':opt.decay},
                    {'params':self.model.layer2.parameters(),'lr':opt.lr,'weight_decay':opt.decay},
                    {'params':self.model.layer3.parameters(),'lr':opt.lr,'weight_decay':opt.decay},
                    {'params':self.model.layer4.parameters(),'lr':opt.lr,'weight_decay':opt.decay},
                    {'params':self.model.last_linear.parameters(),'lr':opt.lr*opt.fc_lr_mul,'weight_decay':opt.decay}]
        else:
            return [{'params':self.parameters(),'lr':opt.lr,'weight_decay':opt.decay}]


"""============================================================="""
class ResNet50_mcn(nn.Module):
    """
    class definition for the ResNet50 model imported from MatConvNet
    """
    def __init__(self, opt):
        super(ResNet50_mcn, self).__init__()

        self.pars = opt
        self.meta = {'mean': [0.485, 0.456, 0.406],
                     'std': [0.229, 0.224, 0.225],
                     'imageSize': [224, 224]}

        self.features_0 = nn.Conv2d(3, 64, kernel_size=[7, 7], stride=(2, 2), padding=(3, 3), bias=False)
        self.features_1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.features_2 = nn.ReLU(inplace=True)
        self.features_3 = nn.MaxPool2d(kernel_size=[3, 3], stride=[2, 2], padding=1, dilation=1, ceil_mode=False)
        self.features_4_0_conv1 = nn.Conv2d(64, 64, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.features_4_0_bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.features_4_0_relu1 = nn.ReLU(inplace=True)
        self.features_4_0_conv2 = nn.Conv2d(64, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.features_4_0_bn2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.features_4_0_relu2 = nn.ReLU(inplace=True)
        self.features_4_0_conv3 = nn.Conv2d(64, 256, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.features_4_0_bn3 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.features_4_0_downsample_0 = nn.Conv2d(64, 256, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.features_4_0_downsample_1 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.features_4_0_id_relu = nn.ReLU(inplace=True)
        self.features_4_1_conv1 = nn.Conv2d(256, 64, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.features_4_1_bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.features_4_1_relu1 = nn.ReLU(inplace=True)
        self.features_4_1_conv2 = nn.Conv2d(64, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.features_4_1_bn2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.features_4_1_relu2 = nn.ReLU(inplace=True)
        self.features_4_1_conv3 = nn.Conv2d(64, 256, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.features_4_1_bn3 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.features_4_1_id_relu = nn.ReLU(inplace=True)
        self.features_4_2_conv1 = nn.Conv2d(256, 64, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.features_4_2_bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.features_4_2_relu1 = nn.ReLU(inplace=True)
        self.features_4_2_conv2 = nn.Conv2d(64, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.features_4_2_bn2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.features_4_2_relu2 = nn.ReLU(inplace=True)
        self.features_4_2_conv3 = nn.Conv2d(64, 256, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.features_4_2_bn3 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.features_4_2_id_relu = nn.ReLU(inplace=True)
        self.features_5_0_conv1 = nn.Conv2d(256, 128, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.features_5_0_bn1 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.features_5_0_relu1 = nn.ReLU(inplace=True)
        self.features_5_0_conv2 = nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(2, 2), padding=(1, 1), bias=False)
        self.features_5_0_bn2 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.features_5_0_relu2 = nn.ReLU(inplace=True)
        self.features_5_0_conv3 = nn.Conv2d(128, 512, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.features_5_0_bn3 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.features_5_0_downsample_0 = nn.Conv2d(256, 512, kernel_size=[1, 1], stride=(2, 2), bias=False)
        self.features_5_0_downsample_1 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.features_5_0_id_relu = nn.ReLU(inplace=True)
        self.features_5_1_conv1 = nn.Conv2d(512, 128, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.features_5_1_bn1 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.features_5_1_relu1 = nn.ReLU(inplace=True)
        self.features_5_1_conv2 = nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.features_5_1_bn2 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.features_5_1_relu2 = nn.ReLU(inplace=True)
        self.features_5_1_conv3 = nn.Conv2d(128, 512, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.features_5_1_bn3 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.features_5_1_id_relu = nn.ReLU(inplace=True)
        self.features_5_2_conv1 = nn.Conv2d(512, 128, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.features_5_2_bn1 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.features_5_2_relu1 = nn.ReLU(inplace=True)
        self.features_5_2_conv2 = nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.features_5_2_bn2 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.features_5_2_relu2 = nn.ReLU(inplace=True)
        self.features_5_2_conv3 = nn.Conv2d(128, 512, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.features_5_2_bn3 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.features_5_2_id_relu = nn.ReLU(inplace=True)
        self.features_5_3_conv1 = nn.Conv2d(512, 128, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.features_5_3_bn1 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.features_5_3_relu1 = nn.ReLU(inplace=True)
        self.features_5_3_conv2 = nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.features_5_3_bn2 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.features_5_3_relu2 = nn.ReLU(inplace=True)
        self.features_5_3_conv3 = nn.Conv2d(128, 512, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.features_5_3_bn3 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.features_5_3_id_relu = nn.ReLU(inplace=True)
        self.features_6_0_conv1 = nn.Conv2d(512, 256, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.features_6_0_bn1 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.features_6_0_relu1 = nn.ReLU(inplace=True)
        self.features_6_0_conv2 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(2, 2), padding=(1, 1), bias=False)
        self.features_6_0_bn2 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.features_6_0_relu2 = nn.ReLU(inplace=True)
        self.features_6_0_conv3 = nn.Conv2d(256, 1024, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.features_6_0_bn3 = nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.features_6_0_downsample_0 = nn.Conv2d(512, 1024, kernel_size=[1, 1], stride=(2, 2), bias=False)
        self.features_6_0_downsample_1 = nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.features_6_0_id_relu = nn.ReLU(inplace=True)
        self.features_6_1_conv1 = nn.Conv2d(1024, 256, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.features_6_1_bn1 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.features_6_1_relu1 = nn.ReLU(inplace=True)
        self.features_6_1_conv2 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.features_6_1_bn2 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.features_6_1_relu2 = nn.ReLU(inplace=True)
        self.features_6_1_conv3 = nn.Conv2d(256, 1024, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.features_6_1_bn3 = nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.features_6_1_id_relu = nn.ReLU(inplace=True)
        self.features_6_2_conv1 = nn.Conv2d(1024, 256, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.features_6_2_bn1 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.features_6_2_relu1 = nn.ReLU(inplace=True)
        self.features_6_2_conv2 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.features_6_2_bn2 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.features_6_2_relu2 = nn.ReLU(inplace=True)
        self.features_6_2_conv3 = nn.Conv2d(256, 1024, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.features_6_2_bn3 = nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.features_6_2_id_relu = nn.ReLU(inplace=True)
        self.features_6_3_conv1 = nn.Conv2d(1024, 256, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.features_6_3_bn1 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.features_6_3_relu1 = nn.ReLU(inplace=True)
        self.features_6_3_conv2 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.features_6_3_bn2 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.features_6_3_relu2 = nn.ReLU(inplace=True)
        self.features_6_3_conv3 = nn.Conv2d(256, 1024, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.features_6_3_bn3 = nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.features_6_3_id_relu = nn.ReLU(inplace=True)
        self.features_6_4_conv1 = nn.Conv2d(1024, 256, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.features_6_4_bn1 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.features_6_4_relu1 = nn.ReLU(inplace=True)
        self.features_6_4_conv2 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.features_6_4_bn2 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.features_6_4_relu2 = nn.ReLU(inplace=True)
        self.features_6_4_conv3 = nn.Conv2d(256, 1024, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.features_6_4_bn3 = nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.features_6_4_id_relu = nn.ReLU(inplace=True)
        self.features_6_5_conv1 = nn.Conv2d(1024, 256, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.features_6_5_bn1 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.features_6_5_relu1 = nn.ReLU(inplace=True)
        self.features_6_5_conv2 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.features_6_5_bn2 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.features_6_5_relu2 = nn.ReLU(inplace=True)
        self.features_6_5_conv3 = nn.Conv2d(256, 1024, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.features_6_5_bn3 = nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.features_6_5_id_relu = nn.ReLU(inplace=True)
        self.features_7_0_conv1 = nn.Conv2d(1024, 512, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.features_7_0_bn1 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.features_7_0_relu1 = nn.ReLU(inplace=True)
        self.features_7_0_conv2 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(2, 2), padding=(1, 1), bias=False)
        self.features_7_0_bn2 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.features_7_0_relu2 = nn.ReLU(inplace=True)
        self.features_7_0_conv3 = nn.Conv2d(512, 2048, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.features_7_0_bn3 = nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.features_7_0_downsample_0 = nn.Conv2d(1024, 2048, kernel_size=[1, 1], stride=(2, 2), bias=False)
        self.features_7_0_downsample_1 = nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.features_7_0_id_relu = nn.ReLU(inplace=True)
        self.features_7_1_conv1 = nn.Conv2d(2048, 512, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.features_7_1_bn1 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.features_7_1_relu1 = nn.ReLU(inplace=True)
        self.features_7_1_conv2 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.features_7_1_bn2 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.features_7_1_relu2 = nn.ReLU(inplace=True)
        self.features_7_1_conv3 = nn.Conv2d(512, 2048, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.features_7_1_bn3 = nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.features_7_1_id_relu = nn.ReLU(inplace=True)
        self.features_7_2_conv1 = nn.Conv2d(2048, 512, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.features_7_2_bn1 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.features_7_2_relu1 = nn.ReLU(inplace=True)
        self.features_7_2_conv2 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.features_7_2_bn2 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.features_7_2_relu2 = nn.ReLU(inplace=True)
        self.features_7_2_conv3 = nn.Conv2d(512, 2048, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.features_7_2_bn3 = nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.features_7_2_id_relu = nn.ReLU(inplace=True)
        self.features_8 = nn.AvgPool2d(kernel_size=[7, 7], stride=[1, 1], padding=0)
        self.fc = nn.Linear(in_features=2048, out_features=512, bias=True)

    def forward(self, data):
        features_0 = self.features_0(data)
        features_1 = self.features_1(features_0)
        features_2 = self.features_2(features_1)
        features_3 = self.features_3(features_2)
        features_4_0_conv1 = self.features_4_0_conv1(features_3)
        features_4_0_bn1 = self.features_4_0_bn1(features_4_0_conv1)
        features_4_0_relu1 = self.features_4_0_relu1(features_4_0_bn1)
        features_4_0_conv2 = self.features_4_0_conv2(features_4_0_relu1)
        features_4_0_bn2 = self.features_4_0_bn2(features_4_0_conv2)
        features_4_0_relu2 = self.features_4_0_relu2(features_4_0_bn2)
        features_4_0_conv3 = self.features_4_0_conv3(features_4_0_relu2)
        features_4_0_bn3 = self.features_4_0_bn3(features_4_0_conv3)
        features_4_0_downsample_0 = self.features_4_0_downsample_0(features_3)
        features_4_0_downsample_1 = self.features_4_0_downsample_1(features_4_0_downsample_0)
        features_4_0_merge = torch.add(features_4_0_downsample_1, 1, features_4_0_bn3)
        features_4_0_id_relu = self.features_4_0_id_relu(features_4_0_merge)
        features_4_1_conv1 = self.features_4_1_conv1(features_4_0_id_relu)
        features_4_1_bn1 = self.features_4_1_bn1(features_4_1_conv1)
        features_4_1_relu1 = self.features_4_1_relu1(features_4_1_bn1)
        features_4_1_conv2 = self.features_4_1_conv2(features_4_1_relu1)
        features_4_1_bn2 = self.features_4_1_bn2(features_4_1_conv2)
        features_4_1_relu2 = self.features_4_1_relu2(features_4_1_bn2)
        features_4_1_conv3 = self.features_4_1_conv3(features_4_1_relu2)
        features_4_1_bn3 = self.features_4_1_bn3(features_4_1_conv3)
        features_4_1_merge = torch.add(features_4_0_id_relu, 1, features_4_1_bn3)
        features_4_1_id_relu = self.features_4_1_id_relu(features_4_1_merge)
        features_4_2_conv1 = self.features_4_2_conv1(features_4_1_id_relu)
        features_4_2_bn1 = self.features_4_2_bn1(features_4_2_conv1)
        features_4_2_relu1 = self.features_4_2_relu1(features_4_2_bn1)
        features_4_2_conv2 = self.features_4_2_conv2(features_4_2_relu1)
        features_4_2_bn2 = self.features_4_2_bn2(features_4_2_conv2)
        features_4_2_relu2 = self.features_4_2_relu2(features_4_2_bn2)
        features_4_2_conv3 = self.features_4_2_conv3(features_4_2_relu2)
        features_4_2_bn3 = self.features_4_2_bn3(features_4_2_conv3)
        features_4_2_merge = torch.add(features_4_1_id_relu, 1, features_4_2_bn3)
        features_4_2_id_relu = self.features_4_2_id_relu(features_4_2_merge)
        features_5_0_conv1 = self.features_5_0_conv1(features_4_2_id_relu)
        features_5_0_bn1 = self.features_5_0_bn1(features_5_0_conv1)
        features_5_0_relu1 = self.features_5_0_relu1(features_5_0_bn1)
        features_5_0_conv2 = self.features_5_0_conv2(features_5_0_relu1)
        features_5_0_bn2 = self.features_5_0_bn2(features_5_0_conv2)
        features_5_0_relu2 = self.features_5_0_relu2(features_5_0_bn2)
        features_5_0_conv3 = self.features_5_0_conv3(features_5_0_relu2)
        features_5_0_bn3 = self.features_5_0_bn3(features_5_0_conv3)
        features_5_0_downsample_0 = self.features_5_0_downsample_0(features_4_2_id_relu)
        features_5_0_downsample_1 = self.features_5_0_downsample_1(features_5_0_downsample_0)
        features_5_0_merge = torch.add(features_5_0_downsample_1, 1, features_5_0_bn3)
        features_5_0_id_relu = self.features_5_0_id_relu(features_5_0_merge)
        features_5_1_conv1 = self.features_5_1_conv1(features_5_0_id_relu)
        features_5_1_bn1 = self.features_5_1_bn1(features_5_1_conv1)
        features_5_1_relu1 = self.features_5_1_relu1(features_5_1_bn1)
        features_5_1_conv2 = self.features_5_1_conv2(features_5_1_relu1)
        features_5_1_bn2 = self.features_5_1_bn2(features_5_1_conv2)
        features_5_1_relu2 = self.features_5_1_relu2(features_5_1_bn2)
        features_5_1_conv3 = self.features_5_1_conv3(features_5_1_relu2)
        features_5_1_bn3 = self.features_5_1_bn3(features_5_1_conv3)
        features_5_1_merge = torch.add(features_5_0_id_relu, 1, features_5_1_bn3)
        features_5_1_id_relu = self.features_5_1_id_relu(features_5_1_merge)
        features_5_2_conv1 = self.features_5_2_conv1(features_5_1_id_relu)
        features_5_2_bn1 = self.features_5_2_bn1(features_5_2_conv1)
        features_5_2_relu1 = self.features_5_2_relu1(features_5_2_bn1)
        features_5_2_conv2 = self.features_5_2_conv2(features_5_2_relu1)
        features_5_2_bn2 = self.features_5_2_bn2(features_5_2_conv2)
        features_5_2_relu2 = self.features_5_2_relu2(features_5_2_bn2)
        features_5_2_conv3 = self.features_5_2_conv3(features_5_2_relu2)
        features_5_2_bn3 = self.features_5_2_bn3(features_5_2_conv3)
        features_5_2_merge = torch.add(features_5_1_id_relu, 1, features_5_2_bn3)
        features_5_2_id_relu = self.features_5_2_id_relu(features_5_2_merge)
        features_5_3_conv1 = self.features_5_3_conv1(features_5_2_id_relu)
        features_5_3_bn1 = self.features_5_3_bn1(features_5_3_conv1)
        features_5_3_relu1 = self.features_5_3_relu1(features_5_3_bn1)
        features_5_3_conv2 = self.features_5_3_conv2(features_5_3_relu1)
        features_5_3_bn2 = self.features_5_3_bn2(features_5_3_conv2)
        features_5_3_relu2 = self.features_5_3_relu2(features_5_3_bn2)
        features_5_3_conv3 = self.features_5_3_conv3(features_5_3_relu2)
        features_5_3_bn3 = self.features_5_3_bn3(features_5_3_conv3)
        features_5_3_merge = torch.add(features_5_2_id_relu, 1, features_5_3_bn3)
        features_5_3_id_relu = self.features_5_3_id_relu(features_5_3_merge)
        features_6_0_conv1 = self.features_6_0_conv1(features_5_3_id_relu)
        features_6_0_bn1 = self.features_6_0_bn1(features_6_0_conv1)
        features_6_0_relu1 = self.features_6_0_relu1(features_6_0_bn1)
        features_6_0_conv2 = self.features_6_0_conv2(features_6_0_relu1)
        features_6_0_bn2 = self.features_6_0_bn2(features_6_0_conv2)
        features_6_0_relu2 = self.features_6_0_relu2(features_6_0_bn2)
        features_6_0_conv3 = self.features_6_0_conv3(features_6_0_relu2)
        features_6_0_bn3 = self.features_6_0_bn3(features_6_0_conv3)
        features_6_0_downsample_0 = self.features_6_0_downsample_0(features_5_3_id_relu)
        features_6_0_downsample_1 = self.features_6_0_downsample_1(features_6_0_downsample_0)
        features_6_0_merge = torch.add(features_6_0_downsample_1, 1, features_6_0_bn3)
        features_6_0_id_relu = self.features_6_0_id_relu(features_6_0_merge)
        features_6_1_conv1 = self.features_6_1_conv1(features_6_0_id_relu)
        features_6_1_bn1 = self.features_6_1_bn1(features_6_1_conv1)
        features_6_1_relu1 = self.features_6_1_relu1(features_6_1_bn1)
        features_6_1_conv2 = self.features_6_1_conv2(features_6_1_relu1)
        features_6_1_bn2 = self.features_6_1_bn2(features_6_1_conv2)
        features_6_1_relu2 = self.features_6_1_relu2(features_6_1_bn2)
        features_6_1_conv3 = self.features_6_1_conv3(features_6_1_relu2)
        features_6_1_bn3 = self.features_6_1_bn3(features_6_1_conv3)
        features_6_1_merge = torch.add(features_6_0_id_relu, 1, features_6_1_bn3)
        features_6_1_id_relu = self.features_6_1_id_relu(features_6_1_merge)
        features_6_2_conv1 = self.features_6_2_conv1(features_6_1_id_relu)
        features_6_2_bn1 = self.features_6_2_bn1(features_6_2_conv1)
        features_6_2_relu1 = self.features_6_2_relu1(features_6_2_bn1)
        features_6_2_conv2 = self.features_6_2_conv2(features_6_2_relu1)
        features_6_2_bn2 = self.features_6_2_bn2(features_6_2_conv2)
        features_6_2_relu2 = self.features_6_2_relu2(features_6_2_bn2)
        features_6_2_conv3 = self.features_6_2_conv3(features_6_2_relu2)
        features_6_2_bn3 = self.features_6_2_bn3(features_6_2_conv3)
        features_6_2_merge = torch.add(features_6_1_id_relu, 1, features_6_2_bn3)
        features_6_2_id_relu = self.features_6_2_id_relu(features_6_2_merge)
        features_6_3_conv1 = self.features_6_3_conv1(features_6_2_id_relu)
        features_6_3_bn1 = self.features_6_3_bn1(features_6_3_conv1)
        features_6_3_relu1 = self.features_6_3_relu1(features_6_3_bn1)
        features_6_3_conv2 = self.features_6_3_conv2(features_6_3_relu1)
        features_6_3_bn2 = self.features_6_3_bn2(features_6_3_conv2)
        features_6_3_relu2 = self.features_6_3_relu2(features_6_3_bn2)
        features_6_3_conv3 = self.features_6_3_conv3(features_6_3_relu2)
        features_6_3_bn3 = self.features_6_3_bn3(features_6_3_conv3)
        features_6_3_merge = torch.add(features_6_2_id_relu, 1, features_6_3_bn3)
        features_6_3_id_relu = self.features_6_3_id_relu(features_6_3_merge)
        features_6_4_conv1 = self.features_6_4_conv1(features_6_3_id_relu)
        features_6_4_bn1 = self.features_6_4_bn1(features_6_4_conv1)
        features_6_4_relu1 = self.features_6_4_relu1(features_6_4_bn1)
        features_6_4_conv2 = self.features_6_4_conv2(features_6_4_relu1)
        features_6_4_bn2 = self.features_6_4_bn2(features_6_4_conv2)
        features_6_4_relu2 = self.features_6_4_relu2(features_6_4_bn2)
        features_6_4_conv3 = self.features_6_4_conv3(features_6_4_relu2)
        features_6_4_bn3 = self.features_6_4_bn3(features_6_4_conv3)
        features_6_4_merge = torch.add(features_6_3_id_relu, 1, features_6_4_bn3)
        features_6_4_id_relu = self.features_6_4_id_relu(features_6_4_merge)
        features_6_5_conv1 = self.features_6_5_conv1(features_6_4_id_relu)
        features_6_5_bn1 = self.features_6_5_bn1(features_6_5_conv1)
        features_6_5_relu1 = self.features_6_5_relu1(features_6_5_bn1)
        features_6_5_conv2 = self.features_6_5_conv2(features_6_5_relu1)
        features_6_5_bn2 = self.features_6_5_bn2(features_6_5_conv2)
        features_6_5_relu2 = self.features_6_5_relu2(features_6_5_bn2)
        features_6_5_conv3 = self.features_6_5_conv3(features_6_5_relu2)
        features_6_5_bn3 = self.features_6_5_bn3(features_6_5_conv3)
        features_6_5_merge = torch.add(features_6_4_id_relu, 1, features_6_5_bn3)
        features_6_5_id_relu = self.features_6_5_id_relu(features_6_5_merge)
        features_7_0_conv1 = self.features_7_0_conv1(features_6_5_id_relu)
        features_7_0_bn1 = self.features_7_0_bn1(features_7_0_conv1)
        features_7_0_relu1 = self.features_7_0_relu1(features_7_0_bn1)
        features_7_0_conv2 = self.features_7_0_conv2(features_7_0_relu1)
        features_7_0_bn2 = self.features_7_0_bn2(features_7_0_conv2)
        features_7_0_relu2 = self.features_7_0_relu2(features_7_0_bn2)
        features_7_0_conv3 = self.features_7_0_conv3(features_7_0_relu2)
        features_7_0_bn3 = self.features_7_0_bn3(features_7_0_conv3)
        features_7_0_downsample_0 = self.features_7_0_downsample_0(features_6_5_id_relu)
        features_7_0_downsample_1 = self.features_7_0_downsample_1(features_7_0_downsample_0)
        features_7_0_merge = torch.add(features_7_0_downsample_1, 1, features_7_0_bn3)
        features_7_0_id_relu = self.features_7_0_id_relu(features_7_0_merge)
        features_7_1_conv1 = self.features_7_1_conv1(features_7_0_id_relu)
        features_7_1_bn1 = self.features_7_1_bn1(features_7_1_conv1)
        features_7_1_relu1 = self.features_7_1_relu1(features_7_1_bn1)
        features_7_1_conv2 = self.features_7_1_conv2(features_7_1_relu1)
        features_7_1_bn2 = self.features_7_1_bn2(features_7_1_conv2)
        features_7_1_relu2 = self.features_7_1_relu2(features_7_1_bn2)
        features_7_1_conv3 = self.features_7_1_conv3(features_7_1_relu2)
        features_7_1_bn3 = self.features_7_1_bn3(features_7_1_conv3)
        features_7_1_merge = torch.add(features_7_0_id_relu, 1, features_7_1_bn3)
        features_7_1_id_relu = self.features_7_1_id_relu(features_7_1_merge)
        features_7_2_conv1 = self.features_7_2_conv1(features_7_1_id_relu)
        features_7_2_bn1 = self.features_7_2_bn1(features_7_2_conv1)
        features_7_2_relu1 = self.features_7_2_relu1(features_7_2_bn1)
        features_7_2_conv2 = self.features_7_2_conv2(features_7_2_relu1)
        features_7_2_bn2 = self.features_7_2_bn2(features_7_2_conv2)
        features_7_2_relu2 = self.features_7_2_relu2(features_7_2_bn2)
        features_7_2_conv3 = self.features_7_2_conv3(features_7_2_relu2)
        features_7_2_bn3 = self.features_7_2_bn3(features_7_2_conv3)
        features_7_2_merge = torch.add(features_7_1_id_relu, 1, features_7_2_bn3)
        features_7_2_id_relu = self.features_7_2_id_relu(features_7_2_merge)
        features_8 = self.features_8(features_7_2_id_relu)
        classifier_flatten = features_8.view(features_8.size(0), -1)
        logits = self.fc(classifier_flatten)

        #No Normalization is used if N-Pair Loss is the target criterion.
        return logits if self.pars.loss=='npair' else torch.nn.functional.normalize(logits, dim=-1)

    def load_pth(self, weights_path):
        if weights_path:
            state_dict = torch.load(weights_path)
            self.load_state_dict(state_dict)

    def to_optim(self, opt):
        return [{'params':self.parameters(),'lr':opt.lr,'weight_decay':opt.decay}]
