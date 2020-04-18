# coding: utf-8
"""
Shared encoders (U-net).
"""
import torch
import torch.nn as nn
from collections import OrderedDict

import torchvision.models as models

from model.blocks import RegularBottleneck, DownsamplingBottleneck, InitialBlock, InputProjectionA, \
    DilatedParallelResidualBlockB, DownSamplerB, C, CBR, BR


class VGGEncoder(nn.Module):
    """
    Simple VGG Encoder
    """

    def __init__(self, num_blocks, in_channels, out_channels):
        super(VGGEncoder, self).__init__()

        self.pretrained_modules = models.vgg16(pretrained=True).features

        self.num_blocks = num_blocks
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._conv_reps = [2, 2, 3, 3, 3]
        self.net = nn.Sequential()
        self.pretrained_net = nn.Sequential()

        for i in range(num_blocks):
            self.net.add_module("block" + str(i + 1), self._encode_block(i + 1))
            self.pretrained_net.add_module("block" + str(i + 1), self._encode_pretrained_block(i + 1))

    def _encode_block(self, block_id, kernel_size=3, stride=1):
        out_channels = self._out_channels[block_id - 1]
        padding = (kernel_size - 1) // 2
        seq = nn.Sequential()

        for i in range(self._conv_reps[block_id - 1]):
            if i == 0:
                in_channels = self._in_channels[block_id - 1]
            else:
                in_channels = out_channels
            seq.add_module("conv_{}_{}".format(block_id, i + 1),
                           nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding))
            seq.add_module("bn_{}_{}".format(block_id, i + 1), nn.BatchNorm2d(out_channels))
            seq.add_module("relu_{}_{}".format(block_id, i + 1), nn.ReLU())
        seq.add_module("maxpool" + str(block_id), nn.MaxPool2d(kernel_size=2, stride=2))
        return seq

    def _encode_pretrained_block(self, block_id):
        seq = nn.Sequential()
        for i in range(0, self._conv_reps[block_id - 1], 4):
            seq.add_module("conv_{}_{}".format(block_id, i + 1), self.pretrained_modules[i])
            seq.add_module("relu_{}_{}".format(block_id, i + 2), self.pretrained_modules[i + 1])
            seq.add_module("conv_{}_{}".format(block_id, i + 3), self.pretrained_modules[i + 2])
            seq.add_module("relu_{}_{}".format(block_id, i + 4), self.pretrained_modules[i + 3])
            seq.add_module("maxpool" + str(block_id), self.pretrained_modules[i + 4])
        return seq

    def forward(self, input_tensor):
        ret = OrderedDict()
        # 5 stage of encoding
        X = input_tensor
        for i, block in enumerate(self.net):
            pool = block(X)
            ret["pool" + str(i + 1)] = pool

            X = pool
        return ret


class ESPNetEncoder(nn.Module):
    """
    ESPNet-C encoder
    """

    def __init__(self, classes=20, p=5, q=3):
        '''
        :param classes: number of classes in the dataset. Default is 20 for the cityscapes
        :param p: depth multiplier
        :param q: depth multiplier
        '''
        super().__init__()
        self.level1 = CBR(3, 16, 3, 2)
        self.sample1 = InputProjectionA(1)
        self.sample2 = InputProjectionA(2)

        self.b1 = BR(16 + 3)
        self.level2_0 = DownSamplerB(16 + 3, 64)

        self.level2 = nn.ModuleList()
        for i in range(0, p):
            self.level2.append(DilatedParallelResidualBlockB(64, 64))
        self.b2 = BR(128 + 3)

        self.level3_0 = DownSamplerB(128 + 3, 128)
        self.level3 = nn.ModuleList()
        for i in range(0, q):
            self.level3.append(DilatedParallelResidualBlockB(128, 128))
        self.b3 = BR(256)

        self.classifier = C(256, classes, 1, 1)

    def forward(self, input):
        '''
        :param input: Receives the input RGB image
        :return: the transformed feature map with spatial dimensions 1/8th of the input image
        '''
        output0 = self.level1(input)
        inp1 = self.sample1(input)
        inp2 = self.sample2(input)

        output0_cat = self.b1(torch.cat([output0, inp1], 1))
        output1_0 = self.level2_0(output0_cat)  # down-sampled

        for i, layer in enumerate(self.level2):
            if i == 0:
                output1 = layer(output1_0)
            else:
                output1 = layer(output1)

        output1_cat = self.b2(torch.cat([output1, output1_0, inp2], 1))

        output2_0 = self.level3_0(output1_cat)  # down-sampled
        for i, layer in enumerate(self.level3):
            if i == 0:
                output2 = layer(output2_0)
            else:
                output2 = layer(output2)

        output2_cat = self.b3(torch.cat([output2_0, output2], 1))

        classifier = self.classifier(output2_cat)

        return classifier


class ENetEncoder(nn.Module):
    """
    ENET Encoder
    """

    def __init__(self, num_classes, encoder_relu=False, decoder_relu=True):
        super().__init__()

    def forward(self, input):
        self.initial_block = InitialBlock(3, 16, padding=1, relu=encoder_relu)

        # Stage 1 - Encoder
        self.downsample1_0 = DownsamplingBottleneck(16, 64, padding=1, return_indices=True, dropout_prob=0.01,
                                                    relu=encoder_relu)
        self.regular1_1 = RegularBottleneck(64, padding=1, dropout_prob=0.01, relu=encoder_relu)
        self.regular1_2 = RegularBottleneck(64, padding=1, dropout_prob=0.01, relu=encoder_relu)
        self.regular1_3 = RegularBottleneck(64, padding=1, dropout_prob=0.01, relu=encoder_relu)
        self.regular1_4 = RegularBottleneck(64, padding=1, dropout_prob=0.01, relu=encoder_relu)

        # Stage 2 - Encoder
        self.downsample2_0 = DownsamplingBottleneck(64, 128, padding=1, return_indices=True, dropout_prob=0.1,
                                                    relu=encoder_relu)
        self.regular2_1 = RegularBottleneck(128, padding=1, dropout_prob=0.1, relu=encoder_relu)
        self.dilated2_2 = RegularBottleneck(128, dilation=2, padding=2, dropout_prob=0.1, relu=encoder_relu)
        self.asymmetric2_3 = RegularBottleneck(128, kernel_size=5, padding=2, asymmetric=True, dropout_prob=0.1,
                                               relu=encoder_relu)
        self.dilated2_4 = RegularBottleneck(128, dilation=4, padding=4, dropout_prob=0.1, relu=encoder_relu)
        self.regular2_5 = RegularBottleneck(128, padding=1, dropout_prob=0.1, relu=encoder_relu)
        self.dilated2_6 = RegularBottleneck(128, dilation=8, padding=8, dropout_prob=0.1, relu=encoder_relu)
        self.asymmetric2_7 = RegularBottleneck(128, kernel_size=5, asymmetric=True, padding=2, dropout_prob=0.1,
                                               relu=encoder_relu)
        self.dilated2_8 = RegularBottleneck(128, dilation=16, padding=16, dropout_prob=0.1, relu=encoder_relu)

        # Stage 3 - Encoder
        self.regular3_0 = RegularBottleneck(128, padding=1, dropout_prob=0.1, relu=encoder_relu)
        self.dilated3_1 = RegularBottleneck(128, dilation=2, padding=2, dropout_prob=0.1, relu=encoder_relu)
        self.asymmetric3_2 = RegularBottleneck(128, kernel_size=5, padding=2, asymmetric=True, dropout_prob=0.1,
                                               relu=encoder_relu)
        self.dilated3_3 = RegularBottleneck(128, dilation=4, padding=4, dropout_prob=0.1, relu=encoder_relu)
        self.regular3_4 = RegularBottleneck(128, padding=1, dropout_prob=0.1, relu=encoder_relu)
        self.dilated3_5 = RegularBottleneck(128, dilation=8, padding=8, dropout_prob=0.1, relu=encoder_relu)
        self.asymmetric3_6 = RegularBottleneck(128, kernel_size=5, asymmetric=True, padding=2, dropout_prob=0.1,
                                               relu=encoder_relu)
        self.dilated3_7 = RegularBottleneck(128, dilation=16, padding=16, dropout_prob=0.1, relu=encoder_relu)
