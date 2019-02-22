# coding: utf-8
"""
LaneNet model
https://arxiv.org/pdf/1807.01726.pdf
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from torch.nn import init
from model.loss import DiscriminativeLoss
import math
from model.encoders import VGGEncoder
from model.decoders import FCNDecoder

class LaneNet(nn.Module):
    def __init__(self, arch="VGG"):

        super(LaneNet, self).__init__()
        encode_num_blocks = 5
        in_channels = [3, 64, 128, 256, 512]
        out_channels = in_channels[1:] + [512]
        self._arch = arch
        if self._arch == 'VGG':
            self._encoder = VGGEncoder(encode_num_blocks, in_channels, out_channels)
            decode_layers = ["pool5", "pool4", "pool3"]
            decode_channels = out_channels[:-len(decode_layers) - 1:-1]
            decode_last_stride = 8
            self._decoder = FCNDecoder(decode_layers, decode_channels, decode_last_stride)
        elif self._arch == 'ESPNet':
            raise NotImplementedError
        elif self._arch == 'ENNet':
            raise NotImplementedError

        self._pix_layer = nn.Sequential(nn.Conv2d(64, 3, 1, bias=False), nn.ReLU())

    def forward(self, input_tensor):
        encode_ret = self._encoder(input_tensor)
        decode_ret = self._decoder(encode_ret)

        decode_logits = decode_ret['logits']

        if torch.cuda.is_available():
            decode_logits = decode_logits.cuda()
        binary_seg_ret = torch.argmax(F.softmax(decode_logits, dim=1), dim=1, keepdim=True)

        decode_deconv = decode_ret['deconv']
        pix_embedding = self._pix_layer(decode_deconv)
        ret = {
            'instance_seg_logits': pix_embedding,
            'binary_seg_pred': binary_seg_ret,
            'binary_seg_logits': decode_logits
        }

        return ret


if __name__ == '__main__':
    model = LaneNet("sknet")
    input_tensor = torch.rand(4, 3, 256, 512, dtype=torch.float32)
    binary_label = torch.randint(1, (4, 1, 256, 512), dtype=torch.long)
    instance_label = torch.randint(4, (4, 1, 256, 512), dtype=torch.long)
    net_output = model(input_tensor, binary_label, instance_label)

    print("logits", net_output['logits'].shape)
