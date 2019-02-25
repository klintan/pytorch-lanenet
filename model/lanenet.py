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
from model.loss import DiscriminativeLoss
from model.encoders import VGGEncoder
from model.decoders import FCNDecoder


class LaneNet(nn.Module):
    def __init__(self, arch="VGG"):
        super(LaneNet, self).__init__()
        # no of instances for segmentation
        self.no_of_instances = 5
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

        self._pix_layer = nn.Conv2d(in_channels=64, out_channels=self.no_of_instances, kernel_size=1, bias=False)
        self.relu = nn.ReLU()

    def forward(self, input_tensor):
        encode_ret = self._encoder(input_tensor)
        decode_ret = self._decoder(encode_ret)

        decode_logits = decode_ret['logits']

        if torch.cuda.is_available():
            decode_logits = decode_logits.cuda()

        binary_seg_ret = torch.argmax(F.softmax(decode_logits, dim=1), dim=1, keepdim=True)

        decode_deconv = decode_ret['deconv']
        pix_embedding = self.relu(self._pix_layer(decode_deconv))

        ret = {
            'instance_seg_logits': pix_embedding.type(torch.LongTensor),
            'binary_seg_pred': binary_seg_ret,
            'binary_seg_logits': decode_logits
        }

        return ret


def compute_loss(net_output, binary_label, instance_label):
    k_binary = 0.7
    k_instance = 0.3

    ce_loss_fn = nn.CrossEntropyLoss()
    binary_seg_logits = net_output["binary_seg_logits"]
    binary_loss = ce_loss_fn(binary_seg_logits, binary_label)

    pix_embedding = net_output["instance_seg_logits"]
    ds_loss_fn = DiscriminativeLoss(0.5, 1.5, 1.0, 1.0, 0.001)

    instance_loss, _, _, _ = ds_loss_fn(pix_embedding, instance_label, [5] * len(pix_embedding))
    binary_loss = binary_loss * k_binary
    instance_loss = instance_loss * k_instance
    total_loss = binary_loss + instance_loss
    out = net_output["binary_seg_pred"]
    iou = 0
    batch_size = out.size()[0]
    for i in range(batch_size):
        PR = out[i].squeeze(0).nonzero().size()[0]
        GT = binary_label[i].nonzero().size()[0]
        TP = (out[i].squeeze(0) * binary_label[i]).nonzero().size()[0]
        union = PR + GT - TP
        iou += TP / union
    iou = iou / batch_size
    return total_loss, binary_loss, instance_loss, out, iou

