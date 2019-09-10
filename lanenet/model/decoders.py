import torch
import torch.nn as nn

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class ESPNetDecoder():
    def __init__(self):

        # light-weight decoder
        self.level3_C = C(128 + 3, classes, 1, 1)
        self.br = nn.BatchNorm2d(classes, eps=1e-03)
        self.conv = CBR(19 + classes, classes, 3, 1)

        self.up_l3 = nn.Sequential(
            nn.ConvTranspose2d(classes, classes, 2, stride=2, padding=0, output_padding=0, bias=False))
        self.combine_l2_l3 = nn.Sequential(BR(2 * classes),
                                           DilatedParllelResidualBlockB(2 * classes, classes, add=False))

        self.up_l2 = nn.Sequential(
            nn.ConvTranspose2d(classes, classes, 2, stride=2, padding=0, output_padding=0, bias=False), BR(classes))

        self.classifier = nn.ConvTranspose2d(classes, classes, 2, stride=2, padding=0, output_padding=0, bias=False)

    def forward(self, input):
        '''
        :param input: RGB image
        :return: transformed feature map
        '''
        output0 = self.modules[0](input)
        inp1 = self.modules[1](input)
        inp2 = self.modules[2](input)

        output0_cat = self.modules[3](torch.cat([output0, inp1], 1))
        output1_0 = self.modules[4](output0_cat)  # down-sampled

        for i, layer in enumerate(self.modules[5]):
            if i == 0:
                output1 = layer(output1_0)
            else:
                output1 = layer(output1)

        output1_cat = self.modules[6](torch.cat([output1, output1_0, inp2], 1))

        output2_0 = self.modules[7](output1_cat)  # down-sampled
        for i, layer in enumerate(self.modules[8]):
            if i == 0:
                output2 = layer(output2_0)
            else:
                output2 = layer(output2)

        output2_cat = self.modules[9](torch.cat([output2_0, output2], 1))  # concatenate for feature map width expansion

        output2_c = self.up_l3(self.br(self.modules[10](output2_cat)))  # RUM

        output1_C = self.level3_C(output1_cat)  # project to C-dimensional space
        comb_l2_l3 = self.up_l2(self.combine_l2_l3(torch.cat([output1_C, output2_c], 1)))  # RUM

        concat_features = self.conv(torch.cat([comb_l2_l3, output0_cat], 1))

        classifier = self.classifier(concat_features)
        return classifier


class ENetDecoder():
    def __init__(self):
        # Stage 4 - Decoder
        self.upsample4_0 = UpsamplingBottleneck(
            128, 64, padding=1, dropout_prob=0.1, relu=decoder_relu)
        self.regular4_1 = RegularBottleneck(
            64, padding=1, dropout_prob=0.1, relu=decoder_relu)
        self.regular4_2 = RegularBottleneck(
            64, padding=1, dropout_prob=0.1, relu=decoder_relu)

        # Stage 5 - Decoder
        self.upsample5_0 = UpsamplingBottleneck(
            64, 16, padding=1, dropout_prob=0.1, relu=decoder_relu)
        self.regular5_1 = RegularBottleneck(
            16, padding=1, dropout_prob=0.1, relu=decoder_relu)
        self.transposed_conv = nn.ConvTranspose2d(
            16,
            num_classes,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=False)

    def forward(self, x):
        # Initial block
        x = self.initial_block(x)

        # Stage 1 - Encoder
        x, max_indices1_0 = self.downsample1_0(x)
        x = self.regular1_1(x)
        x = self.regular1_2(x)
        x = self.regular1_3(x)
        x = self.regular1_4(x)

        # Stage 2 - Encoder
        x, max_indices2_0 = self.downsample2_0(x)
        x = self.regular2_1(x)
        x = self.dilated2_2(x)
        x = self.asymmetric2_3(x)
        x = self.dilated2_4(x)
        x = self.regular2_5(x)
        x = self.dilated2_6(x)
        x = self.asymmetric2_7(x)
        x = self.dilated2_8(x)

        # Stage 3 - Encoder
        x = self.regular3_0(x)
        x = self.dilated3_1(x)
        x = self.asymmetric3_2(x)
        x = self.dilated3_3(x)
        x = self.regular3_4(x)
        x = self.dilated3_5(x)
        x = self.asymmetric3_6(x)
        x = self.dilated3_7(x)

        # Stage 4 - Decoder
        x = self.upsample4_0(x, max_indices2_0)
        x = self.regular4_1(x)
        x = self.regular4_2(x)

        # Stage 5 - Decoder
        x = self.upsample5_0(x, max_indices1_0)
        x = self.regular5_1(x)
        x = self.transposed_conv(x)

        return x


class FCNDecoder(nn.Module):
    def __init__(self, decode_layers, decode_channels=[], decode_last_stride=8):
        super(FCNDecoder, self).__init__()

        self._decode_channels = [512, 256]
        self._out_channel = 64
        self._decode_layers = decode_layers

        self._conv_layers = []
        for _ch in self._decode_channels:
            self._conv_layers.append(nn.Conv2d(_ch, self._out_channel, kernel_size=1, bias=False).to(DEVICE))

        self._conv_final = nn.Conv2d(self._out_channel, 2, kernel_size=1, bias=False)
        self._deconv = nn.ConvTranspose2d(self._out_channel, self._out_channel, kernel_size=4, stride=2, padding=1,
                                          bias=False)

        self._deconv_final = nn.ConvTranspose2d(self._out_channel, self._out_channel, kernel_size=16,
                                                stride=decode_last_stride,
                                                padding=4, bias=False)

    def forward(self, encode_data):
        ret = {}
        input_tensor = encode_data[self._decode_layers[0]]
        input_tensor.to(DEVICE)
        score = self._conv_layers[0](input_tensor)
        for i, layer in enumerate(self._decode_layers[1:]):
            deconv = self._deconv(score)

            input_tensor = encode_data[layer]
            score = self._conv_layers[i](input_tensor)

            fused = torch.add(deconv, score)
            score = fused

        deconv_final = self._deconv_final(score)
        score_final = self._conv_final(deconv_final)

        ret['logits'] = score_final
        ret['deconv'] = deconv_final
        return ret
