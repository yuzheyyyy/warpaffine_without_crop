# -*- coding: utf-8 -*-
import math
import torch
import numpy as np
import torch.nn as nn

def load_checkpoint(net, checkpoint, strict=True):
    from collections import OrderedDict
    temp = OrderedDict()
    if 'state_dict' in checkpoint:
        checkpoint = dict(checkpoint['state_dict'])
    for k in checkpoint:
        k2 = 'module.' + k if not k.startswith('module.') else k
        temp[k2] = checkpoint[k]
    net.load_state_dict(temp, strict=strict)

class Eltwise(nn.Module):
    def __init__(self):
        super(Eltwise, self).__init__()

    def forward(self, *inputs):
        if (len(inputs) == 2):
            return inputs[0] + inputs[1]
        elif (len(inputs[0]) == 2):
            return inputs[0][0] + inputs[0][1]
        else:
            assert False

class Concat(nn.Module):
    def __init__(self, dim=1):
        super(Concat, self).__init__()
        self.dim = dim

    def forward(self, *inputs):
        if(len(inputs) != 1):
            return torch.cat(inputs, dim=self.dim)
        elif(len(inputs[0]) != 1):
            return torch.cat(inputs[0], dim=self.dim)
        else:
            assert False

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

def conv_bn_relu(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        dilation=1,
        use_acb=False,
        norm_layer=nn.BatchNorm2d):
    padding = int(
        math.ceil(
            (kernel_size *
             dilation -
             dilation +
             1 -
             stride) /
            2.))
    if use_acb:
        raise NotImplementedError
    else:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                      padding, dilation, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True))

def conv_dw_bn_relu(
        in_channels,
        kernel_size=3,
        stride=1,
        dilation=1,
        use_acb=False,
        norm_layer=nn.BatchNorm2d):
    padding = int(
        math.ceil(
            (kernel_size *
             dilation -
             dilation +
             1 -
             stride) /
            2.))
    if use_acb:
        raise NotImplementedError

    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups=in_channels,
            bias=False),
        norm_layer(in_channels),
        nn.ReLU(
            inplace=True))

def conv_pw_bn_relu(
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        norm_layer=nn.BatchNorm2d):
    # 可能会有residual，一般不加relu
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            0,
            bias=False),
        norm_layer(out_channels),
        nn.ReLU(
            inplace=True))

def bilinear_kernel(in_channels, kernel_size):
    data = np.zeros([in_channels * 1 * kernel_size * kernel_size],
                    dtype=np.float32)
    f = np.ceil(kernel_size / 2.)
    c = (2. * f - 1. - f % 2.) / (2. * f)
    for i in range(data.shape[0]):
        x = i % kernel_size
        y = (i // kernel_size) % kernel_size
        data[i] = (1. - np.fabs(x / f - c)) * (1. - np.fabs(y / f - c))
    data = data.reshape([in_channels, 1, kernel_size, kernel_size])
    return torch.from_numpy(data)

def conv_t_up_bilinear(channels, scale_factor=2.0):
    kernel_size = int(2 * scale_factor - scale_factor % 2)
    padding = int(math.ceil((scale_factor - 1) / 2.))
    stride = int(scale_factor)
    up = nn.ConvTranspose2d(channels,
                            channels,
                            kernel_size,
                            stride=stride,
                            padding=padding,
                            groups=channels,
                            bias=False)
    up.weight.requires_grad = False
    up.weight.data = bilinear_kernel(channels, kernel_size)
    return up


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def depthwise_conv(inp, oup, kernel_size=3, stride=1, relu=False, dilation=1):
    padding = (kernel_size - 1) // 2 * dilation
    return nn.Sequential(
        nn.Conv2d(
            inp,
            oup,
            kernel_size,
            stride,
            groups=inp,
            bias=False,
            dilation=dilation,
            padding=padding),
        nn.BatchNorm2d(oup),
        nn.ReLU(
            inplace=True) if relu else nn.Sequential(),
    )

class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel), )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y = torch.clamp(y, 0, 1)
        return x * y

class GhostModule(nn.Module):
    def __init__(
            self,
            inp,
            oup,
            kernel_size=1,
            ratio=2,
            dw_size=3,
            stride=1,
            relu=True,
            dilation=1):
        super(GhostModule, self).__init__()
        self.oup = oup

        init_channels = _make_divisible(math.ceil(oup / ratio), 8)
        new_channels = oup - init_channels

        self.primary_conv = nn.Sequential(
            nn.Conv2d(
                inp,
                init_channels,
                kernel_size,
                stride,
                kernel_size // 2,
                bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(
                inplace=True) if relu else nn.Sequential(),
        )

        padding = (dw_size - 1) // 2 * dilation

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(
                init_channels,
                new_channels,
                dw_size,
                1,
                padding=padding,
                groups=init_channels,
                bias=False,
                dilation=dilation),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(
                inplace=True) if relu else nn.Sequential(),
        )
        self.concat = Concat(dim=1)

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = self.concat(x1, x2)
        return out

class GhostBottleneck(nn.Module):
    def __init__(
            self,
            inp,
            expand_ratio,
            oup,
            kernel_size,
            stride,
            use_se,
            ratio=2,
            dilation=1,
            constrain1=False,
            constrain2=False):
        super(GhostBottleneck, self).__init__()
        assert stride in [1, 2]

        hidden_dim = _make_divisible(inp * expand_ratio // 2, 8) * 2
        if constrain1:
            if hidden_dim > 64:
                hidden_dim = _make_divisible(inp * expand_ratio // 4, 8) * 4
        if constrain2:
            if hidden_dim > 512:
                hidden_dim = 512

        self.conv = nn.Sequential(
            # pw
            GhostModule(
                inp,
                hidden_dim,
                kernel_size=1,
                relu=True,
                ratio=ratio,
                dw_size=3,
                dilation=dilation),
            # dw_size=kernel_size or 3
            # dw
            depthwise_conv(hidden_dim, hidden_dim, kernel_size, stride, relu=False,
                           dilation=dilation) if stride == 2 else nn.Sequential(),
            # Squeeze-and-Excite
            SELayer(hidden_dim) if use_se else nn.Sequential(),
            # pw-linear
            GhostModule(
                hidden_dim,
                oup,
                kernel_size=1,
                relu=False,
                ratio=ratio,
                dw_size=3,
                dilation=dilation),
            # dw_size=kernel_size or 3
        )

        if stride == 1 and inp == oup:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                depthwise_conv(inp, inp, 3, stride, relu=True),
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        self.eltwise = Eltwise()

    def forward(self, x):
        out = self.eltwise(self.conv(x), self.shortcut(x))
        return out

class ConvBNReLU(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            groups=1):
        super(ConvBNReLU, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class InvertedResidual(nn.Module):

    def __init__(self, inp, oup, stride, expand_ratio, kernel_size=3):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]
        self.use_res_connect = stride == 1 and inp == oup
        layers = list()
        # inter_channels = int(round(inp * expand_ratio))
        inter_channels = _make_divisible(inp * expand_ratio, 8)

        if inter_channels > 64:
            inter_channels = _make_divisible(inp * expand_ratio // 2, 8) * 2


        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, inter_channels, 1))
        layers.extend([
            # dw
            ConvBNReLU(
                inter_channels,
                inter_channels,
                kernel_size,
                stride,
                groups=inter_channels),
            # pw-linear
            nn.Conv2d(inter_channels, oup, 1, bias=False),
            nn.BatchNorm2d(oup)])
        self.conv = nn.Sequential(*layers)
        self.eltwise = Eltwise()

    def forward(self, x):
        if self.use_res_connect:
            return self.eltwise(x, self.conv(x))
        else:
            return self.conv(x)

def WarpperBlock(input_channel, output_channel, stride, block_choice):

    if block_choice == 0:
        return GhostBottleneck(inp=input_channel, expand_ratio=1.0/8.0, oup=output_channel, kernel_size=3, stride=stride,
                               use_se=False, ratio=2, constrain1=False)
    elif block_choice == 1:
        return GhostBottleneck(inp=input_channel, expand_ratio=2.0/8.0, oup=output_channel, kernel_size=3, stride=stride,
                               use_se=False, ratio=2, constrain1=False)
    elif block_choice == 2:
        return GhostBottleneck(inp=input_channel, expand_ratio=3.0/8.0, oup=output_channel, kernel_size=3, stride=stride,
                               use_se=False, ratio=2, constrain1=False)
    elif block_choice == 3:
        return GhostBottleneck(inp=input_channel, expand_ratio=4.0/8.0, oup=output_channel, kernel_size=3, stride=stride,
                               use_se=False, ratio=2, constrain1=False)
    elif block_choice == 4:
        return GhostBottleneck(inp=input_channel, expand_ratio=5.0/8.0, oup=output_channel, kernel_size=3, stride=stride,
                               use_se=False, ratio=2, constrain1=False)
    elif block_choice == 5:
        return GhostBottleneck(inp=input_channel, expand_ratio=6.0/8.0, oup=output_channel, kernel_size=3, stride=stride,
                               use_se=False, ratio=2, constrain1=False)
    elif block_choice == 6:
        return GhostBottleneck(inp=input_channel, expand_ratio=7.0/8.0, oup=output_channel, kernel_size=3, stride=stride,
                               use_se=False, ratio=2, constrain1=False)
    elif block_choice == 7:
        return GhostBottleneck(inp=input_channel, expand_ratio=1.0, oup=output_channel, kernel_size=3, stride=stride,
                               use_se=False, ratio=2, constrain1=False)
    elif block_choice == 8:
        return GhostBottleneck(inp=input_channel, expand_ratio=1.25, oup=output_channel, kernel_size=3, stride=stride,
                               use_se=False, ratio=2, constrain1=False)
    elif block_choice == 9:
        return GhostBottleneck(inp=input_channel, expand_ratio=1.5, oup=output_channel, kernel_size=3, stride=stride,
                               use_se=False, ratio=2, constrain1=False)
    elif block_choice == 10:
        return GhostBottleneck(inp=input_channel, expand_ratio=1.75, oup=output_channel, kernel_size=3, stride=stride,
                               use_se=False, ratio=2, constrain1=False)
    elif block_choice == 11:
        return GhostBottleneck(inp=input_channel, expand_ratio=2.0, oup=output_channel, kernel_size=3, stride=stride,
                               use_se=False, ratio=2, constrain1=False)
    elif block_choice == 12:
        return GhostBottleneck(inp=input_channel, expand_ratio=2.25, oup=output_channel, kernel_size=3, stride=stride,
                               use_se=False, ratio=2, constrain1=False)
    elif block_choice == 13:
        return GhostBottleneck(inp=input_channel, expand_ratio=2.5, oup=output_channel, kernel_size=3, stride=stride,
                               use_se=False, ratio=2, constrain1=False)
    elif block_choice == 14:
        return GhostBottleneck(inp=input_channel, expand_ratio=2.75, oup=output_channel, kernel_size=3, stride=stride,
                               use_se=False, ratio=2, constrain1=False)
    elif block_choice == 15:
        return GhostBottleneck(inp=input_channel, expand_ratio=3, oup=output_channel, kernel_size=3, stride=stride,
                               use_se=False, ratio=2, constrain1=False)
    elif block_choice == 16:
        return GhostBottleneck(inp=input_channel, expand_ratio=4, oup=output_channel, kernel_size=3, stride=stride,
                               use_se=False, ratio=2, constrain1=False)
    elif block_choice == 17:
        return GhostBottleneck(inp=input_channel, expand_ratio=5, oup=output_channel, kernel_size=3, stride=stride,
                               use_se=False, ratio=2, constrain1=False)
    elif block_choice == 18:
        return GhostBottleneck(inp=input_channel, expand_ratio=6, oup=output_channel, kernel_size=3, stride=stride,
                               use_se=False, ratio=2, constrain1=False)
    elif block_choice == 19:
        return Identity()


class NNflowNasSeg(nn.Module):
    def __init__(
            self,
            n_class=1000,
            model_size=0.7,
            dropout_rate=0.2,
            architecture=None):
        super(NNflowNasSeg, self).__init__()

        self.model_size = model_size
        self.version = 'v0.0.1.200831'
        if model_size == 0.091: #face
            self.stride = [1, 2, 2, 2]
            self.stage_repeats = [1, 3, 3, 9]
            self.stage_out_channels = [16, 32, 48, 96, 256]
            last_conv_out_channel = 1024
            input_channel = 16

        elif model_size == 0.1: #edu
            self.stride = [2, 2, 2, 2]
            self.stage_repeats = [1, 3, 3, 9]
            self.stage_out_channels = [16, 32, 64, 128, 384]
            last_conv_out_channel = 1024
            input_channel = 8


        self.down_block1_channel = input_channel
        self.use_se = False
        self.dropout_rate = dropout_rate
        if architecture is None:
            fix_arch = False
        elif architecture is not None:
            fix_arch = True
        self.fix_arch = fix_arch

        self.first_conv = conv_bn_relu(3, input_channel, kernel_size=3, stride=2)

        if model_size == 0.091:
            self.first_conv = conv_bn_relu(3, input_channel, stride=4, kernel_size=4)

        block_id = 0
        features = []
        out_stride = 2
        self.down_block1_channel = input_channel
        self.down_block1 = None

        for stage_id in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[stage_id]
            output_channel = self.stage_out_channels[stage_id]
            for i in range(numrepeat):
                if self.stride[stage_id] == 2 and i == 0:
                    stride = 2
                    out_stride = out_stride * stride
                else:
                    stride = 1

                block_choice = architecture[block_id]
                block_id += 1

                features.append(
                    WarpperBlock(
                        input_channel,
                        output_channel,
                        stride=stride,
                        block_choice=block_choice))

                input_channel = output_channel
            if out_stride == 2 and self.stride[stage_id + 1] == 2:
                self.down_block1 = nn.Sequential(*features)
                self.down_block1_channel = input_channel
                features = []
                self.stage1_num = block_id

            elif out_stride == 4 and self.stride[stage_id + 1] == 2:
                self.down_block2 = nn.Sequential(*features)
                self.down_block2_channel = input_channel
                features = []
                self.stage2_num = block_id

            elif out_stride == 8 and self.stride[stage_id + 1] == 2:
                self.down_block3 = nn.Sequential(*features)
                self.down_block3_channel = input_channel
                features = []
                self.stage3_num = block_id

            elif out_stride == 16 and (stage_id == len(self.stage_repeats) - 1 or self.stride[stage_id + 1] == 2):
                self.down_block4 = nn.Sequential(*features)
                self.down_block4_channel = input_channel
                features = []
                self.stage4_num = block_id

            elif out_stride == 32 and stage_id == len(self.stage_repeats) - 1:
                self.down_block5 = nn.Sequential(*features)
                self.down_block5_channel = input_channel
                features = []
                self.stage5_num = block_id

        self.out_stride = out_stride

        # self.final_expand_layer = conv_bn_relu(self.stage_out_channels[-2], self.stage_out_channels[-1],
        #                                      kernel_size=1)
        # self.globalpool = nn.AdaptiveAvgPool2d((1, 1))
        #
        # self.feature_mix_layer = nn.Sequential(
        #     nn.Linear(self.stage_out_channels[-1], last_conv_out_channel, bias=True),
        #     nn.BatchNorm1d(last_conv_out_channel),
        #     nn.ReLU(inplace=True),
        # )
        # self.dropout = nn.Dropout(self.dropout_rate)
        # self.output = nn.Linear(last_conv_out_channel, n_class, bias=True)
        self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def forward(self, x):
        down1 = self.first_conv(x)
        if self.down_block1 is not None:
            down1 = self.down_block1(down1)
            down2 = self.down_block2(down1)
            down3 = self.down_block3(down2)
            down4 = self.down_block4(down3)
            down5 = self.down_block5(down4) if self.out_stride == 32 else down4

        else:
            down2 = self.down_block2(down1)
            down3 = self.down_block3(down2)
            down4 = self.down_block4(down3)
            down5 = self.down_block5(down4) if self.out_stride == 32 else down4

        # x = self.final_expand_layer(down5)

        # x = self.globalpool(x)
        #
        # x = x.view(x.data.size(0), -1)
        # x = self.feature_mix_layer(x)
        # x = self.dropout(x)
        # x = self.output(x)
        # return x
        if self.model_size == 0.091:
            return down3, x
        else:
            return down4, down5


def get_nnflow_nas(n_class=1000,
                   flops=189.5,
                   resume=None,
                   dropout_rate=0.2,
                   architecture=None,
                   model_size=None):


    if flops == 28.7:
        model_size = 0.1
        architecture = [7, 8, 8, 8, 3, 3, 3, 16, 8, 9, 13, 7, 4, 4, 4, 9]


    net = NNflowNasSeg(n_class=n_class, model_size=model_size, architecture=architecture, dropout_rate=dropout_rate)
    if resume is not None:
        net = nn.DataParallel(net, device_ids=[0])
        device = torch.device("cuda")
        net = net.to(device)
        checkpoint = torch.load(resume, map_location='cpu')
        load_checkpoint(net, checkpoint, strict=True)
        print ('load model success')

        # torch.save(net.state_dict(), 'model.pth.tar', _use_new_zipfile_serialization=False)

    return net

if __name__ == "__main__":
    net = get_nnflow_nas(flops=20)
    # exit()
    # from byted_nnflow.compression.torch_frame import ByteOicsrPruner
    # net = net.cuda()
    # input_shape = (1, 3, 120, 120)
    # total_flops, param = ByteOicsrPruner.cal_network_flops_param(net, input_shape)
    # print(total_flops/1000000.0, param)


    exit()