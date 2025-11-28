


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch as t


def conv1x1(in_planes, out_planes, stride=1):

    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ChannelSpatialAttn(nn.Module):


    def __init__(self, channel, reduction=16):
        super(ChannelSpatialAttn, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_gate = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(channel, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        n, c, h, w = x.size()
        avg_out = self.avg_pool(x).view(n, c)
        channel_weights = self.channel_gate(avg_out).view(n, c, 1, 1)
        x_channel = x * channel_weights.expand_as(x)
        spatial_weights = self.spatial_gate(x_channel)
        out = x_channel * spatial_weights.expand_as(x)
        return out

class AATN(nn.Module):

        def __init__(self, cfg):
            super(AATN, self).__init__()
            channels = cfg.TRAIN.aatnchannel

            self.conv_shape = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, 4, kernel_size=3, stride=1, padding=1),
            )

            self.conv1 = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),  # 添加 padding 以保持尺寸
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),  # 添加 padding 以保持尺寸
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            )

            self.schwatt_res = ChannelSpatialAttn(channels, reduction=16)

            for modules in [self.conv_shape, self.conv1, self.conv2]:
                for l in modules.modules():
                    if isinstance(l, nn.Conv2d):
                        t.nn.init.normal_(l.weight, std=0.01)
                        if l.bias is not None:
                            t.nn.init.constant_(l.bias, 0)


        def xcorr_depthwise(self, x, kernel):
            batch = kernel.size(0)
            channel = kernel.size(1)
            x = x.view(1, batch * channel, x.size(2), x.size(3))
            kernel = kernel.view(batch * channel, 1, kernel.size(2), kernel.size(3))
            out = F.conv2d(x, kernel, groups=batch * channel)
            out = out.view(batch, channel, out.size(2), out.size(3))
            return out

        def forward(self, x, z):
            x = self.conv1(x)
            z = self.conv2(z)

            res = self.xcorr_depthwise(x, z)
            res = self.schwatt_res(res)
            shape_pred = self.conv_shape(res)

            return shape_pred, res


class clsandloc(nn.Module):

    def __init__(self, cfg):
        super(clsandloc, self).__init__()
        channel = cfg.TRAIN.clsandlocchannel
        aatn_channel = cfg.TRAIN.aatnchannel

        self.conv1 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),  # 添加 padding 以保持尺寸
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),  # 添加 padding 以保持尺寸
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
        )

        self.convloc = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(cfg.TRAIN.groupchannel, channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(cfg.TRAIN.groupchannel, channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, 4, kernel_size=3, stride=1, padding=1),
        )
        self.convcls = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(cfg.TRAIN.groupchannel, channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(cfg.TRAIN.groupchannel, channel),
            nn.ReLU(inplace=True),
        )
        self.conv_offset = nn.Sequential(
            nn.Conv2d(aatn_channel, channel, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(cfg.TRAIN.groupchannel, channel),
            nn.ReLU(inplace=True),
        )

        self.add = nn.ConvTranspose2d(channel * 2, channel, 3, 1, padding=1)  # 修正 TransposeConv 的 padding
        self.resize = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(cfg.TRAIN.groupchannel, channel),
            nn.ReLU(inplace=True),
        )

        self.scatt_final = ChannelSpatialAttn(channel, reduction=16)

        self.relu = nn.ReLU(inplace=True)

        self.cls1 = nn.Conv2d(channel, 2, kernel_size=3, stride=1, padding=1)
        self.cls2 = nn.Conv2d(channel, 2, kernel_size=3, stride=1, padding=1)
        self.cls3 = nn.Conv2d(channel, 1, kernel_size=3, stride=1, padding=1)


    def xcorr_depthwise(self, x, kernel):
        batch = kernel.size(0)
        channel = kernel.size(1)
        x = x.view(1, batch * channel, x.size(2), x.size(3))
        kernel = kernel.view(batch * channel, 1, kernel.size(2), kernel.size(3))
        out = F.conv2d(x, kernel, groups=batch * channel)
        out = out.view(batch, channel, out.size(2), out.size(3))
        return out

    def forward(self, x, z, ress):

        x = self.conv1(x)
        z = self.conv2(z)
        res = self.xcorr_depthwise(x, z)


        res = self.resize(res)

        ress = self.conv_offset(ress)

        res = self.add(self.relu(t.cat((res, ress), 1)))

        res = self.scatt_final(res)

        cls = self.convcls(res)
        cls1 = self.cls1(cls)
        cls2 = self.cls2(cls)
        cls3 = self.cls3(cls)

        loc = self.convloc(res)

        return cls1, cls2, cls3, loc