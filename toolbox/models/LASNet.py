import os
import torch.nn as nn
import torch
from resnet import Backbone_ResNet152_in3
import torch.nn.functional as F
import numpy as np
from toolbox.dual_self_att import CAM_Module


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        #self.relu = nn.ReLU(inplace=True)
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super(ChannelAttention, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)


class CorrelationModule(nn.Module):
    def  __init__(self, all_channel=64):
        super(CorrelationModule, self).__init__()
        self.linear_e = nn.Linear(all_channel, all_channel,bias = False)
        self.channel = all_channel
        self.fusion = BasicConv2d(all_channel, all_channel, kernel_size=3, padding=1)

    def forward(self, exemplar, query): # exemplar: middle, query: rgb or T
        fea_size = exemplar.size()[2:]
        all_dim = fea_size[0]*fea_size[1]
        exemplar_flat = exemplar.view(-1, self.channel, all_dim) #N,C,H*W
        query_flat = query.view(-1, self.channel, all_dim)
        exemplar_t = torch.transpose(exemplar_flat,1,2).contiguous()  #batchsize x dim x num, N,H*W,C
        exemplar_corr = self.linear_e(exemplar_t) #
        A = torch.bmm(exemplar_corr, query_flat)
        B = F.softmax(torch.transpose(A,1,2),dim=1)
        exemplar_att = torch.bmm(query_flat, B).contiguous()

        exemplar_att = exemplar_att.view(-1, self.channel, fea_size[0], fea_size[1])
        exemplar_out = self.fusion(exemplar_att)

        return exemplar_out

class CLM(nn.Module):
    def __init__(self, all_channel=64):
        super(CLM, self).__init__()
        self.corr_x_2_x_ir = CorrelationModule(all_channel)
        self.corr_ir_2_x_ir = CorrelationModule(all_channel)
        self.smooth1 = BasicConv2d(all_channel, all_channel, kernel_size=3, padding=1)
        self.smooth2 = BasicConv2d(all_channel, all_channel, kernel_size=3, padding=1)
        self.fusion = BasicConv2d(2*all_channel, all_channel, kernel_size=3, padding=1)
        self.pred = nn.Conv2d(all_channel, 2, kernel_size=3, padding=1, bias = True)

    def forward(self, x, x_ir, ir):  # exemplar: middle, query: rgb or T
        corr_x_2_x_ir = self.corr_x_2_x_ir(x_ir,x)
        corr_ir_2_x_ir = self.corr_ir_2_x_ir(x_ir,ir)

        summation = self.smooth1(corr_x_2_x_ir + corr_ir_2_x_ir)
        multiplication = self.smooth2(corr_x_2_x_ir * corr_ir_2_x_ir)

        fusion = self.fusion(torch.cat([summation,multiplication],1))
        sal_pred = self.pred(fusion)

        return fusion, sal_pred


class CAM(nn.Module):
    def __init__(self, all_channel=64):
        super(CAM, self).__init__()
        #self.conv1 = BasicConv2d(all_channel, all_channel, kernel_size=3, padding=1)
        self.conv2 = BasicConv2d(all_channel, all_channel, kernel_size=3, padding=1)
        self.sa = SpatialAttention()
        # self-channel attention
        self.cam = CAM_Module(all_channel)

    def forward(self, x, ir):
        multiplication = x * ir
        summation = self.conv2(x + ir)

        sa = self.sa(multiplication)
        summation_sa = summation.mul(sa)

        sc_feat = self.cam(summation_sa)

        return sc_feat


class ESM(nn.Module):
    def __init__(self, all_channel=64):
        super(ESM, self).__init__()
        self.conv1 = BasicConv2d(all_channel, all_channel, kernel_size=3, padding=1)
        self.conv2 = BasicConv2d(all_channel, all_channel, kernel_size=3, padding=1)
        self.dconv1 = BasicConv2d(all_channel,int( all_channel/4), kernel_size=3, padding=1)
        self.dconv2 = BasicConv2d(all_channel,int( all_channel/4), kernel_size=3, dilation=3, padding=3)
        self.dconv3 = BasicConv2d(all_channel,int( all_channel/4), kernel_size=3, dilation=5, padding=5)
        self.dconv4 = BasicConv2d(all_channel,int( all_channel/4), kernel_size=3, dilation=7, padding=7)
        self.fuse_dconv = nn.Conv2d(all_channel, all_channel, kernel_size=3,padding=1)
        self.pred = nn.Conv2d(all_channel, 2, kernel_size=3, padding=1, bias = True)

    def forward(self, x, ir):
        multiplication = self.conv1(x * ir)
        summation = self.conv2(x + ir)
        fusion = (summation + multiplication)
        x1 = self.dconv1(fusion)
        x2 = self.dconv2(fusion)
        x3 = self.dconv3(fusion)
        x4 = self.dconv4(fusion)
        out = self.fuse_dconv(torch.cat((x1, x2, x3, x4), dim=1))
        edge_pred = self.pred(out)

        return out, edge_pred


class prediction_decoder(nn.Module):
    def __init__(self, channel1=64, channel2=128, channel3=256, channel4=256, channel5=512, n_classes=9):
        super(prediction_decoder, self).__init__()
        # 15 20
        self.decoder5 = nn.Sequential(
                nn.Dropout2d(p=0.1),
                BasicConv2d(channel5, channel5, kernel_size=3, padding=3, dilation=3),
                BasicConv2d(channel5, channel4, kernel_size=3, padding=1),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                )
        # 30 40
        self.decoder4 = nn.Sequential(
                nn.Dropout2d(p=0.1),
                BasicConv2d(channel4, channel4, kernel_size=3, padding=3, dilation=3),
                BasicConv2d(channel4, channel3, kernel_size=3, padding=1),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                )
        # 60 80
        self.decoder3 = nn.Sequential(
                nn.Dropout2d(p=0.1),
                BasicConv2d(channel3, channel3, kernel_size=3, padding=3, dilation=3),
                BasicConv2d(channel3, channel2, kernel_size=3, padding=1),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                )
        # 120 160
        self.decoder2 = nn.Sequential(
                nn.Dropout2d(p=0.1),
                BasicConv2d(channel2, channel2, kernel_size=3, padding=3, dilation=3),
                BasicConv2d(channel2, channel1, kernel_size=3, padding=1),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                )
        self.semantic_pred2 = nn.Conv2d(channel1, n_classes, kernel_size=3, padding=1)
        # 240 320 -> 480 640
        self.decoder1 = nn.Sequential(
                nn.Dropout2d(p=0.1),
                BasicConv2d(channel1, channel1, kernel_size=3, padding=3, dilation=3),
                BasicConv2d(channel1, channel1, kernel_size=3, padding=1),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), # 480 640
                BasicConv2d(channel1, channel1, kernel_size=3, padding=1),
                nn.Conv2d(channel1, n_classes, kernel_size=3, padding=1)
                )

    def forward(self, x5, x4, x3, x2, x1):
        x5_decoder = self.decoder5(x5)
        # for PST900 dataset
        # since the input size is 720x1280, the size of x5_decoder and x4_decoder is 23 and 45, so we cannot use 2x upsampling directrly.
        # x5_decoder = F.interpolate(x5_decoder, size=fea_size, mode="bilinear", align_corners=True)
        x4_decoder = self.decoder4(x5_decoder + x4)
        x3_decoder = self.decoder3(x4_decoder + x3)
        x2_decoder = self.decoder2(x3_decoder + x2)
        semantic_pred2 = self.semantic_pred2(x2_decoder)
        semantic_pred = self.decoder1(x2_decoder + x1)

        return semantic_pred,semantic_pred2


class LASNet(nn.Module):
    def __init__(self, n_classes):
        super(LASNet, self).__init__()

        (
            self.layer1_rgb,
            self.layer2_rgb,
            self.layer3_rgb,
            self.layer4_rgb,
            self.layer5_rgb,
        ) = Backbone_ResNet152_in3(pretrained=True)

        # reduce the channel number, input: 480 640
        self.rgbconv1 = BasicConv2d(64, 64, kernel_size=3, padding=1)   # 240 320
        self.rgbconv2 = BasicConv2d(256, 128, kernel_size=3, padding=1)  # 120 160
        self.rgbconv3 = BasicConv2d(512, 256, kernel_size=3, padding=1)  # 60 80
        self.rgbconv4 = BasicConv2d(1024, 256, kernel_size=3, padding=1) # 30 40
        self.rgbconv5 = BasicConv2d(2048, 512, kernel_size=3, padding=1) # 15 20

        self.CLM5 = CLM(512)
        self.CAM4 = CAM(256)
        self.CAM3 = CAM(256)
        self.CAM2 = CAM(128)
        self.ESM1 = ESM(64)

        self.decoder = prediction_decoder(64,128,256,256,512, n_classes)

    def forward(self, rgb, depth):
        x = rgb
        ir = depth[:, :1, ...]
        ir = torch.cat((ir, ir, ir), dim=1)

        x1 = self.layer1_rgb(x)
        x2 = self.layer2_rgb(x1)
        x3 = self.layer3_rgb(x2)
        x4 = self.layer4_rgb(x3)
        x5 = self.layer5_rgb(x4)

        ir1 = self.layer1_rgb(ir)
        ir2 = self.layer2_rgb(ir1)
        ir3 = self.layer3_rgb(ir2)
        ir4 = self.layer4_rgb(ir3)
        ir5 = self.layer5_rgb(ir4)

        x1 = self.rgbconv1(x1)
        x2 = self.rgbconv2(x2)
        x3 = self.rgbconv3(x3)
        x4 = self.rgbconv4(x4)
        x5 = self.rgbconv5(x5)

        ir1 = self.rgbconv1(ir1)
        ir2 = self.rgbconv2(ir2)
        ir3 = self.rgbconv3(ir3)
        ir4 = self.rgbconv4(ir4)
        ir5 = self.rgbconv5(ir5)

        out5, sal  = self.CLM5(x5, x5*ir5, ir5)
        out4 = self.CAM4(x4, ir4)
        out3 = self.CAM3(x3, ir3)
        out2 = self.CAM2(x2, ir2)
        out1, edge = self.ESM1(x1, ir1)

        semantic, semantic2 = self.decoder(out5, out4, out3, out2, out1)
        semantic2 = torch.nn.functional.interpolate(semantic2, scale_factor=2, mode='bilinear')
        sal = torch.nn.functional.interpolate(sal, scale_factor=32, mode='bilinear')
        edge = torch.nn.functional.interpolate(edge, scale_factor=2, mode='bilinear')


        return semantic, semantic2, sal, edge

if __name__ == '__main__':
    LASNet(9)
    # for PST900 dataset
    # LASNet(5)
