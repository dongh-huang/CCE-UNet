import torch
import torch.nn as nn
from attention import cbam_block, eca_block
from nets.resnet import resnet50
from nets.vgg import VGG16
from non import NonLocalBlock
attention_blocks = [cbam_block, eca_block]

class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs

class MixPool(nn.Module):
    def __init__(self, in_c, out_c):
        super(MixPool, self).__init__()
        self.fmask = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c, out_c//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c//2),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_c, out_c//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c//2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, m):
        fmask = (self.fmask(x) > 0.5).type(torch.cuda.FloatTensor)
        m = nn.MaxPool2d((m.shape[2]//x.shape[2], m.shape[3]//x.shape[3]))(m)
        x1 = x * torch.logical_or(fmask, m).type(torch.cuda.FloatTensor)
        x1 = self.conv1(x1)
        x2 = self.conv2(x)
        x = torch.cat([x1, x2], axis=1)
        return x

class CIFM(nn.Module):
    def __init__(self, in_channels, out_channels, dilations=[1, 4, 8, 12], use_non_local=True):
        super(CIFM, self).__init__()
        self.aspp_blocks = nn.ModuleList()
        for dilation in dilations:
            self.aspp_blocks.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False)
            )
        self.global_pooling = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.conv_fusion = nn.Conv2d(out_channels * (len(dilations) + 1), in_channels, kernel_size=1, bias=False)
        self.relu_fusion = nn.ReLU(inplace=True)

        # 添加Non-Local模块
        if use_non_local:
            self.non_local_block = NonLocalBlock(in_channels)
        else:
            self.non_local_block = lambda x: x  # 如果不使用Non-Local，就是一个恒等函数

        # 应用L2正则化到可学习的参数
        for block in self.aspp_blocks:
            nn.utils.weight_norm(block)
        nn.utils.weight_norm(self.global_pooling[1])  # 只对Conv2d层应用权重规范化
        nn.utils.weight_norm(self.conv_fusion)

    def forward(self, x):
        x = self.non_local_block(x)

        aspp_outs = []
        for aspp_block in self.aspp_blocks:
            aspp_outs.append(aspp_block(x))
        global_pooling_out = self.global_pooling(x)
        global_pooling_out = nn.Upsample(size=x.size()[2:], mode='bilinear', align_corners=False)(global_pooling_out)
        out = torch.cat([*aspp_outs, global_pooling_out], dim=1)
        out = self.conv_fusion(out)
        out = self.relu_fusion(out)

        return out

class Unet(nn.Module):
    def __init__(self, num_classes=3, pretrained=False, backbone='vgg', phi=1):
        super(Unet, self).__init__()
        if backbone == 'vgg':
            self.vgg = VGG16(pretrained=pretrained)
            in_filters = [192, 384, 768, 1024]
        elif backbone == "resnet50":
            self.resnet = resnet50(pretrained=pretrained)
            in_filters = [192, 512, 1024, 3072]
        else:
            raise ValueError('Unsupported backbone - `{}`, Use vgg, resnet50.'.format(backbone))
        out_filters = [64, 128, 256, 512]

        # upsampling
        # self.up_concat5 = unetUp(in_filters[4], out_filters[4])
        # 64,64,512
        self.up_concat4 = unetUp(in_filters[3], out_filters[3])
        # 128,128,256
        self.up_concat3 = unetUp(in_filters[2], out_filters[2])
        # 256,256,128
        self.up_concat2 = unetUp(in_filters[1], out_filters[1])
        # 512,512,64
        self.up_concat1 = unetUp(in_filters[0], out_filters[0])

        # CIFM模块
        self.aspp = CIFM(in_channels=512, out_channels=256)

        if backbone == 'resnet50':
            self.up_conv = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
                nn.ReLU(),
            )
        else:
            self.up_conv = None

        self.final = nn.Conv2d(out_filters[0], num_classes, 1)

        self.backbone = backbone

        self.phi = phi  # 控制注意力机制的使用
        if phi >= 1:
            # CBAM注意力机制用于浅层特征
            # ECA注意力机制用于深层特征
            self.feat1_attention = cbam_block(64)  # 第一层使用CBAM
            self.feat2_attention = cbam_block(128)  # 第二层使用CBAM
            self.feat3_attention = None  # 第三层不使用注意力机制
            self.feat4_attention = eca_block(512)  # 第四层使用ECA
            self.feat5_attention = eca_block(1024)  # 第五层使用ECA

    def forward(self, inputs):
        if self.backbone == "vgg":
            [feat1, feat2, feat3, feat4, feat5] = self.vgg.forward(inputs)
        elif self.backbone == "resnet50":
            [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(inputs)

        feat5 = self.aspp(feat5)
        # Add MixPool after feat5
        # up_mixpool = MixPool(512, 512)  # Create MixPool instance
        # up5 = self.up_concat5(feat5, up_mixpool(feat5))  # Apply MixPool to feat5

        if self.phi >= 1:
            feat1 = self.feat1_attention(feat1)
            feat2 = self.feat2_attention(feat2)
            # feat3 不应用注意力机制
            feat4 = self.feat4_attention(feat4)
            feat5 = self.feat5_attention(feat5)

        up4 = self.up_concat4(feat4, feat5)
        up3 = self.up_concat3(feat3, up4)
        up2 = self.up_concat2(feat2, up3)
        up1 = self.up_concat1(feat1, up2)

        if self.up_conv is not None:
            up1 = self.up_conv(up1)

        final = self.final(up1)

        return final

    def freeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = False
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = True
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = True
