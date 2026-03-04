# ------------------------------------------------------------------------------
# Written by Yiwen Bai (wen1109@stud.tjut.edu.cn)
# ------------------------------------------------------------------------------
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from model_utils_speed import BasicBlock, Bottleneck, DSBasicBlock, DSBottleneck, segmenthead, DAPPM, PAPPM, DCPP_A, DCPP_B, DCPP_C, L_DGF, L_SGF, Bag, Light_Bag, TGF
import logging

BatchNorm2d = nn.BatchNorm2d
bn_mom = 0.1
algc = False


def parse_args():
    parser = argparse.ArgumentParser(description='Speed Measurement')

    parser.add_argument('--a', help='lcfnet-slim, lcfnet-standard or lcfnet-super', default='lcfnet-slim', type=str)
    parser.add_argument('--c', help='number of classes', type=int, default=19)
    parser.add_argument('--r', help='input resolution', type=int, nargs='+', default=(1024, 2048))

    args = parser.parse_args()

    return args

class LCFNet(nn.Module):

    def __init__(self, m=2, n=3, num_classes=19, planes=64, ppm_planes=96, head_planes=128, augment=True):
        super(LCFNet, self).__init__()
        self.augment = augment

        self.relu = nn.ReLU(inplace=True)

        # Select different fusion modules-----------------------------------------------------------------
        self.fusion3 = TGF(planes * 4, planes * 4)
        # self.fusion3 = Light_Bag(planes * 4, planes * 4)
        # self.fusion3 = Bag(planes * 4, planes * 4)
        # ------------------------------------------------------------------------------------------------

        self.compression3 = nn.Sequential(
            nn.Conv2d(planes * 4, planes * 4, kernel_size=1, bias=False),
            BatchNorm2d(planes * 4, momentum=bn_mom),
        )

        self.compression4 = nn.Sequential(
            nn.Conv2d(planes * 4, planes * 4, kernel_size=1, bias=False),
            BatchNorm2d(planes * 4, momentum=bn_mom),
        )

        # Semantic Branch--------------------------------------------------------------------------------

        # Select different pyramid pooling modules-------------------------------------------------------
        # self.spp = PAPPM(planes * 16, ppm_planes, planes * 4)
        # self.spp = DAPPM(planes * 16, ppm_planes, planes * 4)
        # self.spp = DCPP_A(planes * 16, ppm_planes, planes * 4)
        self.spp = DCPP_B(planes * 16, ppm_planes, planes * 4)
        # self.spp = DCPP_C(planes * 16, ppm_planes, planes * 4)
        # -----------------------------------------------------------------------------------------------

        self.layer1_s = self._make_layer(BasicBlock, planes, planes * 2, m, stride=2)
        self.layer2_s = self._make_layer(BasicBlock, planes * 2, planes * 4, n, stride=2)
        self.layer3_s = self._make_layer(BasicBlock, planes * 4, planes * 8, n, stride=2)
        self.layer4_s = self._make_layer(Bottleneck, planes * 8, planes * 8, 2, stride=2)

        self.compression1 = nn.Sequential(
            nn.Conv2d(planes * 2, planes * 4, kernel_size=1, bias=False),
            BatchNorm2d(planes * 4, momentum=bn_mom),
        )
        self.fusion1 = L_SGF(planes * 4, planes * 4)
        self.compression5 = nn.Sequential(
            nn.Conv2d(planes * 4, planes * 2, kernel_size=1, bias=False),
            BatchNorm2d(planes * 2, momentum=bn_mom),
        )

        # Detail Branch----------------------------------------------------------------------------------
        self.compression2 = nn.Sequential(
            nn.Conv2d(planes * 4, planes * 2, kernel_size=1, bias=False),
            BatchNorm2d(planes * 2, momentum=bn_mom),
        )
        self.fusion2 = L_DGF(planes * 2, planes * 2)

        self.layer1_d = self._make_layer(BasicBlock, planes * 2, planes * 2, m)
        self.layer2_d = self._make_layer(BasicBlock, planes * 2, planes * 2, m)
        self.layer3_d = self._make_layer(Bottleneck, planes * 2, planes * 2, 1, stride=2)

        # LCFNet-slim------------------------------------------------------------------------------------
        # self.layer1_d = self._make_layer(DSBasicBlock, planes * 2, planes * 2, m)
        # self.layer3_d = self._make_layer(DSBottleneck, planes * 2, planes * 2, 1, stride=2)
        # -----------------------------------------------------------------------------------------------

        # Compensation Branch----------------------------------------------------------------------------
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, planes, kernel_size=3, stride=2, padding=1),
            BatchNorm2d(planes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, stride=2, padding=1),
            BatchNorm2d(planes, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(BasicBlock, planes, planes, m)
        self.layer1_c = self._make_layer(BasicBlock, planes, planes * 2, m)

        if m == 2:
            self.layer2_c = self._make_single_layer(BasicBlock, planes * 2, planes * 2)
            self.layer3_c = self._make_layer(Bottleneck, planes * 4, planes * 2, 1)

        self.layer4_c = self._make_layer(Bottleneck, planes * 2, planes * 2, 1, stride=2)
        # -----------------------------------------------------------------------------------------------

        # Segmentation Head------------------------------------------------------------------------------
        if self.augment:
            self.seghead_d = segmenthead(planes * 2, head_planes, num_classes)
            self.seghead_s = segmenthead(planes * 2, planes, 1)

        self.final_layer = segmenthead(planes * 4, head_planes, num_classes)
        # -----------------------------------------------------------------------------------------------

        # Kaiming Initialization-------------------------------------------------------------------------
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # -----------------------------------------------------------------------------------------------

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=bn_mom),
                nn.ReLU(inplace=True),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            if i == (blocks - 1):
                layers.append(block(inplanes, planes, stride=1, no_relu=True))
            else:
                layers.append(block(inplanes, planes, stride=1, no_relu=False))

        return nn.Sequential(*layers)

    def _make_single_layer(self, block, inplanes, planes, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=bn_mom),
                nn.ReLU(inplace=True),
            )

        layer = block(inplanes, planes, stride, downsample, no_relu=True)

        return layer

    # -----------------------------------------------------------------------------------------------
    def forward(self, x):

        width_output = x.shape[-1] // 8  # 1/8
        height_output = x.shape[-2] // 8  # 1/8

        x = self.conv1(x)  # 1/4,planes,torch.Size([3, 64, 256, 256])
        # print(x.shape)
        x = self.layer1(x)  # 1/4,planes,torch.Size([3, 64, 256, 256])
        # print(x.shape)

        x_s = self.relu(self.layer1_s(self.relu(x)))  # 1/8, Semantic, planes*2,torch.Size([3, 128, 128, 128])
        # print(x_s.shape)
        x_c = self.layer1_c(x)  # 1/4, Compensation, planes*2,torch.Size([3, 128, 256, 256])
        # print(x_c.shape)
        x_d = self.layer1_d(x_c)  # 1/4, Detail, planes*2,torch.Size([3, 128, 256, 256])
        # print(x_d.shape)
        x_c = self.layer2_c(x_c)  # 1/4, Compensation, planes*2,torch.Size([3, 128, 256, 256])
        # print(x_c.shape)

        x_s = self.relu(self.layer2_s(x_s))  # 1/16, Semantic, planes*4,torch.Size([3, 256, 64, 64])
        # print(x_s.shape)
        x_c = self.compression1(x_c)  # 1/4, planes*4,torch.Size([3, 256, 256, 256])
        # print(x_c.shape)
        x_s1 = self.fusion1(x_c, x_s)  # 1/16, Semantic, planes*4,torch.Size([3, 256, 64, 64])
        # print(x_s1.shape)
        x_s2 = self.compression5(x_s1)

        if self.augment:
            temp_s = x_s2

        x_s = self.relu(self.layer3_s(x_s1))  # 1/32, Semantic, planes*8,torch.Size([3, 512, 32, 32])
        # print(x_s.shape)
        x_d = self.layer2_d(self.relu(x_d))  # 1/4, Detail, planes*2,torch.Size([3, 128, 256, 256])
        # print(x_d.shape)
        x_c = self.layer3_c(self.relu(x_c))  # 1/4, Compensation, planes*4,torch.Size([3, 256, 256, 256])
        # print(x_c.shape)
        x_c = self.compression2(x_c)
        x_d1 = self.fusion2(x_c, x_d)  # 1/4, Detail, planes*2,torch.Size([3, 128, 256, 256])
        # print(x_d1.shape)

        if self.augment:
            temp_d = x_d1

        x_d = self.layer3_d(self.relu(x_d1))  # 1/8, Detail, planes*4,torch.Size([3, 256, 128, 128])?
        # print(x_d.shape)
        # pdb.set_trace()
        x_c = self.layer4_c(self.relu(x_c))  # 1/8, Compensation, planes*4,torch.Size([3, 256, 128, 128])?
        # print(x_c.shape)
        x_s = F.interpolate(
            self.spp(self.layer4_s(x_s)),
            size=[height_output, width_output],
            mode='bilinear', align_corners=algc)  # 1/8, Semantic, planes*4,torch.Size([3, 256, 128, 128])
        # print(x_s.shape)

        x_d = self.compression3(x_d)  # 1/8, Detail, planes*4,torch.Size([3, 256, 128, 128])
        x_c = self.compression3(x_c)  # 1/8, Compensation, planes*4,torch.Size([3, 256, 128, 128])
        x_s = self.compression4(x_s)  # 1/8, Semantic, planes*4,torch.Size([3, 256, 128, 128])
        # print(x_d.shape)
        # print(x_c.shape)
        # print(x_s.shape)

        x_sum = x_d + x_c + x_s  # 1/8, planes*4,torch.Size([3, 256, 128, 128])
        # print(x_sum.shape)

        # To remove the TGF module, comment out self.final_layer------------------------------------
        # x = self.final_layer(x_sum)
        # -----------------------------------------------------------------------------------------

        x = self.final_layer(self.fusion3(x_d, x_c, x_s, x_sum))  # 1/8,torch.Size([3, 19, 128, 128])
        # print(x.shape)

        if self.augment:
            x_extra_d = self.seghead_d(temp_d)
            x_extra_s = self.seghead_s(temp_s)
            return [x_extra_d, x, x_extra_s]
        else:
            return x


def get_seg_model(cfg, imgnet_pretrained=False):
    if 'slim' in cfg.MODEL.NAME:
        model = LCFNet(m=2, n=3, num_classes=cfg.DATASET.NUM_CLASSES, planes=32, ppm_planes=96, head_planes=128,
                       augment=True)
    elif 'standard' in cfg.MODEL.NAME:
        model = LCFNet(m=2, n=3, num_classes=cfg.DATASET.NUM_CLASSES, planes=32, ppm_planes=96, head_planes=128,
                       augment=True)
    else:
        model = LCFNet(m=2, n=3, num_classes=cfg.DATASET.NUM_CLASSES, planes=64, ppm_planes=96, head_planes=128,
                       augment=True)

    if imgnet_pretrained:
        pretrained_state = torch.load(cfg.MODEL.PRETRAINED, map_location='cpu')['state_dict']
        model_dict = model.state_dict()
        pretrained_state = {k: v for k, v in pretrained_state.items() if
                            (k in model_dict and v.shape == model_dict[k].shape)}
        model_dict.update(pretrained_state)
        msg = 'Loaded {} parameters!'.format(len(pretrained_state))
        logging.info('Attention!!!')
        logging.info(msg)
        logging.info('Over!!!')
        model.load_state_dict(model_dict, strict=False)
    else:
        pretrained_dict = torch.load(cfg.MODEL.PRETRAINED, map_location='cpu')
        if 'state_dict' in pretrained_dict:
            pretrained_dict = pretrained_dict['state_dict']
        model_dict = model.state_dict()
        pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if
                           (k[6:] in model_dict and v.shape == model_dict[k[6:]].shape)}
        msg = 'Loaded {} parameters!'.format(len(pretrained_dict))
        logging.info('Attention!!!')
        logging.info(msg)
        logging.info('Over!!!')
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)

    return model


def get_pred_model(name, num_classes):
    if 'slim' in name:
        model = LCFNet(m=2, n=3, num_classes=num_classes, planes=32, ppm_planes=96, head_planes=128, augment=False)
    elif 'standard' in name:
        model = LCFNet(m=2, n=3, num_classes=num_classes, planes=32, ppm_planes=96, head_planes=128, augment=False)
    else:
        model = LCFNet(m=2, n=3, num_classes=num_classes, planes=64, ppm_planes=96, head_planes=128, augment=False)

    return model


if __name__ == '__main__':
    args = parse_args()
    # Comment batchnorms here and in model_utils before testing speed since the batchnorm could be integrated into conv operation
    # (do not comment all, just the batchnorm following its corresponding conv layer)
    device = torch.device('cuda')
    model = get_pred_model(name=args.a, num_classes=args.c)
    model.eval()
    model.to(device)
    iterations = None

    input = torch.randn(1, 3, args.r[0], args.r[1]).cuda()
    with torch.no_grad():
        for _ in range(10):
            model(input)

        if iterations is None:
            elapsed_time = 0
            iterations = 100
            while elapsed_time < 1:
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                t_start = time.time()
                for _ in range(iterations):
                    model(input)
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                elapsed_time = time.time() - t_start
                iterations *= 2
            FPS = iterations / elapsed_time
            iterations = int(FPS * 6)

        print('=========Speed Testing=========')
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        t_start = time.time()
        for _ in range(iterations):
            model(input)
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        elapsed_time = time.time() - t_start
        latency = elapsed_time / iterations * 1000
    torch.cuda.empty_cache()
    FPS = 1000 / latency
    print(FPS)
