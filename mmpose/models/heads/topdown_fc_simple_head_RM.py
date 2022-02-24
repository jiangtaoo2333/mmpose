import logging as log
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import (build_conv_layer, build_norm_layer, build_upsample_layer,
                      constant_init, normal_init)
from torch.nn.modules.batchnorm import _BatchNorm

from mmpose.core.evaluation import pose_pck_accuracy
from mmpose.core.post_processing import flip_back
from mmpose.models.builder import build_loss
from mmpose.models.utils.ops import resize
from ..builder import HEADS


class GlobalAvgPool2d(nn.Module):
    """ This layer averages each channel to a single number.
    """
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        B = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        x = F.avg_pool2d(x, (H, W))
        #print("gapSize: ", B, C, H, W, x.size())
        x = x.view(B, C)
        #print("gap2d", x.size())
        return x

class Conv2dBatchReLU(nn.Module):
    """ This convenience layer groups a 2D convolution, a batchnorm and a ReLU.
    They are executed in a sequential manner.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int or tuple): Size of the kernel of the convolution
        stride (int or tuple): Stride of the convolution
        padding (int or tuple): padding of the convolution
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, isPadding=True, isBias=False):
        super(Conv2dBatchReLU, self).__init__()

        # Parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.isBias = isBias
        if isinstance(kernel_size, (list, tuple)):
            self.padding = [int(ii/2) for ii in kernel_size]
        else:
            self.padding = int(kernel_size/2)

        # Layer
        if isPadding == True:

            self.layers = nn.Sequential(
                nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, bias=self.isBias),
                nn.BatchNorm2d(self.out_channels),
                nn.ReLU(inplace=True)
            )
        else:

            self.layers = nn.Sequential(
                nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, 0, bias=self.isBias),
                nn.BatchNorm2d(self.out_channels),
                nn.ReLU(inplace=True)
            )


    def __repr__(self):
        s = '{name} ({in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}, padding={padding})'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def forward(self,x):

        x = self.layers(x)

        return x

class FullyConnectLayer(nn.Module):
    """ This layer averages each channel to a single number.
    """
    def __init__(self, in_channels, out_channels):
        super(FullyConnectLayer, self).__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.layers = nn.Sequential(
            nn.Linear(self.in_channels, self.out_channels),
        )

    def forward(self, x):
        if len(x.size()) < 2:
            print("FullyConnectLayer input error!\n")
            sys.exit()
        flattenNum = 1
        for i in range(1,len(x.size())): 
            flattenNum *= x.size(i)  

        x = x.view(-1, flattenNum)
        x = self.layers(x)  
        return x

@HEADS.register_module()
class RMFcSimpleHead(nn.Module):

    def __init__(self,
                in_channels,
                out_channels,
                loss_gaze,
                frozen_parameters=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.loss = build_loss(loss_gaze)
        self.frozen_parameters = frozen_parameters

        layer0 = Conv2dBatchReLU(self.in_channels, 128, 3, 1)   #16
        layer1 = Conv2dBatchReLU(128, 256, 3, 2)   #8
        layer2 = Conv2dBatchReLU(256, 512, 3, 2)   #4
        layer3 = GlobalAvgPool2d()                 #1
        
        layer4 = FullyConnectLayer(512, 10)
        layer5 = FullyConnectLayer(10, self.out_channels)

        self.features = nn.Sequential(
            layer0,
            layer1,
            layer2,
            layer3,
        )
        
        self.classifier = nn.Sequential(
            layer4,
            layer5,
        )

    def forward(self, x):

        x = self.features(x)
        x = self.classifier(x)

        return x 

    def init_weights(self):
        """Initialize model weights."""
        for _, m in self.features.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                normal_init(m, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
        for m in self.classifier.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)

    def get_loss(self, output, target):
        """Calculate top-down keypoint loss.

        Args:
            output (torch.Tensor[Nx2]): Output heatmaps.
            target (torch.Tensor[Nx2]): Target heatmaps.
            target_weight (torch.Tensor[Nx1]):
                Weights across different joint types.
        """

        losses = dict()

        losses['gaze_loss'] = self.loss(output, target)

        return losses

    def inference_model(self,x):

        x = self.features(x)
        x = self.classifier(x)
        x = x.detach().cpu().numpy()
        return x 

    def _frozen_parameters(self):

        for m in self.modules():
            if isinstance(m, _BatchNorm):
                m.eval()
        for param in self.parameters():
            param.requires_grad = False

    def train(self,mode=True):
        super().train(mode)
        if self.frozen_parameters:
            self._frozen_parameters()
