import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Blazeblock1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Blazeblock1, self).__init__()
        
        self.half = in_channels // 2
        
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=self.half, kernel_size=1, stride=1, padding=0, bias=True),
            nn.PReLU(num_parameters=self.half),
            nn.Conv2d(in_channels=self.half, out_channels=self.half, kernel_size=3, stride=1, padding=1, groups=self.half, bias=True),
            nn.Conv2d(in_channels=self.half, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=True),
        )
        
        self.act = nn.PReLU(num_parameters=out_channels)
        
    def forward(self, x):
        return self.act(x + self.convs(x))

    
    
class Blazeblock2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Blazeblock2, self).__init__()
        
        self.channel_pad = out_channels - in_channels
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=2, stride=2, padding=0, bias=True),
            nn.PReLU(num_parameters=in_channels),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, bias=True),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=True),
        )
        
        self.act = nn.PReLU(num_parameters=out_channels)
        
    def forward(self, x):
        h = x
        x = self.max_pool(x)
        x = F.pad(x, (0, 0, 0, 0, 0, self.channel_pad), "constant", 0)
        return self.act(x + self.convs(h)) 

       

class Blazeblock3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Blazeblock3, self).__init__()
        
        self.half = in_channels // 2
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=self.half, kernel_size=2, stride=2, padding=0, bias=True),
            nn.PReLU(num_parameters=self.half),
            nn.Conv2d(in_channels=self.half, out_channels=self.half, kernel_size=3, stride=1, padding=1, groups=self.half, bias=True),
            nn.Conv2d(in_channels=self.half, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=True),
        )
        
        self.act = nn.PReLU(num_parameters=out_channels)
        
    def forward(self, x):
        h = x
        x = self.max_pool(x)       
        return self.act(x + self.convs(h))
        
        

class Blazeiris(nn.Module):
    """The BlazeFace face detection model from MediaPipe.

    The version from MediaPipe is simpler than the one in the paper;
    it does not use the "double" BlazeBlocks.

    Because we won't be training this model, it doesn't need to have
    batchnorm layers. These have already been "folded" into the conv
    weights by TFLite.

    The conversion to PyTorch is fairly straightforward, but there are
    some small differences between TFLite and PyTorch in how they handle
    padding on conv layers with stride 2.

    This version works on batches, while the MediaPipe version can only
    handle a single image at a time.

    Based on code from https://github.com/tkat0/PyTorch_BlazeFace/ and
    https://github.com/google/mediapipe/
    """
    def __init__(self):
        super(Blazeiris, self).__init__()

        # These are the settings from the MediaPipe example graph
        # mediapipe/graphs/face_detection/face_detection_mobile_gpu.pbtxt
        self.score_clipping_thresh = 100.0
        self.min_score_thresh = 0.75

        self._define_layers()

    def _define_layers(self):
        self.backbone1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=0, bias=True),
            nn.PReLU(num_parameters=64),
                        
            Blazeblock1(in_channels=64, out_channels=64),
            Blazeblock1(in_channels=64, out_channels=64),
            Blazeblock1(in_channels=64, out_channels=64),
            Blazeblock1(in_channels=64, out_channels=64),
            
            Blazeblock2(in_channels=64, out_channels=128),
            
            Blazeblock1(in_channels=128, out_channels=128),
            Blazeblock1(in_channels=128, out_channels=128),
            Blazeblock1(in_channels=128, out_channels=128),
            Blazeblock1(in_channels=128, out_channels=128),
            
            Blazeblock3(in_channels=128, out_channels=128),
        )
        
        self.backbone2 = nn.Sequential(
            Blazeblock1(in_channels=128, out_channels=128),
            Blazeblock1(in_channels=128, out_channels=128),
            
            Blazeblock3(in_channels=128, out_channels=128),
            
            Blazeblock1(in_channels=128, out_channels=128),
            Blazeblock1(in_channels=128, out_channels=128),
            
            Blazeblock3(in_channels=128, out_channels=128),
            
            Blazeblock1(in_channels=128, out_channels=128),
            Blazeblock1(in_channels=128, out_channels=128),
            
            nn.Conv2d(in_channels=128, out_channels=213, kernel_size=2, stride=1, padding=0, bias=True),
        )
        
        self.backbone3 = nn.Sequential(
            Blazeblock1(in_channels=128, out_channels=128),
            Blazeblock1(in_channels=128, out_channels=128),
            
            Blazeblock3(in_channels=128, out_channels=128),
            
            Blazeblock1(in_channels=128, out_channels=128),
            Blazeblock1(in_channels=128, out_channels=128),
            
            Blazeblock3(in_channels=128, out_channels=128),
            
            Blazeblock1(in_channels=128, out_channels=128),
            Blazeblock1(in_channels=128, out_channels=128),
            
            nn.Conv2d(in_channels=128, out_channels=15, kernel_size=2, stride=1, padding=0, bias=True),
        )
        

    def forward(self, x):
        # TFLite uses slightly different padding on the first conv layer
        # than PyTorch, so do it manually.
        x = F.pad(x, (0, 1, 0, 1), "constant", 0)
        
        b = x.shape[0]      # batch size, needed for reshaping later

        x = self.backbone1(x)
        output_eyes_contours_and_brows = self.backbone2(x).permute(0, 2, 3, 1).reshape(1,-1)
        output_iris = self.backbone3(x).permute(0, 2, 3, 1).reshape(1,-1)
        
        return [output_eyes_contours_and_brows, output_iris]

    def _device(self):
        """Which device (CPU or GPU) is being used by this model?"""
        return "cpu"

    def load_weights(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()

    def _preprocess(self, x):
        """Converts the image pixels to the range [-1, 1]."""
        return x.float() / 127.5 - 1.0
