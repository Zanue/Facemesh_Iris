import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BlazeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(BlazeBlock, self).__init__()

        self.stride = stride
        self.channel_pad = out_channels - in_channels

        # TFLite uses slightly different padding than PyTorch
        # on the depthwise conv layer when the stride is 2.
        if stride == 2:
            self.max_pool = nn.MaxPool2d(kernel_size=stride, stride=stride)
            padding = 0
        else:
            padding = (kernel_size - 1) // 2

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                      kernel_size=kernel_size, stride=stride, padding=padding,
                      groups=in_channels, bias=True),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=1, stride=1, padding=0, bias=True),
        )

        self.act = nn.PReLU(out_channels)

    def forward(self, x):
        if self.stride == 2:
            h = F.pad(x, (0, 2, 0, 2), "constant", 0)
            x = self.max_pool(x)
        else:
            h = x

        if self.channel_pad > 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.channel_pad), "constant", 0)

        return self.act(self.convs(h) + x)


class BlazeFacemark(nn.Module):
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
        super(BlazeFacemark, self).__init__()

        # These are the settings from the MediaPipe example graph
        # mediapipe/graphs/face_detection/face_detection_mobile_gpu.pbtxt
        self.score_clipping_thresh = 100.0
        self.min_score_thresh = 0.75

        self._define_layers()

    def _define_layers(self):   
        self.backbone1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=0, bias=True),
            nn.PReLU(16),
            
            BlazeBlock(16,16),
            BlazeBlock(16,16),
            
            BlazeBlock(16,32, stride=2),
            BlazeBlock(32,32),
            BlazeBlock(32,32),
        
            BlazeBlock(32,64, stride=2),
            BlazeBlock(64,64),
            BlazeBlock(64,64),
       
            BlazeBlock(64,128, stride=2),
            BlazeBlock(128,128),
            BlazeBlock(128,128),
            
            BlazeBlock(128,128, stride=2),
            BlazeBlock(128,128),
            BlazeBlock(128,128),
        )

        self.backbone2 = nn.Sequential(
            BlazeBlock(128,128, stride=2),
            BlazeBlock(128,128),
        )
        
        self.backbone3_0_0 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1, groups=128, bias=True),
        )
        
        self.backbone2_2_0 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, groups=128, bias=True),
        )
        
        self.backbone3_0_1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0, bias=True),
        )
        
        self.backbone2_2_1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0, bias=True),
        )
        
        self.backbone3_0_2 = nn.Sequential(
            nn.PReLU(128),
        )
        
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.backbone2_2_2 = nn.Sequential(
            nn.PReLU(128),
        )
        
        self.backbone3_1_0 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=1, stride=1, padding=0, bias=True),
        )
        
        self.backbone2_3_0 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=1, stride=1, padding=0, bias=True),
        )
        
        self.backbone3_1_1 = nn.Sequential(
            nn.PReLU(32),
        )
        
        self.backbone2_3_1 = nn.Sequential(
            nn.PReLU(32),
        )
        
        self.backbone3_2_0 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, groups=32, bias=True),
        )
        
        self.backbone2_4_0 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, groups=32, bias=True),
        )
        
        self.backbone3_2_1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0, bias=True),
        )
        
        self.backbone2_4_1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0, bias=True),
        )
        
        self.backbone3_2_2 = nn.Sequential(
            nn.PReLU(32),
        )
        
        self.backbone2_4_2 = nn.Sequential(
            nn.PReLU(32),
        )
        
        self.backbone3_3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=3, padding=0, bias=True),
        )
        
        self.backbone2_5 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=1404, kernel_size=3, stride=3, padding=0, bias=True),
        )
        
        
        

    def forward(self, x):
        # TFLite uses slightly different padding on the first conv layer
        # than PyTorch, so do it manually.
        x = F.pad(x, (0, 1, 0, 1), "constant", 0)

        x = self.backbone1(x)
        
        x1 = self.backbone2(x)
        h1 = self.backbone2_2_0(x1)
        h1 = self.backbone2_2_1(h1)
        x1 = x1 + h1
        x1 = self.backbone2_2_2(x1)
        x1 = self.backbone2_3_0(x1)
        x1 = self.backbone2_3_1(x1)
        h1 = self.backbone2_4_0(x1)
        h1 = self.backbone2_4_1(h1) 
        x1 = x1 + h1
        x1 = self.backbone2_4_2(x1)
        conv2d_20 = self.backbone2_5(x1)
        
        x2 = self.backbone3_0_0(x)
        x2 = self.backbone3_0_1(x2)
        x2 = x2 + self.maxpool2(x)
        x2 = self.backbone3_0_2(x2)
        x2 = self.backbone3_1_0(x2)
        x2 = self.backbone3_1_1(x2)
        h2 = self.backbone3_2_0(x2)
        h2 = self.backbone3_2_1(h2)
        x2 = x2 + h2
        x2 = self.backbone3_2_2(x2)
        conv2d_30 = self.backbone3_3(x2)
        
        
        conv2d_20 = conv2d_20.permute(0, 2, 3, 1)
        conv2d_30 = conv2d_30.permute(0, 2, 3, 1)
        
        return [conv2d_20, conv2d_30]

    def load_weights(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()

    def _preprocess(self, x):
        """Converts the image pixels to the range [-1, 1]."""
        return x.float() / 127.5 - 1.0

    def predict_on_image(self, img):
        """Makes a prediction on a single image.

        Arguments:
            img: a NumPy array of shape (H, W, 3) or a PyTorch tensor of
                 shape (3, H, W). The image's height and width should be 
                 128 pixels.

        Returns:
            A tensor with face detections.
        """
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).permute((2, 0, 1))

        return self.predict_on_batch(img.unsqueeze(0))

    def predict_on_batch(self, x):
        """Makes a prediction on a batch of images.

        Arguments:
            x: a NumPy array of shape (b, H, W, 3) or a PyTorch tensor of
               shape (b, 3, H, W). The height and width should be 128 pixels.

        Returns:
            A list containing a tensor of face detections for each image in 
            the batch. If no faces are found for an image, returns a tensor
            of shape (0, 17).

        Each face detection is a PyTorch tensor consisting of 17 numbers:
            - ymin, xmin, ymax, xmax
            - x,y-coordinates for the 6 keypoints
            - confidence score
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).permute((0, 3, 1, 2))

        assert x.shape[1] == 3
        assert x.shape[2] == 192
        assert x.shape[3] == 192

        # 1. Preprocess the images into tensors:
        x = x.to(self._device())
        x = self._preprocess(x)

        # 2. Run the neural network:
        with torch.no_grad():
            out = self.__call__(x)
        
        thresh = self.score_clipping_thresh
        out[1] = out[1].clamp(-thresh, thresh)
        return out
    
    def _device(self):
        """Which device (CPU or GPU) is being used by this model?"""
        return "cpu"