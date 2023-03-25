import torch.nn as nn 
import torch.nn.functional as F 
from torchsummary import summary 
import torch

class Residual(nn.Module): 

    def __init__(self, fn): 
        super().__init__()
        self.fn = fn 

    def forward(self, x):
        return self.fn(x) + x 

# https://arxiv.org/abs/2201.09792
class ConvMixer(nn.Module): 

    def __init__(self, dim, depth, kernel_size=5, patch_size=2, n_classes=10):  ## CIFAR10 
        
        super().__init__()

        self.layer = nn.Sequential(

            ## Patch embedding layer 
            nn.Conv2d(in_channels=3, out_channels=dim, kernel_size=patch_size, stride=patch_size),
            nn.GELU(),
            nn.BatchNorm2d(num_features=dim),

            *[nn.Sequential(
                
                # Residual 
                Residual(
                    nn.Sequential(
                        nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                        nn.GELU(),
                        nn.BatchNorm2d(dim)
                    )
                ),

                ## Pointwise 
                nn.Sequential(
                    nn.Conv2d(dim, dim, 1), 
                    nn.GELU(), 
                    nn.BatchNorm2d(dim)
                )
            ) for r in range(depth)], 

            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(), 
            nn.Linear(dim, n_classes)
        )   
    def forward(self, x): 
        return self.layer(x)

    def summarize(self, input): 
        summary(self, input_size=input)

def main(): 
    depth = 10
    hdim = 256
    psize = 2
    conv_ks = 5

    net = ConvMixer(hdim, depth, conv_ks, psize).to(getDevice())
    net.summarize((3, 32, 32))

def getDevice(): 
  return torch.device("cuda" if isCuda() else "cpu")

def isCuda():
  return torch.cuda.is_available()

if __name__ == "__main__": 
    main()