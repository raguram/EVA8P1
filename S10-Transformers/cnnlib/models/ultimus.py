import torch.nn as nn 
import torch.nn.functional as F 
from torchsummary import summary 
import torch
import sys

class Embedding(nn.Module):

    def __init__(self, out_channels):

        super(Embedding, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=out_channels[0], kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=out_channels[0])
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels[0], out_channels=out_channels[1], kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=out_channels[1])
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels[1], out_channels=out_channels[2], kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=out_channels[2])
        )

        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x): 
        
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.gap(out)
        return out

class Ultimus(nn.Module): 

    def __init__(self, io_channels, intermediate_channels):
        super(Ultimus, self).__init__()
        
        self.io_channels = io_channels
        self.intermediate_channels = intermediate_channels

        self.w_q = nn.Linear(io_channels, intermediate_channels)
        self.w_k = nn.Linear(io_channels, intermediate_channels)
        self.w_v = nn.Linear(io_channels, intermediate_channels)
        self.outFc = nn.Linear(intermediate_channels, io_channels)

    def forward(self, x): 
        
        bs = x.size(0)

        k = self.w_k(x).view(bs, self.intermediate_channels, -1)
        q = self.w_q(x).view(bs, self.intermediate_channels, -1)
        v = self.w_v(x).view(bs, self.intermediate_channels, -1)

        scores = q @ k.transpose(1, 2)

        attention_weights = F.softmax(scores / self.intermediate_channels ** 0.5, dim = 2)

        out = attention_weights @ v
        
        out = self.outFc(out.view(bs, -1))
        return out

class Net(nn.Module): 

    def __init__(self): 

        super(Net, self).__init__()

        self.embedding = Embedding(out_channels=(16, 32, 48))
        self.ultimus1 = Ultimus(48, 8)
        self.ultimus2 = Ultimus(48, 8)
        self.ultimus3 = Ultimus(48, 8)
        self.ultimus4 = Ultimus(48, 8)
        self.outFc = nn.Linear(48, 10)

    def forward(self, x): 

        ## Shape of input is (batchsize)x3x32x32 
        out = self.embedding(x)

        ## Shape of out embedding is (batchsize)x48x1x1
        out = out.reshape(-1, 48)
        ## Reshaped to (batchsize)x48
        
        out = self.ultimus1(out)
        out = self.ultimus2(out)
        out = self.ultimus3(out)
        out = self.ultimus4(out)
        out = self.outFc(out)

        return out

    def summarize(self, input): 
        summary(self, input_size=input)

def main(): 
    net = Net()
    # net.summarize((3, 32, 32))

    # sys.path.insert(0, "/home/raguramk/workspace/Eva/EVA8P1/S9")

    # from cnnlib.image_utils import load_image_to_tensor

    # img = load_image_to_tensor("/home/raguramk/workspace/Eva/EVA8P1/cifar10.png")
    
    # img = img.reshape((1, 3, 32, 32))
    # pred = net(img)

    # print(f"{pred}")

if __name__ == "__main__":
    main()