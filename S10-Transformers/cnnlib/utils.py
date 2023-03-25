import torch

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2471, 0.2435, 0.2616)

def isCuda():
  return torch.cuda.is_available()

def getDevice(): 
  return torch.device("cuda" if isCuda() else "cpu")

def randInt(min, max, size):
    return torch.LongTensor(size).random_(min, max)

def pickRandomElements(data, count):
    randIndex = randInt(0, len(data), count)
    if (count >= len(data)):
        randIndex = [i for i in range(0, len(data))]

    return randIndex

def unnormalize(images, mean, sig):
    """
    Unnormalize the tensor
    """
    copy = images.clone().detach()

    for img in copy:
        for t, m, s in zip(img, mean, sig):
            t.mul_(s).add_(m)
    return copy
