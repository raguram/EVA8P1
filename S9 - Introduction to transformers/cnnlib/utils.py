import torch

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
