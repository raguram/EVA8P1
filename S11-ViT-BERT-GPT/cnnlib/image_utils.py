from matplotlib import pyplot as plt
import numpy as np
from statistics import mean
from PIL import Image
from os.path import join
from torchvision import transforms

from PIL import Image

def show_images(images, titles=None, cols=10, figSize=(15, 15)):
    """
    Shows images with its labels. Expected PIL Image.
    """
    figure = plt.figure(figsize=figSize)
    num_of_images = len(images)
    rows = np.ceil(num_of_images / float(cols))
    for index in range(0, num_of_images):
        plt.subplot(rows, cols, index + 1)
        plt.axis('off')
        if titles is not None:
            plt.title(titles[index])
        plt.imshow(np.asarray(images[index]), cmap="gray")

def load_image_to_tensor(file): 
    img = Image.open(file).convert('RGB')
    img = img.resize((32, 32), Image.Resampling.BILINEAR)
    transformation = transforms.ToTensor()
    return transformation(img)