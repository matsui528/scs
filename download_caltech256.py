import torchvision
import pathlib
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

# Download multiple datasets. The images will be in their respective directories under ./img, i.e., 
# ./img
#    caltech256
#      001.ak47
#        001_0001.jpg
#        001_0002.jpg
#        001_0003.jpg
#        ...
#      002.american-flag
#        002_0001.jpg
#        ...
#    mnist
#      MNIST
#        processed
#        raw
#        ...

# Download the Caltech256 dataset
_ = torchvision.datasets.Caltech256(root='./img/caltech256', download=True)

# Download the MNIST dataset
_ = MNIST(root='./img/mnist', train=True, download=True, transform=ToTensor())
_ = MNIST(root='./img/mnist', train=False, download=True, transform=ToTensor())

# For Caltech256, for each class, pick up the first image (00X_0001.jpg) and rename it as follows:
# e.g., ./img/caltech256/001.ak47/001_0001.jpg -> ./img/caltech256/001_0001.jpg

caltech_img_path = pathlib.Path('./img/caltech256/')
for file_path in caltech_img_path.glob('**/*0001.jpg'):
    new_path = caltech_img_path / file_path.name
    print(f"{file_path} -> {new_path}")
    file_path.rename(new_path)

# Note: MNIST images are stored in a processed format and do not need to be renamed.
