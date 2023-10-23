import torchvision
import pathlib


# Download the Caltech256 dataset. The images (30K) will be in ./img, i.e., 
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
#      ...

_ = torchvision.datasets.Caltech256(root='./img', download=True)


# Then, rename them, i.e.,
# ./img/caltech256/001.ak47/001_0001.jpg -> ./img/001_0001.jpg

root_dir = pathlib.Path('./img/caltech256')
for file_path in root_dir.glob('**/*.jpg'):
    new_path = pathlib.Path('./img') / file_path.name
    file_path.rename(new_path)