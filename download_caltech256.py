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


# For each class, pick up the fist image (00X_0001.jpg) and rename it as follows:
# e.g., ./img/caltech256/001.ak47/001_0001.jpg -> ./img/001_0001.jpg

img_path = pathlib.Path('./img/')
for file_path in (img_path / "caltech256").glob('**/*0001.jpg'):
    new_path = img_path / file_path.name
    print(f"{file_path} -> {new_path}")
    file_path.rename(new_path)



# Finally, there are 257 jpg images right under the img directory (256 classes + 1 clutter class)