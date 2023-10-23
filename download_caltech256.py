import torchvision
_ = torchvision.datasets.Caltech256(root='./img', download=True)

# The images will be downloaded in ./img, i.e., 
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

