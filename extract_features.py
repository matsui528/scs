import torch
from PIL import Image
import open_clip
from pathlib import Path
import numpy as np

if __name__ == '__main__':
    # Prepare a CLIP model
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')

    # Initialize a dictionary to hold features for different datasets
    dataset_features = {}

    # Process Caltech256 images
    caltech_imgs = []
    for img_path in sorted(Path("./img/caltech256").glob("**/*.jpg")):
        img = Image.open(img_path).convert("RGB")
        caltech_imgs.append(preprocess(img))
    if caltech_imgs:
        caltech_imgs = torch.tensor(np.stack(caltech_imgs))
        with torch.no_grad(), torch.cuda.amp.autocast():
            caltech_features = model.encode_image(caltech_imgs)
            caltech_features /= caltech_features.norm(dim=-1, keepdim=True)
        dataset_features['caltech256'] = caltech_features

    # Process MNIST images
    mnist_imgs = []
    for img_path in sorted(Path("./img/mnist").glob("**/*.png")):
        img = Image.open(img_path).convert("RGB")
        mnist_imgs.append(preprocess(img))
    if mnist_imgs:
        mnist_imgs = torch.tensor(np.stack(mnist_imgs))
        with torch.no_grad(), torch.cuda.amp.autocast():
            mnist_features = model.encode_image(mnist_imgs)
            mnist_features /= mnist_features.norm(dim=-1, keepdim=True)
        dataset_features['mnist'] = mnist_features

    # Save the features for each dataset
    for dataset_name, features in dataset_features.items():
        torch.save(features, f'feature/{dataset_name}_features.pt')

    # Updated to support feature extraction from multiple datasets
