import torch
from PIL import Image
import open_clip
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    print("Read the model")
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')

    img_paths = [img_path for img_path in sorted(Path("./img").glob("*.jpg"))]
    img_paths = img_paths[:100]

    print("Read images")
    imgs = []
    for img_path in img_paths:
        img = Image.open(img_path).convert("RGB")
        imgs.append(preprocess(img))
    imgs = torch.tensor(np.stack(imgs))

    print("Extract features")
    with torch.no_grad(), torch.cuda.amp.autocast():
        features = model.encode_image(imgs)
        features /= features.norm(dim=-1, keepdim=True)
        print(features.shape)  # 1, 512

