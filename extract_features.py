import torch
from PIL import Image
import open_clip
from pathlib import Path
import numpy as np

if __name__ == '__main__':
    # Prepare a CLIP model
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')

    # Read images
    imgs = []
    for img_path in sorted(Path("./img").glob("*.jpg")):
        img = Image.open(img_path).convert("RGB")
        imgs.append(preprocess(img))
    imgs = torch.tensor(np.stack(imgs))

    # Extract image features and normalize them
    with torch.no_grad(), torch.cuda.amp.autocast():
        img_features = model.encode_image(imgs)  # shape=(257, 512)
        img_features /= img_features.norm(dim=-1, keepdim=True)

    # Save the features
    torch.save(img_features, 'feature/features.pt')
    