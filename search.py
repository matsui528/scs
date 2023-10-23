import torch
import open_clip
import numpy as np
from pathlib import Path
from PIL import Image

if __name__ == '__main__':
    # Prepare a CLIP model
    model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    tokenizer = open_clip.get_tokenizer('ViT-B-32')

    # Read image names and image features
    img_paths = [img_path for img_path in sorted(Path("./img").glob("*.jpg"))]
    img_features = torch.load("feature/features.pt")

    while 1:
        # Get the query text
        query = input("Query> ")
        print(f"Query is '{query}'")

        # Extract a text feature and normalize it
        text = tokenizer([query])
        with torch.no_grad(), torch.cuda.amp.autocast():
            text_feature = model.encode_text(text)  # shape=(1, 512)
            text_feature /= text_feature.norm(dim=-1, keepdim=True)

        # Find the top 5 nearest images
        dists = np.linalg.norm(text_feature - img_features, axis=1)
        ids = np.argsort(dists)[:5]

        # Save them in "out"
        for n, id in enumerate(ids):
            Image.open(img_paths[id]).save(f"out/{n}.jpg")
            print(f"{img_paths[id]} -> out/{n}.jpg")
    


