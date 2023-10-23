import torch
import open_clip
import numpy as np
from pathlib import Path
from PIL import Image

if __name__ == '__main__':
    model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    tokenizer = open_clip.get_tokenizer('ViT-B-32')

    img_features = torch.load("feature/features.pt")
    img_paths = [img_path for img_path in sorted(Path("./img").glob("*.jpg"))]

    while 1:
        query = input("Query> ")
        print(f"Query is '{query}'")

        text = tokenizer([query])

        with torch.no_grad(), torch.cuda.amp.autocast():
            text_feature = model.encode_text(text)
            text_feature /= text_feature.norm(dim=-1, keepdim=True)

        dists = np.linalg.norm(text_feature - img_features, axis=1)
        ids = np.argsort(dists)[:5]
        for n, id in enumerate(ids):
            Image.open(img_paths[id]).save(f"out/{n}.jpg")
            print(img_paths[id])
    


