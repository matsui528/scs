import streamlit as st
import torch
import open_clip
import numpy as np
from pathlib import Path
from PIL import Image


# Prepare a CLIP model
model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
tokenizer = open_clip.get_tokenizer('ViT-B-32')

# Read image names and image features
img_paths = [img_path for img_path in sorted(Path("./img").glob("*.jpg"))]
img_features = torch.load("feature/features.pt")


# Get the query text
query = st.text_input("Query", "plant")
st.write(f"Query is {query}")


# Extract a text feature and normalize it
text = tokenizer([query])
with torch.no_grad(), torch.cuda.amp.autocast():
    text_feature = model.encode_text(text)  # shape=(1, 512)
    text_feature /= text_feature.norm(dim=-1, keepdim=True)

# Find the top 9 nearest images
dists = np.linalg.norm(text_feature - img_features, axis=1)
ids = np.argsort(dists)[:9]

# Show images
for c in range(3):
    cols = st.columns(3)
    for n, col in enumerate(cols):
        with col:
            st.image(str(img_paths[ids[n + 3 * c]]))




