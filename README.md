# SCS: Simple CLIP Search

Simple CLIP-based image search system by querying a text.

This repostiory is for educational purpose. You can run everthing in codespaces.

## Prerequisites
Prepare python 3.

```console
pip install -r requirements.txt
```


## How to run
- Download images in `img`.
  ```console
  python download_caltech256.py
  ```
- Extract features in `feature`.
  ```console
  python extract features
  ```
- Search by querying a text. You can find the search results in `out`.
  ```console
  python search.py
  ```