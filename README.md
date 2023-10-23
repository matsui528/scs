# SCS: Simple CLIP Search

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/matsui528/scs)

Simple CLIP Search (SCS) is a system for searching images based on a text query using the OpenCLIP model. It is designed for educational purposes and can be run in a codespace (push the above badge!).

## Prerequisites
Before running SCS, you will need to have Python 3 installed on your system. You will also need to install the following dependencies:

```console
pip install -r requirements.txt
```

## How to run
To run SCS, follow these steps:
- Download images (Caltech256 dataset) to the `img` directory:
  ```console
  python download_caltech256.py
  ```
- Extract features to the `feature` directory:
  ```console
  python extract_features.py
  ```
- Search by querying a text. You can find the search results in the `out` directory:
  ```console
  python search.py
  ```


## Reference
- [OpenCLIP](https://github.com/mlfoundations/open_clip): Our project is heavily based on this repository.
