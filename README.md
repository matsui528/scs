# SCS: Simple CLIP Search

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/matsui528/scs)

Simple CLIP Search (SCS) is a system for searching images based on a text query using the OpenCLIP model. It is designed for educational purposes and can be run in a codespace (click the above badge!).


![image](https://github.com/matsui528/scs/assets/2842345/8382fade-3e8e-4c98-b868-0457d51770ff)

## Tutorial
See our video tutorial (in Japanese):
1. [Algorithm](https://www.youtube.com/watch?v=RIMkLikJr_w)
1. [Implementation](https://www.youtube.com/watch?v=CqvosCORZhc)

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

## How to use your own images
To use your own images, delete all images in the `img` directory and replace them with your own JPEG images. Then, run `extract_features.py` and `search.py` as described above.


## Bonus: GUI by streamlit
```console
pip install streamlit
streamlit run seach_streamlit.py
```
![image](https://github.com/matsui528/scs/assets/2842345/82af0525-c028-488a-bbc0-afc8fed2644e)



## Reference
- [OpenCLIP](https://github.com/mlfoundations/open_clip): Our project is heavily based on this repository.
