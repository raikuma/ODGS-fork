


<p align="center">
  <h1 align="center">ODGS: 3D Scene Reconstruction from Omnidirectional Images <br> with 3D Gaussian Splatting</h1>
  <p align="center">
    <a href="https://esw0116.github.io/">Suyoung Lee</a>*
    &nbsp;·&nbsp;
    <a href="https://robot0321.github.io/">Jaeyoung Chung</a>*
    &nbsp;·&nbsp;
    Jaeyoo Huh
    &nbsp;·&nbsp;
    <a href="https://cv.snu.ac.kr/index.php/~kmlee/">Kyoung Mu Lee</a>
    </br>
    (* denotes equal contribution)
  </p>
  <h3 align="center">NeurIPS 2024</h3>
</p>


<!-- <div align="center">

[![ArXiv]()]()
[![Github](https://img.shields.io/github/stars/luciddreamer-cvlab/LucidDreamer)](https://github.com/luciddreamer-cvlab/LucidDreamer)
[![LICENSE](https://img.shields.io/badge/license-MIT-lightgrey)](https://github.com/luciddreamer-cvlab/LucidDreamer/blob/master/LICENSE)

</div> -->




<p align="center">
    <img src="assets/logo_cvlab.png" height=60>
</p>

---
This is an official implementation of "ODGS: 3D Scene Reconstruction from Omnidirectional Images with 3D Gaussian Splatting."


### Update Log
**24.11.08:**  First Upload (CUDA Rasterizer and training code)


## Installation
~~~bash
git clone https://github.com/esw0116/ODGS.git --recursive
cd ODGS

# Set Environment
conda env create --file environment.yml
conda activate ODGS
pip install submodules/simple-knn
pip install submodules/odgs-gaussian-rasterization
~~~

### Training (Optimization)
ODGS requires optimization for each scene. Run the script below to start optimization:
~~~python
python train.py -s <source(dataset)_path> -m <output_path> --eval
~~~

