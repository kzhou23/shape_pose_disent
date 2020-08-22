# Unsupervised Shape and Pose Disentanglement for 3D Meshes
Repo for **"Unsupervised Shape and Pose Disentanglement for 3D Meshes, ECCV'20 (Poster)"**

Link to paper: https://arxiv.org/abs/2007.11341
Link to project: https://virtualhumans.mpi-inf.mpg.de/unsup_shape_pose/

## Prerequisites
1. Cuda 9.0
2. Python 2.7
3. Pytorch 1.3
4. Scikit-sparse
5. MPI mesh library (https://github.com/MPI-IS/mesh)
6. OpenDR (https://github.com/mattloper/opendr)

## Data Preprocessing
1. Download and uncompress AMASS Dataset (https://amass.is.tue.mpg.de/)
2. Download SMPL+H body models (https://mano.is.tue.mpg.de/)
3. Preprocess AMASS to generate training/validation/test sets: `python data/data_extraction.py`

## Model Training
1. Edit `config.json` to use your own directory structures and model hyperparameters
2. Run `python train.py`

## Pretrained Models
Coming soon.

Please consider citing our work if you found it useful:
```
@inproceedings{zhou20unsupervised,
    title = {Unsupervised Shape and Pose Disentanglement for 3D Meshes},
    author = {Zhou, Keyang and Bhatnagar, Bharat Lal and Pons-Moll, Gerard},
    booktitle = {European Conference on Computer Vision (ECCV)},
    month = {August},
    year = {2020},
}
```
