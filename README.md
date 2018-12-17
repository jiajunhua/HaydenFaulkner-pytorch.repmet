# Prototypical Networks, Magnet Loss and RepMet in PyTorch
**NOTE 1: THIS PROJECT IS BEING REWORKED, CURRENTLY ONLY PROTOTYPICAL NETS WORKS**

### Prototypical Networks
![Figure 1 from paper](proto.png)

"[Prototypical Networks for Few-shot Learning](https://arxiv.org/pdf/1703.05175.pdf)" learn a metric space in which classification can be performed by computing distances to prototype representations of each class.

My implementation takes a lot from [orobix/Prototypical-Networks-for-Few-shot-Learning-PyTorch](https://github.com/orobix/Prototypical-Networks-for-Few-shot-Learning-PyTorch)


### Magnet Loss
![Figure 3 from paper](magnet.png)

"[Metric Learning with Adaptive Density Discrimination](http://arxiv.org/pdf/1511.05939v2.pdf)" introduced
a new optimization objective for distance metric learning called Magnet Loss that, unlike related losses,
operates on entire neighborhoods in the representation space and adaptively defines the similarity that is
being optimized to account for the changing representations of the training data.

This takes a lot from the Tensorflow Magnet Loss code: [pumpikano/tf-magnet-loss](https://github.com/pumpikano/tf-magnet-loss)

### RepMet
![Figure 2 from paper](repmet.png)

"[RepMet: Representative-based Metric Learning for Classification and One-shot Object Detection](https://arxiv.org/pdf/1806.04728.pdf)"
extends upon magnet loss by storing the centroid as representations that are learnable, rather than just
 statically calculated every now and again with k-means.

## Install

Tested with **python 3.6 + pytorch 1.0 + cuda 9.1**

We suggest using a virtual environment, and cuda

Also requires tensorboard, tensorflow, and [tensorboardX](https://github.com/lanpa/tensorboardX)
```
pip install tensorboard  # will install tensorflow CPU version as dependency first
pip install tensorboardX
```

## Implementation

**NOTE: Currently only Prototypical Networks working**


See `classification/train.py` for training the model, and the `classification/experiments` directory for the config `.yaml` files.

### RepMet.v2
There are two versions of the RepMet loss implemented in this code, as the original authors suggested this modification from the original paper.

**Version 1:**
As in the original paper it uses the closest (R*) representative in the numerator and disregards same class representatives in the denominator:

![eq repmetv1](https://latex.codecogs.com/gif.latex?L%28%5CTheta%20%29%20%3D%20%5Cfrac%7B1%7D%7BN%7D%5Csum_%7Bn%3D1%7D%5E%7BN%7D%5Cleft%20%7C%20-%5Ctextup%7Blog%7D%20%5Cleft%20%28%20%5Cfrac%7Be%5E%7B-%5Cfrac%7B1%7D%7B2%5Csigma%5E2%7D%5Cleft%20%5C%7C%20E_n%20-%20R%5E*_n%20%5Cright%20%5C%7C%5E2_2%7D%7D%7B%5Cunderset%7Bi%3AC%28R_i%29%5Cneq%20C%28E_n%29%7D%5Csum%20e%5E%7B-%5Cfrac%7B1%7D%7B2%5Csigma%5E2%7D%5Cleft%20%5C%7C%20E_n%20-%20R_i%20%5Cright%20%5C%7C%5E2_2%7D%7D%20%5Cright%20%29%20&plus;%20%5Calpha%20%5Cright%20%7C_&plus;)

where:

![eq repmetv1b](https://latex.codecogs.com/gif.latex?R%5E*_n%20%3D%20%5Ctextup%7Barg%7D%20%5Cunderset%7Bi%3AC%28R_i%29%3DC%28E_n%29%7D%7B%5Ctextup%7Bmin%7D%7D%5Cleft%20%5C%7C%20E_n%20-%20R_i%20%5Cright%20%5C%7C)


**Version 2:**
Sums the distance of all representatives of the same class in the numerator, and doesn't disregard any in the denominator.

![eq repmetv2](https://latex.codecogs.com/gif.latex?L%28%5CTheta%20%29%20%3D%20%5Cfrac%7B1%7D%7BN%7D%5Csum_%7Bn%3D1%7D%5E%7BN%7D%5Cleft%20%7C%20-%5Ctextup%7Blog%7D%20%5Cleft%20%28%20%5Cfrac%7B%5Cunderset%7Bi%3AC%28R_i%29%3D%20C%28E_n%29%7D%5Csum%20e%5E%7B-%5Cfrac%7B1%7D%7B2%5Csigma%5E2%7D%5Cleft%20%5C%7C%20E_n%20-%20R_i%20%5Cright%20%5C%7C%5E2_2%7D%7D%7B%5Cunderset%7Bi%7D%5Csum%20e%5E%7B-%5Cfrac%7B1%7D%7B2%5Csigma%5E2%7D%5Cleft%20%5C%7C%20E_n%20-%20R_i%20%5Cright%20%5C%7C%5E2_2%7D%7D%20%5Cright%20%29%20&plus;%20%5Calpha%20%5Cright%20%7C_&plus;)


## Datasets
Currently Omniglot Supported

Soon OxfordFlowers, OxfordPet, StanfordDogs, PascalVoc, ImageNet


## Results

These results are calculated with `classification/evaluate.py`

### Prototypical Networks

**Omniglot (Test Set):**

| Model            |  1-Shot, 5-way  |  5-Shot, 5-way  |  1-Shot, 20-way  |  5-Shot, 20-way  |
| ---------------- | --------------: | --------------: | ---------------: | ---------------: |
| `.yaml` file     | `protonets_1_5` | `protonets_5_5` | `protonets_1_20` | `protonets_5_20` |
| *Original Paper* |           98.8% |           99.7% |            96.0% |            98.9% |
| *This Code*      |          98.47% |          99.59% |           94.58% |           98.37% |


### Magnet Loss

**COMING SOON**