# Prototypical Networks, Magnet Loss and RepMet in PyTorch
**NOTE 1: THIS PROJECT IS BEING REWORKED, CURRENTLY ONLY PROTOTYPICAL NETS WORKS**

### Prototypical Networks (Few-Shot Classification)
![Figure 1 from paper](proto.png)

"[Prototypical Networks for Few-shot Learning](https://arxiv.org/pdf/1703.05175.pdf)"
learn a metric space in which classification can be performed by computing
distances to prototype representations of each class. They use this technique
to perform episode based few-shot learning.


My implementation takes a lot from [orobix/Prototypical-Networks-for-Few-shot-Learning-PyTorch](https://github.com/orobix/Prototypical-Networks-for-Few-shot-Learning-PyTorch)


#### Training
Model embeds image into vector, with batches consisting of a subset of classes,
 with a set of support and set of query examples for each class. The supports
 are used to build the prototype vector (just avg them) and the querys are then
 used to calculate the loss by comparing how close they are to their prototypes.

**The Goal:** Train a model that embeds samples of similar class close, while others far.

#### Testing
Testing is done in the same way, now we can think of the number of supports
per class as the n-shot.

### Magnet Loss (Fine-Grained Classification)
![Figure 3 from paper](magnet.png)

"[Metric Learning with Adaptive Density Discrimination](http://arxiv.org/pdf/1511.05939v2.pdf)"
learn a metric space in which classification can be performed by computing
distances to cluster centres, with clusters belonging to classes. They don't
perform few shot classification and instead focus on fine-grained classification.

This takes a lot from the Tensorflow Magnet Loss code: [pumpikano/tf-magnet-loss](https://github.com/pumpikano/tf-magnet-loss)

#### Training
Model embeds images into vectors which are used to make **k** clusters per class (with kmeans).
We forward pass the entire training set to embed all samples to perform kmeans
and build (and update) the clusters. The batches consist of **m** clusters,
a semi-random seed cluster is selected (chosen based on loss value of its members)
 then the next closest m-1 clusters of
different classes are chosen. From the **m** clusters, **d** samples which
belong to each of these clusters are randomly chosen (samples belong if they
are closest to the cluster center than any other cluster centre).

**The Goal:** Train a model which embeds samples of similar class close, while others far.
Also, find cluster means and variances for each class.

#### Testing
Using the clusters (means and variances) learnt in training (obtained by
performing kmeans over the training set) embed the test set and classify.

### RepMet (Fine-Grained Classification + Few-Shot Detection)
![Figure 2 from paper](repmet.png)

"[RepMet: Representative-based Metric Learning for Classification and One-shot Object Detection](https://arxiv.org/pdf/1806.04728.pdf)"
extends upon magnet loss by storing the centroid as representations that are learnable, rather than just
 statically calculated every now and again with k-means. They also perform fine-grained classification,
 but also extend their work to perform few-shot object detection.

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


## Datasets
### [Omniglot](https://github.com/brendenlake/omniglot)
This dataset contains 1623 different handwritten characters from 50 different alphabets.
Each of the 1623 characters was drawn online via Amazon's Mechanical Turk by 20 different people.
Images are greyscale and square 105 x 105 px.


**Train:** 82240 samples spanning 4112 classes (avg 20 per class)

**Val:** 13760 samples spanning 688 classes (avg 20 per class)

**Test:** 33840 samples spanning 1692 classes (avg 20 per class)

Note: Classes are mutually exclusive in the splits, for the few shot scenario.

### [Oxford Flowers](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/)
This dataset contains images of flowers, covering 102 classes with each class consisting of between 40 and 258 images.
Images are RGB with shortest edge being 500px.


**Train:** 1020 samples spanning 102 classes (avg 10 per class)

**Val:** 1020 samples spanning 102 classes (avg 10 per class)

**Test:** 6149 samples spanning 102 classes (avg 60 per class)

### [Oxford Pets](http://www.robots.ox.ac.uk/~vgg/data/pets/)
This dataset contains images of pet animals (cats and dogs), covering 37 classes with each class consisting of around 200 images.
Images are RGB with different scales and ratios.


**TrainVal:** 3680 samples spanning 37 classes (avg 99 per class)

**Test:** 3669 samples spanning 37 classes (avg 99 per class)

### Coming Soon
StanfordDogs, PascalVoc, ImageNet


## Results

These results are calculated with `classification/evaluate.py`

### Prototypical Networks

**Omniglot (Test Set):**

| Model            |  1-Shot, 5-way  |  5-Shot, 5-way  |  1-Shot, 20-way  |  5-Shot, 20-way  |
| ---------------- | --------------: | --------------: | ---------------: | ---------------: |
| `.yaml` file     | `protonets_1_5` | `protonets_5_5` | `protonets_1_20` | `protonets_5_20` |
| *Original Paper* |           98.8% |           99.7% |            96.0% |            98.9% |
| *This Code* *    |          98.85% |          99.69% |           94.74% |           98.49% |

*One test run of 100 episodes

Model checkpoints will be available soon

### Magnet Loss

**COMING SOON**