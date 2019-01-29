# Prototypical Networks, Magnet Loss and RepMet in PyTorch
**THIS PROJECT IS STILL UNDER REVISIONS - USE WITH CAUTION**

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

1. Clone the repo
```
git clone https://github.com/HaydenFaulkner/pytorch.repmet.git
```

2. Make a `data` and a `models` directory to store the data and models respectively.
```
cd pytorch.repmet
mkdir data
mkdir models
```

### Detection
If wanting to use the detection pipeline you will need to compile the roi layers.
```
cd model_definitions/detectors/faster_rcnn  # change the working dir
setup.py build develop  # run the compiler via the setup.py file
```

The expected kind of output is shown in the `setup.py` file.


## Implementation
### Classification
See `classification/train.py` for training the model, and the `classification/experiments` directory for the config `.yaml` files.

### Detection
See `detection/train.py` for training the model, and the `detection/experiments` directory for the config `.yaml` files.


## Datasets
Datasets are automatically downloaded, organised and stored in the `data` directory when called.

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

### [Stanford Dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/)
This dataset contains images of 120 breeds of dogs with each class consisting of around 170 images.
Images are RGB with different scales and ratios.


**Train:** 12000 samples spanning 120 classes (avg 100 per class)

**No Val set**

**Test:** 8580 samples spanning 120 classes (avg 71 per class)

### [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/)
This dataset is used for detection and contains images with bounding boxes marked out around 20 general object categories.
Images are RGB with different scales and ratios.

**2007**
**Train:** 6301 samples (boxes) spanning 20 classes and 2501 images (avg 315 per class, 2 per image)

**Val:** 6307 samples (boxes) spanning 20 classes and 2510 images (avg 315 per class, 2 per image)

**Test:** 12032 samples (boxes) spanning 20 classes and 4952 images (avg 601 per class, 2 per image)

**2012**
**Train:** 13609 samples (boxes) spanning 20 classes and 5717 images (avg 680 per class, 2 per image)

**Val:** 13841 samples (boxes) spanning 20 classes and 5823 images (avg 692 per class, 2 per image)

**Uses 2007 Test Set**

### Coming Soon
ImageNet

## Training Behaviour
Time format: HH:MM:SS

GPU Used: GTX 980 Ti

GPU Memory calculation is approximate.

### Prototypical Networks

**Omniglot:**

| Model (`.yaml`)  | Epochs | Steps | Total Time | GPU Memory | Line |
| :--------------- | -----: | ----: | ---------: | ---------: | ---- |
| `protonets_1_5`  |    100 | 10000 |   00:24:12 |     1.1 GB |      |
| `protonets_5_5`  |    100 | 10000 |   00:32:33 |     1.4 GB |      |
| `protonets_1_20` |    100 | 10000 |   00:31:27 |     1.4 GB |      |
| `protonets_5_20` |    100 | 10000 |   00:44:14 |     1.3 GB |      |

Model checkpoints will be available soon

### Magnet Loss

**COMING SOON** - Had some issues with training as backbone net unavailable.

### RepMet

**Flowers (Training Set):**

| Model (`.yaml`)                | Epochs | Steps | Total Time | GPU Memory | Line |
| :----------------------------- | -----: | ----: | ---------: | ---------: | ---- |
| `repmet_flowers_1_12_4`        |     60 |  1500 |   00:32:17 |     1.1 GB |      |
| `repmet_flowers_3_12_4`        |     60 |  1500 |   00:33:45 |     1.3 GB |      |
| `repmet_flowers_3_12_4_r18l`   |    150 |  3750 |   01:28:26 |     3.5 GB |      |
| `repmet_flowers_inc3F_1_12_4`  |    150 |  3750 |   02:04:54 |     3.5 GB |      |
| `repmet_flowers_inc3F_3_12_4`  |    150 |  3750 |   01:59:57 |     3.5 GB |      |
| `repmet_flowers_inc3L_1_12_4`  |    150 |  3750 |   00:00:00 |     8.6 GB |      |

**Pets (Training Set):**

| Model (`.yaml`)                | Epochs | Steps | Total Time | GPU Memory | Line |
| :----------------------------- | -----: | ----: | ---------: | ---------: | ---- |
| `repmet_flowers_1_12_4_r18F`   |    100 |  2500 |   00:25:29 |     ?.? GB |      |


## Results

These results are calculated with `classification/evaluate.py`

### Prototypical Networks

**Omniglot (Test Set):**

| Model (`.yaml`)  | Shot  |  Way  |     | *Orig. Paper* | *This Code**  |
| :--------------- | ----- | ----- | --- | ------------: | ------------: |
| `protonets_1_5`  |   1   |   5   |     |         98.8% |        98.81% |
| `protonets_5_5`  |   5   |   5   |     |         99.7% |        99.56% |
| `protonets_1_20` |   1   |   20  |     |         96.0% |        94.81% |
| `protonets_5_20` |   5   |   20  |     |         98.9% |        98.51% |

*One test run of 100 episodes

Model checkpoints will be available soon

### Magnet Loss

**COMING SOON** - Had some issues with training as backbone net unavailable.

### RepMet

**Flowers (Test Set):**

| Model (`.yaml`)                | Backbone    | PreTrained^ | Frozen  |  k  |  m  |  d  |     | *Orig. Paper* | *This Code**  |
| :----------------------------- | ----------- | ---------- | ------- | --- | --- | --- | --- | ------------: | ------------: |
| `repmet_flowers_1_12_4`        | Resnet18    | `true`     | `true`  |  1  | 12  |  4  |     |             - |        75.35% |
| `repmet_flowers_3_12_4`        | Resnet18    | `true`     | `true`  |  3  | 12  |  4  |     |             - |        76.18% |
| `repmet_flowers_3_12_4_r18l`** | Resnet18    | `true`     | `false` |  3  | 12  |  4  |     |             - |        44.70% |
| `repmet_flowers_inc3F_1_12_4`  | InceptionV3 | `true`     | `true`  |  1  | 12  |  4  |     |             - |        73.26% |
| `repmet_flowers_inc3F_3_12_4`  | InceptionV3 | `true`     | `true`  |  3  | 12  |  4  |     |             - |        73.25% |
| `repmet_flowers_inc3L_1_12_4`  | InceptionV3 | `true`     | `false` |  1  | 12  |  4  |     |             - |             % |
| `repmet_flowers_inc3LU_1_12_4` | InceptionV3 | `false`    | `false` |  1  | 12  |  4  |     |             - |             % |
| `repmet_flowers_3_12_4_inc3l`  | InceptionV3 | ?          | `false` |  3  | 12  |  4  |     |         89.0% |             % |

^From `torchvision`

*One test run of 500 episodes

**Currently overfits, investigating


**Pets (Test Set):**

| Model (`.yaml`)                | Backbone    | Frozen  |  k  |  m  |  d  |     | *Orig. Paper* | *This Code**  |
| :----------------------------- | ----------- | ------- | --- | --- | --- | --- | ------------: | ------------: |
| `repmet_pets_r18F_1_12_4`**    | Resnet18    | `true`  |  1  | 12  |  4  |     |             - |        84.95% |
| `repmet_pets_3_12_4_inc3l`     | InceptionV3 | `false` |  3  | 12  |  4  |     |         93.1% |             % |

*One test run of 500 episodes

**No Validation set, so might be overfitting, will rerun with val

Model checkpoints will be available soon
