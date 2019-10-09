# Let's Detect Obejcts by Keras API

## Introduction

<p align="center">
    <img width="510" height="276" src = "https://upload.wikimedia.org/wikipedia/commons/3/38/Detected-with-YOLO--Schreibtisch-mit-Objekten.jpg"></p>

__Object Detection__ is one of the most popular computer vision technologies in many areas.(Face detection, Self-driving car etc) Recently, __Deep Learning__ technology has greatly influenced the Object Detection field, such as accuracy, performance improvement. 
There are several popular deep learning algorithms. __Faster R-CNN(2015)__, __YOLO(2015)__, __SSD(2015)__ and __RetinaNet(2017)__. In this tutorial, we will  __Faster R-CNN(2015)__ to learn what object detection is. 

#### _What's Faster R-CNN?_
[Faster R-CNN(2015)](https://arxiv.org/pdf/1506.01497.pdf) is one of the R-CNN models that extracts Region Proposals **used by Region Proposal Network(RPN)** and classifies them on the basis of CNN models. 

[model picture]()

Before Faster R-CNN, region proposals were extracted using selvective serarch or edge boxes. However, These methods are slower than gpu computation and cause to occur bottleneck because they operate on cpu computation outside the CNN model.


#### _What's Deepfashion2?_

<p align="center">
    <img width="510" height="276" src = "https://raw.githubusercontent.com/switchablenorms/DeepFashion2/master/images/deepfashion2_bigbang.png"></p>

 In fact, many object detection tutorials use famous dataset such as [COCO](http://cocodataset.org/), [VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) etc. But, this tutorial uses [DeepFashion2(2019)](https://arxiv.org/pdf/1901.07973.pdf). ~~DeepFashion2 is comprehensive fashion dataset that contains 491k images, each of which is richly labeled with style, scale, occlusion, zooming, viewpoint, bounding box, dense landmarks and pose, pixel-level masks, and pair of images of identical item from consumer and commercial store.~~(To be written by the responsible personnel.)

## Setting up Environments
In order to running this tutorial(object detection based on deep learning), many development packages and environment settings are needed. **BUT, DON'T WORRY.** In this tutorial, you can easily set up a development environment on your computer or server using a docker. **JUST FOLLOW ME**

First, we use a Docker (OS : [Ubuntu](https://docs.docker.com/install/linux/docker-ce/ubuntu/), [windows](https://docs.docker.com/docker-for-windows/)) to set up the developments package and environments required for deep learning development. ~~And, If you don't have Docker Hub ID, you can't download [our docker image](). so, you need to sign up in [Docker Hub](https://hub.docker.com/).~~ See below.

```bash
    # yolk/
    $ docker 
```

## Explain Tutorial Code

```python
    import keras
    import yolk
    
    # function path can be changed
    deepfashion2_train, deepfashion2_valid = yolk.datasets.deepfashion2()

    # OD models can be changed
    yolov3 = yolk.models.yolov3()

    # metrics should changed
    yolov3.compile(
        loss = yolk.losses.yolov3loss,
        optimizer='adam', metrics=['yolov3']
    )

    # parameter can be changed
    yolov3.fit_generator(
        generator = deepfashion2_train, 
        validation_data = deepfashion2_valid,
        epochs=20
    )
```

### 1) Download dataset & model
```python
    
```

### 2) Loading pre-trained model & dataset

```python
    
```
### 3) Let's Detect Obejcts!
```python
```

### 4) Result