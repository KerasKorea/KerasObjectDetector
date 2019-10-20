# Keras Object Detection API

_You Only Look Keras_ <img width="50" height="27" src="https://user-images.githubusercontent.com/23257678/65056804-81fdb180-d9ac-11e9-931b-d027649c67cc.png" alt="">

![Object](https://img.shields.io/badge/Object-Detector-Yellow.svg)
![Language](https://img.shields.io/badge/Language-Python-blue.svg)
[![DeepLearning](https://img.shields.io/badge/DeepLearning-Keras-red.svg)](https://keras.io)
[![KerasKorea](https://img.shields.io/badge/Community-KerasKorea-purple.svg)](https://www.facebook.com/groups/KerasKorea/)
[![KerasKorea](https://img.shields.io/badge/2019-Contributhon-green.svg)](https://www.kosshackathon.kr/)

<p align="center">
  <img width="509" height="276" src="
./res/YOLKteam_object_dection.png" alt="">
</p>
<p align="center">

As a 2019 Open Source Countibuthon Project, we create Kears Object Detection API.  
The purpose of Keras Object Detection API is to make it easy and fast so that everyone can create their own Object Detection model without difficulty.
Detailed instructions for use will be updated at a later. You can look forward to it. 🤓

## Contents
* [Directory Structure](##Directory-Structure)
* [Installation](##Installation)
* [Tutorial](##Tutorial)
* [Quick Start](##Quick-Start)
* [Dependencies](##Dependencies)
* [Release information](##Release-information)
* [Contributors](##Contributors)

## Directory Structure
<!--need to edit-->
```
KerasObjectDetector
├── README.md
├── setup.py
├── setup.md
├── docker files
├── datasets
├── utils
│   ├── image1.png
│   ├── image2.jpeg
│   ├── result_image.png
│   ├── show_bbox.py
│   ├── test.py...
```

## Installation (On Linux)

First, [Download YOLK API](https://github.com/KerasKorea/KerasObjectDetector) that help to set up development environment for working on object detection. Enter the following command in terminal.

```bash
  # Download YOLK API
  $ git clone https://github.com/KerasKorea/KerasObjectDetector.git
  $ cd KerasObjectDetector

  # If there is no 'setuptools' in docker, please download This package.
  # pip install setuptools
  # install library
  $ apt-get install libatlas-base-dev libxml2-dev libxslt-dev python-tk
  
  # build setup code
  # ./KerasObjectDetector
  $ python setup.py install
```

If you want to running on Docker, Get Docker Image we made and easily configure development environment.(Later, It will be upload on Docker Hub)

```bash
  #  pull yolk docker image
  $ docker pull kerasyolk/yolk

  # yolk run
  $ docker run --name=yolk -Pit -p 8888:8888 -p 8022:22 kerasyolk/yolk:latest

  # running jupyter-notebook
  $ jupyter-notebook
```

## Tutorial
<!-- used by inference -->
You can test your image with YOLK API. Go to the [Tutorial](https://github.com/KerasKorea/KerasObjectDetector/Object_detection_tutorial_by_keras_API.md) :)

## Quick Start
_to be added later..._

## Dependencies
|Name|Version(Min)|
|---|---|
|Tensorflow|1.14.0|
|Keras|2.3.0|
|Python|3.6|
|Numpy|1.14|
|Matplotlib||
|SciPy|0.14|
|h5py||
|Pillow||
|progressbar2||
|opencv-python|3.3.0|
|six|1.9.0|
|PyYAML||
|Cython||

## Release information
#### ver 1.0.0 (November 20, 2019) 
Finally, API that can detect multiple objects in keras has been completed!! There are still many things to supply, but we plan to continue to update. This release includes: 

1. A three of object detetion model and a data generator that changes in a suitable data format for selected model. 
    - Object Detection Models : [SSD](https://github.com/pierluigiferrari/ssd_keras), [YOLOv3](https://github.com/qqwweee/keras-yolo3), [RetinaNet](https://github.com/fizyr/keras-retinanet)
    - Dataset and data generator : [PASCAL VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/), [COCO](http://cocodataset.org/#home), [Custom dataset]()  <!--need to description-->
      ㄴ Yolk's dataset downloader is 3X faster than existing downloader.
2. Docker files that help to set up easliy development environment.
3. Easy & Detail Obejct Detection Tutorial (SSD+VOC2012)

## Contributors
Thanks goes to these beautiful peaple (github ID) :
[@fuzzythecat](https://github.com/fuzzythecat), [@mijeongjeon](https://github.com/mijeongjeon), [@tykimos](https://github.com/tykimos), [@SooDevv](https://github.com/SooDevv), [@karl6885](https://github.com/karl6885), [@EthanJYK](https://github.com/EthanJYK), [@minus31](https://github.com/minus31), [김형섭](), [최민영](), [@mike2ox](https://github.com/mike2ox), [@hngskj](https://github.com/hngskj), [@hics33](https://github.com/hics33), [@aaajeong](https://github.com/aaajeong), [@parkjh688](https://github.com/parkjh688), [@Uwonsang](https://github.com/Uwonsang), [@simba328](https://github.com/simba328), [@visionNoob](https://github.com/visionNoob), [이혜리](), [@melonicedlatte](https://github.com/melonicedlatte), [전지영](), [@ahracho](https://github.com/ahracho)
