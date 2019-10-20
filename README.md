# Keras Object Detection API (YOLK)

_You Only Look Keras :fried_egg:_

![Object](https://img.shields.io/badge/Object-Detector-Yellow.svg)
![Language](https://img.shields.io/badge/Language-Python-blue.svg)
[![DeepLearning](https://img.shields.io/badge/DeepLearning-Keras-red.svg)](https://keras.io)
[![KerasKorea](https://img.shields.io/badge/Community-KerasKorea-purple.svg)](https://www.facebook.com/groups/KerasKorea/)
[![KerasKorea](https://img.shields.io/badge/2019-Contributhon-green.svg)](https://www.kosshackathon.kr/)

<p align="center">
  <img width="509" height="276" src="https://user-images.githubusercontent.com/23257678/65056804-81fdb180-d9ac-11e9-931b-d027649c67cc.png" alt="">
</p>

As a 2019 Open Source Countibuthon Project, we create Kears Object Detection API.  
The purpose of Keras Object Detection API is to make it easy and fast so that everyone can create their own Object Detection model without difficulty.
Detailed instructions for use will be updated at a later. You can look forward to it. 🤓

## Contents
* [Directory Structure](#Directory-Structure)
* [Installation](#Installation)
* [Testing](#Testing)
* ~~[Training](#Training)~~
* ~~[Usage](#Usage)~~
* [Dependencies](#Dependencies)
* [Release information](#Release-information)
* [Contributors](#Contributors)

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
  # ./KerasObjectDetector
  $ docker run --name=yolk -Pit
  # start yolk container
  $ docker start yolk
```

## Testing
<!-- used by inference -->
You can test your image with YOLK API. Enter the following command in terminal.

```bash
  # ./
  # python examples/retinanet_inference_example.py --filepath 000000008021.jpg
```
`retinanet_inference_example.py` is as follows. If you want to know object detection more logically, Go to [this tutorial]().

```python
  import sys, os
  sys.path.insert(0, os.path.abspath('..'))
  import yolk
  from PIL import Image

  # import miscellaneous modules
  import matplotlib.pyplot as plt
  import cv2
  import numpy as np

  def main():
      # Enter your image path for test (in `Image.open(here!!)`)
      image = np.asarray(Image.open('000000008021.jpg').convert('RGB'))
      image = image[:, :, ::-1].copy()
      # Return image info and scale
      image, scale = yolk.detector.preprocessing_image(image)

      # Approach a pretrained-retinanet (inference)
      model_path = os.path.join('..', 'resnet50_coco_best_v2.1.0.h5')
      model = yolk.detector.load_model(model_path)

      # Return bounding box, score, classes(it's labels) passed by the model
      boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))

      # Print bounding box in image
      print(boxes)
      
  if __name__ == '__main__':
      main()
```
## Training
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
    - Dataset and data generator : [PASCAL VOC2012](), [COCO](), [Custom dataset]()  <!--need to description-->
      ㄴ Yolk's dataset downloader is 3X faster than existing downloader.
2. Docker files that help to set up easliy development environment.
3. Easy & Detail Obejct Detection Tutorial (SSD+VOC2012)

## Contributors
Thanks goes to these beautiful peaple (github ID) :
[@fuzzythecat](https://github.com/fuzzythecat), [@mijeongjeon](https://github.com/mijeongjeon), [@tykimos](https://github.com/tykimos), [@SooDevv](https://github.com/SooDevv), [@karl6885](https://github.com/karl6885), [@EthanJYK](https://github.com/EthanJYK), [@minus31](https://github.com/minus31), [김형섭](), [최민영](), [@mike2ox](https://github.com/mike2ox), [홍석주](), [박근표](), [@aaajeong](https://github.com/aaajeong), [@parkjh688](https://github.com/parkjh688), [@Uwonsang](https://github.com/Uwonsang), [@simba328](https://github.com/simba328), [@visionNoob](https://github.com/visionNoob), [이혜리](), [임재곤](), [전지영](), [@ahracho](https://github.com/ahracho)
