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
* [Training](#Training)
* [Usage](#Usage)
* [Dependencies](#Dependencies)
* [Release information](#Release-information)
* [Contributors](#Contributors)

## Directory Structure
```
KerasObjectDetector
├── README.md
├── setup.py
├── setup.md
├── datasets
├── utils
│   ├── image1.png
│   ├── image2.jpeg
│   ├── result_image.png
│   ├── show_bbox.py
│   ├── test.py...
```

## Installation (On Linux)

First, [Download YOLK package](https://github.com/KerasKorea/KerasObjectDetector). And, Set up development environment for working on YOLK. 터미널에서 밑의 명령어를 입력하시오

```bash
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

If you want to use Docker, Get Docker Image we made and easily configure development environment.

```bash
  # ./KerasObjectDetector
  $ docker run --name=yolk -Pit
  # start yolk container
  $ docker start yolk
```
<!-- 이미지 다운로드-->
```python
  # 코드 형식 참고 - https://keras.io/applications/#fine-tune-inceptionv3-on-a-new-set-of-classes
  import keras
  import yolk

  voc_train, voc_valid = yolk.datasets.pascal_voc07()

  load_model = load_model(model_path, backbone_name='resnet50')

  preds = model.predict_detection(x) # preds = (boxes, scores, labels)
  # ㄴmodel.convert2pred_model() 
  #   ㄴmodel.add(yolk.layers.yolo_predict_layer())
  #     yolk.models.yolov3() 선언할 때 Sequential(name="yolov3") 해주고
  #     여기에 맞춰서 각 모델에 맞는 convert_model() 함수 호출

  # bbox 포함한 결과 plotting
  plt = yolk.plotter(preds)
  plt.plot()
```

## Testing

## Training

## Usage

## Dependencies

|Tool|Version|
|---|---|
|python|3.6|
|tensorflow|1.14.0|
|keras|2.3.0|

## Release information

## Contributors