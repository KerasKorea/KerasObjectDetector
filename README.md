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
* [Directory Struct](#Directory-Struct)
* [Installation](#Installation)
* [Testing](#Testing)
* [Usage](#Usage)
* [Setup environment](#Setup-environment)
* [Release information](#Release-information)
* [Contributors](#Contributors)

## Directory Struct
```
KerasObjectDetector
├── README.md
├── setup.py
├── utils
│   ├── image1.png
│   ├── image2.jpeg
│   ├── result_image.png
│   ├── show_bbox.py
│   ├── test.py
```

## Installation (In Linux)
First, Install and set up Docker Environment in your computer or server. If you don't have Docker ID, You need to sign up in [Docker Hub](https://hub.docker.com/).
```bash
  //Typing your Docker Id, Password
  docker login
  
  //Check docker version
  docker --version
```

Next, [Download YOLK package](https://github.com/KerasKorea/KerasObjectDetector). And, Set up environment for working on YOLK.
```bash
  git clone https://github.com/KerasKorea/KerasObjectDetector.git
  cd KerasObjectDetector

  //If there is no 'setuptools' in docker, please download This package.
  //pip install setuptools
   
  python setup.py
  sh setup.sh
```

## Testing

## Usage

## Setup environment

||version|
|---|---|
|python|3.6|
|tensorflow|1.14.0|
|keras|2.2.4|

## Release information
