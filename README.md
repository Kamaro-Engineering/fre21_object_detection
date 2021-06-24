# FRE Object Detection

<p float="left" align="middle"> 
  <img src="https://kamaro-engineering.de/wp-content/uploads/2015/03/Kamaro_Logo-1.png" width="250" style="margin: 10px;">
</p>
<p align="middle">
  <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black"/></a>
  <a href="https://opensource.org/licenses/BSD-3-Clause"><img src="https://img.shields.io/badge/License-BSD%203--Clause-blue.svg" alt="License: GPL v3"/></a>
</p>
<p align="middle">
  This package contains the code used by Kamaro Engineering e.V. for the [Task 3 (object detection)](https://web.archive.org/web/20210617220835/https://www.fieldrobot.com/event/index.php/contest/task-3/) on the virtual Field Robot Event 2021 (June 8-10).
</p>
<p align="middle">
  This package assumes you are working on <b>ROS Noetic</b>. Older versions of ROS will be incompatible, since this package uses Python 3.
</p>

## Working principle

We use camera data and perform [semantic segmentation](https://en.wikipedia.org/wiki/Image_segmentation#Groups_of_image_segmentation) on camera images. Each pixel is assigned a class (background/maize/weed/litter). If a sufficient number of pixels in the current image belong to either the "weed" or the "litter" class, this is counted as a detection.

## How to use
Our fully trained network used in Task 3 is available in the ONNX format [on our NextCloud](https://nextcloud.kamaro-engineering.de/s/zNkaM3Nw7eA6Mi7). 

If you want to use it right away, place it into the `resources/` folder in this repository with the filename `net.onnx`.

To start the object detection use:
```
rosrun fre_object_detection fre21_object_detection_node.py
```

The following topics are used:
```
Node [/fre21_object_detection_node]
Publications: 
 * /detector_debug [sensor_msgs/Image]
 * /fre_detections [std_msgs/String]
 * /rosout [rosgraph_msgs/Log]

Subscriptions: 
 * /front/image_raw [sensor_msgs/Image]
 * /odometry/filtered [nav_msgs/Odometry]
```

## Dataset
Our dataset containing 192 hand-labeled images is available [on our NextCloud](https://nextcloud.kamaro-engineering.de/s/zYgz2JrgQCzY52t).
You will need to download it, or create your own dataset, in order to train the network. Place the downloaded files into the `gazebo_data/` directory in the root of this repository, containing the `images` and `labels` subdirectories.

### Dataset format
There are two folders:
* `images` - contains data (640x480 pixels, RGB, PNG)
* `labels` - files have the same names as in `images`, with relevant areas painted in
  * red (`#ff0000`) to mark litter
  * green (`#00ff00`) to mark maize plants
  * blue (`#0000ff`) to mark weeds

The data loader can be found in `gazebo_screenshot_dataset.py`. Some tolerance is applied to the color values. Other colors are ignored and interpreted as the background class.

## How to train your own Network
For our entry, in the FRE 2021 competition, we used ResNet50 with a 75%/25% split on training/validation data. In this repository, we provide three Jupyter notebooks illustrating the training process, in order to document our training process and to facilitate training on new datasets.

* [`0-Dataset-Visualization.ipynb`](0-Dataset-Visualization.ipynb) - Walks through our data loader
* [`1-Training.ipynb`](1-Training.ipynb) - Train a network on a dataset and export it to onnx
* [`2-Inference.ipynb`](2-Inference.ipynb) - Run inference using ONNX using the trained network

The code in the training notebook is heavily based on [a tutorial from Pytorch's official documentation](https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html). As you can see in the notebook, we used a pretrained ResNet50 network from the torchvision packages as a starting point. You can use any other semantic segmentation network that is capable segmenting a 640x480 RGB image into four classes and has the same resolution for the output. Once you export it to onnx, it should work with our ROS node as a drop-in replacement.

Different models usually have different requirements regarding performance and memory, and produce outputs of varying quality. We found ResNet50 to produce quality results, while being trainable on older graphics cards (e.g. GTX 1080) and having sufficient performance when being run on the CPU for inference during the competition.

## Install dependencies

(ROS Noetic only)

```bash
rosdep install fre_object_detection
pip3 install onnxruntime
```

Rosdep should install all missing dependencies. Here is a list of all needed packages:
```
ros-noetic-rospy
ros-noetic-std-msgs
ros-noetic-sensor-msgs
ros-noetic-nav-msgs
ros-noetic-geometry-msgs
ros-noetic-cv-bridge
python3-rospkg
python3-numpy
python3-opencv
onnxruntime # pip3
```

For training your own networks, you additionaly need pytorch and torchvision, which should be installed according to their [official instructions](https://pytorch.org/get-started/locally/) for your platform. For running the notebooks, you will also need:
```bash
pip3 install jupyter ipywidgets matplotlib
```

# License [![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
The code in this repository is (c) by Kamaro Engineering e.V., subject to the file [LICENSE.kamaro](LICENSE.kamaro).

The neural network training code has been derived from [Pytorch's official tutorial](https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html) written by Nathan Inkawhich. That code is (c) by the Pytorch contributors and made
available under the terms of [LICENSE.pytorch](LICENSE.pytorch).
