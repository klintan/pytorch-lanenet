# LaneNet in PyTorch

Adapted from https://github.com/MaybeShewill-CV/lanenet-lane-detection and https://github.com/leonfrank/lanenet-danet-pytorch

Inspiration drawn from 
https://github.com/davidtvs/PyTorch-ENet
https://github.com/sacmehta/ESPNet

Using ESPNet as Encoder-Decoder instead of ENet.


## Installation

`python setup.py install`


## Usage


### Train

To train on the test data included in the repo,

`python3 lanenet/train.py --dataset ./data/training_data_example`


#### Custom dataset
To train on a custom dataset, the easiest approach is to make sure it follows the format laid out in the data folder.
Alternatively write a custom PyTorch dataset class (if you do, feel free to provide a PR) 


### Test





## Resources


### Papers
Towards End-to-End Lane Detection: an Instance Segmentation
Approach

https://arxiv.org/pdf/1802.05591.pdf

ESPNet: Efficient Spatial Pyramid of Dilated Convolutions for Semantic Segmentation

https://arxiv.org/abs/1803.06815

ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation

https://arxiv.org/abs/1606.02147


LaneNet: Real-Time Lane Detection Networks for Autonomous Driving (This is a very similar paper, which unfortunately calls the architecture exactly the same as in 1)

https://arxiv.org/pdf/1807.01726.pdf

https://maybeshewill-cv.github.io/lanenet-lane-detection/
