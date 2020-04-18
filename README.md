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


#### TUSimple dataset
Download TUsimple dataset from https://github.com/TuSimple/tusimple-benchmark/issues/3

When done run the script in the `scripts`-folder (From https://github.com/MaybeShewill-CV/lanenet-lane-detection)
`python tusimple_transform.py --src_dir <directory of downloaded tusimple>`

After this run training as before:
`python3 lanenet/train.py --dataset <tusimple_transform script output folder>`

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

https://maybeshewill-cv.github.io/lanenet-lane-detection/
