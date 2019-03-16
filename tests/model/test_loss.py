import torch

from lanenet.model import HNetLoss


def test_hnet():
    gt_labels = torch.tensor([[[1.0, 1.0, 1.0], [2.0, 2.0, 1.0], [3.0, 3.0, 1.0]],
                              [[1.0, 1.0, 1.0], [2.0, 2.0, 1.0], [3.0, 3.0, 1.0]]],
                             dtype=torch.float32).view(6,3)
    transformation_coffecient = torch.tensor([0.58348501, -0.79861236, 2.30343866,
                                               -0.09976104, -1.22268307, 2.43086767],
                                             dtype=torch.float32)

    # import numpy as np
    # c_val = [0.58348501, -0.79861236, 2.30343866,
    #          -0.09976104, -1.22268307, 2.43086767]
    # R = np.zeros([3, 3], np.float32)
    # R[0, 0] = c_val[0]
    # R[0, 1] = c_val[1]
    # R[0, 2] = c_val[2]
    # R[1, 1] = c_val[3]
    # R[1, 2] = c_val[4]
    # R[2, 1] = c_val[5]
    # R[2, 2] = 1
    #
    # print(np.mat(R).I)
    hnet_loss = HNetLoss(gt_labels, transformation_coffecient, 'loss')
    hnet_inference = HNetLoss(gt_labels, transformation_coffecient, 'inference')

    _loss = hnet_loss._hnet_loss()

    _pred = hnet_inference._hnet_transformation()

    print("loss: ", _loss)
    print("pred: ", _pred)
