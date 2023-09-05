"""
From https://github.com/nyoki-mtl/pytorch-discriminative-loss/blob/master/src/loss.py
This is the implementation of following paper:
https://arxiv.org/pdf/1802.05591.pdf
This implementation is based on following code:
https://github.com/Wizaron/instance-segmentation-pytorch
"""

from torch.nn.modules.loss import _Loss
from torch.autograd import Variable
import torch
from torch.functional import F

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class DiscriminativeLoss(_Loss):

    def __init__(self, delta_var=0.5, delta_dist=1.5, norm=2, alpha=1.0, beta=1.0, gamma=0.001,
                 usegpu=False, size_average=True):
        super(DiscriminativeLoss, self).__init__(reduction='mean')
        self.delta_var = delta_var
        self.delta_dist = delta_dist
        self.norm = norm
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.usegpu = usegpu
        assert self.norm in [1, 2]

    def forward(self, input, target):
        # _assert_no_grad(target)
        return self._discriminative_loss(input, target)

    def _discriminative_loss(self, embedding, seg_gt):
        batch_size = embedding.shape[0]
        embed_dim = embedding.shape[1]

        var_loss = torch.tensor(0, dtype=embedding.dtype, device=embedding.device)
        dist_loss = torch.tensor(0, dtype=embedding.dtype, device=embedding.device)
        reg_loss = torch.tensor(0, dtype=embedding.dtype, device=embedding.device)

        for b in range(batch_size):
            embedding_b = embedding[b]  # (embed_dim, H, W)
            seg_gt_b = seg_gt[b]

            labels = torch.unique(seg_gt_b)
            labels = labels[labels != 0]
            num_lanes = len(labels)
            if num_lanes == 0:
                # please refer to issue here: https://github.com/harryhan618/LaneNet/issues/12
                _nonsense = embedding.sum()
                _zero = torch.zeros_like(_nonsense)
                var_loss = var_loss + _nonsense * _zero
                dist_loss = dist_loss + _nonsense * _zero
                reg_loss = reg_loss + _nonsense * _zero
                continue

            centroid_mean = []
            for lane_idx in labels:
                seg_mask_i = (seg_gt_b == lane_idx)
                if not seg_mask_i.any():
                    continue
                embedding_i = embedding_b[seg_mask_i]

                mean_i = torch.mean(embedding_i, dim=0)
                centroid_mean.append(mean_i)

                # ---------- var_loss -------------
                var_loss = var_loss + torch.mean(F.relu(
                    torch.norm(embedding_i - mean_i, dim=0) - self.delta_var) ** 2) / num_lanes
            centroid_mean = torch.stack(centroid_mean)  # (n_lane, embed_dim)

            if num_lanes > 1:
                centroid_mean1 = centroid_mean.reshape(-1, 1, embed_dim)
                centroid_mean2 = centroid_mean.reshape(1, -1, embed_dim)
                dist = torch.norm(centroid_mean1 - centroid_mean2, dim=2)  # shape (num_lanes, num_lanes)
                dist = dist + torch.eye(num_lanes, dtype=dist.dtype,
                                        device=dist.device) * self.delta_dist  # diagonal elements are 0, now mask above delta_d

                # divided by two for double calculated loss above, for implementation convenience
                dist_loss = dist_loss + torch.sum(F.relu(-dist + self.delta_dist) ** 2) / (
                        num_lanes * (num_lanes - 1)) / 2

            # reg_loss is not used in original paper
            # reg_loss = reg_loss + torch.mean(torch.norm(centroid_mean, dim=1))

        var_loss = var_loss / batch_size
        dist_loss = dist_loss / batch_size
        reg_loss = reg_loss / batch_size
        return var_loss, dist_loss, reg_loss


class HNetLoss(_Loss):
    """
    HNet Loss
    """

    def __init__(self, gt_pts, transformation_coefficient, name, usegpu=True):
        """

        :param gt_pts: [x, y, 1]
        :param transformation_coeffcient: [[a, b, c], [0, d, e], [0, f, 1]]
        :param name:
        :return: 
        """
        super(HNetLoss, self).__init__()

        self.gt_pts = gt_pts

        self.transformation_coefficient = transformation_coefficient
        self.name = name
        self.usegpu = usegpu

    def _hnet_loss(self):
        """

        :return:
        """
        H, preds = self._hnet()
        x_transformation_back = torch.matmul(torch.inverse(H), preds)
        loss = torch.mean(torch.pow(self.gt_pts.t()[0, :] - x_transformation_back[0, :], 2))

        return loss

    def _hnet(self):
        """

        :return:
        """
        self.transformation_coefficient = torch.cat((self.transformation_coefficient, torch.tensor([1.0])),
                                                    dim=0)
        H_indices = torch.tensor([0, 1, 2, 4, 5, 7, 8])
        H_shape = 9
        H = torch.zeros(H_shape)
        H.scatter_(dim=0, index=H_indices, src=self.transformation_coefficient)
        H = H.view((3, 3))

        pts_projects = torch.matmul(H, self.gt_pts.t())

        Y = pts_projects[1, :]
        X = pts_projects[0, :]
        Y_One = torch.ones(Y.size())
        Y_stack = torch.stack((torch.pow(Y, 3), torch.pow(Y, 2), Y, Y_One), dim=1).squeeze()
        w = torch.matmul(torch.matmul(torch.inverse(torch.matmul(Y_stack.t(), Y_stack)),
                                      Y_stack.t()),
                         X.view(-1, 1))

        x_preds = torch.matmul(Y_stack, w)
        preds = torch.stack((x_preds.squeeze(), Y, Y_One), dim=1).t()
        return (H, preds)

    def _hnet_transformation(self):
        """
        """
        H, preds = self._hnet()
        x_transformation_back = torch.matmul(torch.inverse(H), preds)

        return x_transformation_back

    def forward(self, input, target, n_clusters):
        return self._hnet_loss(input, target)
