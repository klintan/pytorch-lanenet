import time
import os
import sys

import torch
from data_loader.data_loaders import LaneDataSet
from model.lanenet import LaneNet
from torch.utils.data import DataLoader
from torch.autograd import Variable

from utils.cli_helper import parse_args
from test import test

import numpy as np
import cv2

class AverageMeter():
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(train_loader, model, optimizer, epoch):
    batch_time = AverageMeter()
    mean_iou = AverageMeter()
    total_losses = AverageMeter()
    binary_losses = AverageMeter()
    instance_losses = AverageMeter()
    end = time.time()
    step = 0

    for batch_idx, batch in enumerate(train_loader):
        step += 1
        image_data = Variable(batch[0]).type(torch.FloatTensor)
        instance_label = Variable(batch[1]).type(torch.LongTensor)
        binary_label = Variable(batch[2]).type(torch.LongTensor)

        # forward pass
        net_output = model(image_data)

        # compute loss
        total_loss, binary_loss, instance_loss, out, train_iou = output_loss(net_output, binary_label, instance_label)

        # update loss in AverageMeter instance
        total_losses.update(total_loss.item(), image_data.size()[0])
        binary_losses.update(binary_loss.item(), image_data.size()[0])
        instance_losses.update(instance_loss.item(), image_data.size()[0])
        mean_iou.update(train_iou, image_data.size()[0])

        # reset gradients
        optimizer.zero_grad()

        # backpropagate
        total_loss.backward()

        # update weights
        optimizer.step()

        # update batch time
        batch_time.update(time.time() - end)
        end = time.time()

        if np.isnan(total_loss.item()) or np.isnan(binary_loss.item()) or np.isnan(instance_loss.item()):
            print('cost is: {:.5f}'.format(total_loss.item()))
            print('binary cost is: {:.5f}'.format(binary_loss.item()))
            print('instance cost is: {:.5f}'.format(instance_loss.item()))
            cv2.imwrite('nan_image.png', image_data[0].cpu().numpy().transpose(1, 2, 0) + VGG_MEAN)
            cv2.imwrite('nan_instance_label.png', image_data[0].cpu().numpy().transpose(1, 2, 0))
            cv2.imwrite('nan_binary_label.png', binary_label[0].cpu().numpy().transpose(1, 2, 0) * 255)
            cv2.imwrite('nan_embedding.png', pix_embedding[0].cpu().numpy().transpose(1, 2, 0))
            break
        if step % 500 == 0:
            print(
                "Epoch {ep} Step {st} |({batch}/{size})| ETA: {et:.2f}|Total:{tot:.5f}|Binary:{bin:.5f}|Instance:{ins:.5f}|IoU:{iou:.5f}".format(
                    ep=epoch + 1,
                    st=step,
                    batch=batch_idx + 1,
                    size=len(train_loader),
                    et=batch_time.val,
                    tot=total_losses.avg,
                    bin=binary_losses.avg,
                    ins=instance_losses.avg,
                    iou=train_iou,
                ))
            sys.stdout.flush()
            train_img_list = []
            for i in range(3):
                train_img_list.append(
                    compose_img(image_data, out, binary_label, net_output["instance_seg_logits"], instance_label, i))
            train_img = np.concatenate(train_img_list, axis=1)
            cv2.imwrite(os.path.join("./output", "train_" + str(epoch + 1) + "_step_" + str(step) + ".png"), train_img)
    return mean_iou.avg


def main():
    args = parse_args()

    save_path = args.save

    if not os.path.isdir(save_path):
        os.makedirs(save_path)


    train_dataset_file = os.path.join(args.dataset, 'train.txt')
    val_dataset_file = os.path.join(args.dataset, 'val.txt')

    train_dataset = LaneDataSet(train_dataset_file)
    val_dataset = LaneDataSet(val_dataset_file)

    model = LaneNet()

    train_loader = DataLoader(train_dataset, batch_size=8, num_workers=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, num_workers=8, shuffle=True)

    model = model

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    for epoch in range(0, args.epochs):
        train_iou = train(train_loader, model, optimizer, epoch)
        val_iou = test(val_loader, model, epoch)
        if (epoch + 1) % 5 == 0:
            save_model(save_path, epoch, model)
        best_iou = max(val_iou, best_iou)
        print('Best IoU : {}'.format(best_iou))


if __name__ == '__main__':
    main()
