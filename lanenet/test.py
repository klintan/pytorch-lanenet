import time
import os
import sys

import torch
from torch.autograd import Variable
import numpy as np
import cv2

from lanenet.model.model import compute_loss
from lanenet.utils.average_meter import AverageMeter

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def test(val_loader, model, epoch):
    model.eval()
    step = 0
    batch_time = AverageMeter()
    total_losses = AverageMeter()
    binary_losses = AverageMeter()
    instance_losses = AverageMeter()
    mean_iou = AverageMeter()
    end = time.time()
    val_img_list = []
    # val_img_md5 = open(os.path.join(im_path, "val_" + str(epoch + 1) + ".txt"), "w")
    for batch_idx, input_data in enumerate(val_loader):
        step += 1
        image_data = Variable(input_data["input_tensor"]).to(DEVICE)
        instance_label = Variable(input_data["instance_label"]).to(DEVICE)
        binary_label = Variable(input_data["binary_label"]).to(DEVICE)

        # output process
        net_output = model(image_data)
        total_loss, binary_loss, instance_loss, out, val_iou = compute_loss(net_output, binary_label, instance_label)
        total_losses.update(total_loss.item(), image_data.size()[0])
        binary_losses.update(binary_loss.item(), image_data.size()[0])
        instance_losses.update(instance_loss.item(), image_data.size()[0])
        mean_iou.update(val_iou, image_data.size()[0])

        # if step % 100 == 0:
        #    val_img_list.append(
        #        compose_img(image_data, out, binary_label, net_output["instance_seg_logits"], instance_label, 0))
        #    val_img_md5.write(input_data["img_name"][0] + "\n")
    #        lane_cluster_and_draw(image_data, net_output["binary_seg_pred"], net_output["instance_seg_logits"], input_data["o_size"], input_data["img_name"], json_path)
    batch_time.update(time.time() - end)
    end = time.time()

    print(
        "Epoch {ep} Validation Report | ETA: {et:.2f}|Total:{tot:.5f}|Binary:{bin:.5f}|Instance:{ins:.5f}|IoU:{iou:.5f}".format(
            ep=epoch + 1,
            et=batch_time.val,
            tot=total_losses.avg,
            bin=binary_losses.avg,
            ins=instance_losses.avg,
            iou=mean_iou.avg,
        ))
    sys.stdout.flush()
    val_img = np.concatenate(val_img_list, axis=1)
    # cv2.imwrite(os.path.join(im_path, "val_" + str(epoch + 1) + ".png"), val_img)
    # val_img_md5.close()
    return mean_iou.avg
