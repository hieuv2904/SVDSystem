import cv2
import base64
from PIL import Image
from io import BytesIO

import json
import cv2
import argparse
import os
import time
import numpy as np
import torch
from PIL import Image
import time
from datetime import datetime
from dataset.transforms import BaseTransform
from utils.misc import load_weight
from config import build_dataset_config, build_model_config
from models import build_model
import pandas as pd
import csv
from ..forms import *
# from test_video_ava import process_frame
import torch.backends.cudnn as cudnn
from alert import SimpleANN
# import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel as DDP

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# CUDA_LAUNCH_BLOCKING=1
# torch.cuda.set_device(0)

model_alert = SimpleANN()

model_alert.load_state_dict(torch.load(r'D:\yowov2V7\YOWOv2\model_weights.pth'))

model_alert.eval()


torch.backends.cudnn.enabled = False
class Args:
    def __init__(self):
        self.img_size = 224
        self.cuda = True
        self.save_folder = 'D:/yowov2V7/YOWOv2/video_output'
        self.vis_thresh = 0.1
        self.dataset = 'ava_v2.2'
        self.version = 'yowo_v2_large'
        self.weight = "D:/yowov2V7/YOWOv2/backup_dir/ava_v2.2/fps32_k16_bs16_yolo_large_newdata_p2/epoch4/yowo_v2_large_epoch_4.pth"
        self.topk = 40
        self.threshold = 0.1
args = Args()
    # cuda
if args.cuda:
    cudnn.benchmark = True
    print('use cuda')
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# config
d_cfg = build_dataset_config(args)
m_cfg = build_model_config(args)

class_names = d_cfg['label_map']
num_classes = 3

# transform
transform = BaseTransform(
    img_size=d_cfg['test_size'],
    # pixel_mean=d_cfg['pixel_mean'],
    # pixel_std=d_cfg['pixel_std']
    # pixel_mean=0,
    # pixel_std=1
    )

# build model
model = build_model(
    args=args,
    d_cfg=d_cfg,
    m_cfg=m_cfg,
    device=device, 
    num_classes=num_classes, 
    trainable=False
    )

# load trained weight
model = load_weight(model=model, path_to_ckpt=args.weight)

# to eval
model = model.to(device).eval()

def predict_frame(video_clip, list_count_fighter, num_frame, orig_h, orig_w, frame):
    fight = 0
    x = transform(video_clip)
    x = torch.stack(x, dim=1)
    x = x.unsqueeze(0).to(device) # [B, 3, T, H, W], B=1

    batch_bboxes = model(x)
    bboxes = batch_bboxes[0]
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox[:4]
        det_conf = bbox[4]
        #cls_out = [det_conf * cls_conf for cls_conf in bbox[5:]]
        cls_out = det_conf * bbox[5:]
        # rescale bbox
        x1, x2 = int(x1 * orig_w), int(x2 * orig_w)
        y1, y2 = int(y1 * orig_h), int(y2 * orig_h)

        # numpy
        cls_scores = np.array(cls_out)
        # tensor
        #cls_scores = cls_out.cpu().detach().numpy()


        if max(cls_scores) < args.threshold:
            continue
        indices = np.argmax(cls_scores)
        scores = cls_scores[indices]
        indices = [indices]
        scores = [scores]

        if len(scores) > 0:
            blk   = np.zeros(frame.shape, np.uint8)
            font  = cv2.FONT_HERSHEY_SIMPLEX
            coord = []
            text  = []
            text_size = []

#-----------------------------old---------------------------------------------#
            if indices[0]== 0:
                fight += 1
            else:
                fight+=0

            for _, cls_ind in enumerate(indices):
#-----------------------------old---------------------------------------------#
                if class_names[cls_ind] == "bully":
                    color = (0,0,255)                   
                else:
                    class_name = class_names[cls_ind]
                    if class_name == "victim":
                        color = (255,0,0)
                    else:
                        color = (0,255,0)



                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            frame = cv2.addWeighted(frame, 1.0, blk, 0.25, 1)
            for t in range(len(text)):
                cv2.putText(frame, text[t], coord[t], font, 0.75, (0, 0, 255), 2)

    # print("after predict time", time.time() - t1, "s")
    if fight >= 1:
        fight = 1
    list_count_fighter.append(fight)
    if len(list_count_fighter) > num_frame:
        list_count_fighter.pop(0)
    
    
    return frame, list_count_fighter

def addLog(camID, alert, link):
    data = {
        "alert": alert,
        "camera_number": camID,
        "clip_link": link
    }
    print(data)
    form = AlertLogForms(data)
    if form.is_valid():
        print('valid')
        form.save()

    else: print('not valid')