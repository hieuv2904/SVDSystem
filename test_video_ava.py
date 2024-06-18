import argparse
import cv2
import os
import time
import numpy as np
import torch
from PIL import Image
import time
from alert import SimpleANN
from datetime import datetime
from dataset.transforms import BaseTransform
from utils.misc import load_weight
from config import build_dataset_config, build_model_config
from models import build_model
import pandas as pd
import csv

import torch.backends.cudnn as cudnn
# import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel as DDP

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# torch.cuda.set_device(0)
torch.backends.cudnn.enabled = False
#dist.init_process_group(backend='nccl')
model_alert = SimpleANN()

model_alert.load_state_dict(torch.load(r'D:\yowov2V7\YOWOv2\model_weights.pth'))

model_alert.eval()

def parse_args():
    parser = argparse.ArgumentParser(description='YOWOv2')

    # basic
    parser.add_argument('-size', '--img_size', default=224, type=int,
                        help='the size of input frame')
    parser.add_argument('--show', action='store_true', default=False,
                        help='show the visulization results.')
    parser.add_argument('--cuda', action='store_true', default=False, 
                        help='use cuda.')
    parser.add_argument('--save_folder', default='det_results/', type=str,
                        help='Dir to save results')
    parser.add_argument('-vs', '--vis_thresh', default=0.1, type=float,
                        help='threshold for visualization')
    parser.add_argument('--video', default='9Y_l9NsnYE0.mp4', type=str,
                        help='AVA video name.')
    parser.add_argument('-d', '--dataset', default='ava_v2.2',
                        help='ava_v2.2')

    # model
    parser.add_argument('-v', '--version', default='yowo_v2_large', type=str,
                        help='build YOWOv2')
    parser.add_argument('--weight', default='weight/',
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('--topk', default=40, type=int,
                        help='NMS threshold')
    parser.add_argument('--threshold', default=0.1, type=int,
                        help='threshold')

    return parser.parse_args()
                    
def process_frame(frame, video_clip, num_frame, transform, list_count_fighter, model, device, class_names, args, count_n_frames) : 
    
        count_n_frames += 1
        
    # to PIL image
        fight = 0
        max_score = 0
        frame_pil = Image.fromarray(frame.astype(np.uint8))

        
        if len(video_clip) <= 0:
            for _ in range(num_frame):
                video_clip.append(frame_pil)
        
        video_clip.append(frame_pil)
        video_clip.pop(0)
        #del video_clip[0]
        # orig size
        orig_h, orig_w = frame.shape[:2]
        # transform
        t_transform = time.time()
        x = transform(video_clip)
        # print("before transform", time.time() - t_transform, "s")
        # List [T, 3, H, W] -> [3, T, H, W]
        x = torch.stack(x, dim=1)
        x = x.unsqueeze(0).to(device) # [B, 3, T, H, W], B=1

        # print("preprocessing input", time.time() - start_time, "s")
        t0 = time.time()
        # inference
        batch_bboxes = model(x)
        # print("inference time ", time.time() - t0, "s")
        t1 = time.time()
        # batch size = 1
        bboxes = batch_bboxes[0]
        # visualize detection results
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
            # max_score = max(cls_scores)
            # if max_score < args.threshold:
            #     continue
            # indices = [np.argmax(cls_scores)]
            # scores = [max_score]
            # indices = np.where(cls_scores > 0.0)
            # scores = cls_scores[indices]
            # indices = list(indices[0])
            # scores = list(scores)
            if len(scores) > 0:
                blk   = np.zeros(frame.shape, np.uint8)
                font  = cv2.FONT_HERSHEY_SIMPLEX
                coord = []
                text  = []
                text_size = []

#-----------------------------old---------------------------------------------#
                if indices[0]== 0:
                    fight += 1
                    max_score = max(cls_scores[indices], max_score)
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


                    #color =  (0,255,0)
                    #print(class_names[cls_ind])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    #text.append("[{:.2f}] ".format(scores[_]) + str(class_names[cls_ind]))
                    #text_size.append(cv2.getTextSize(text[-1], font, fontScale=0.75, thickness=2)[0])
                    #coord.append((x1+3, y1+25+10*_))
                    #cv2.rectangle(blk, (coord[-1][0]-1, coord[-1][1]-20), (coord[-1][0]+text_size[-1][0]+1, coord[-1][1]+text_size[-1][1]-8), (0, 255, 0), cv2.FILLED)
                frame = cv2.addWeighted(frame, 1.0, blk, 0.25, 1)
                for t in range(len(text)):
                    cv2.putText(frame, text[t], coord[t], font, 0.75, (0, 0, 255), 2)
        # print("after predict time", time.time() - t1, "s")

        if fight >= 1:
            fight = 1
        list_count_fighter.append(fight)
        if len(list_count_fighter) > num_frame:
            list_count_fighter.pop(0)
        
        
        return frame, list_count_fighter,  fight, max_score, count_n_frames
@torch.no_grad()
def run(args, d_cfg, model, device, transform, class_names):
    csv_file = "D:/yowov2V7/YOWOv2/alert_test.csv"
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
    df = pd.read_csv(csv_file)
    video_value = "test_3"

    path_to_video = f"D:/NO/Django_code/video_test/{video_value}.mp4" 
    name = path_to_video.split("/")[-1]
    video = cv2.VideoCapture(0)
    save_size = (1280, 720)
    fps = 30
    #id_frame = 30/fps
    id_frame = 5
    num_frame = 16
    video_clip = []
    list_count_fighter = []
    alert = "Normal"
    color = (0,255,0)
    count_fight = 0
    count_frame = 0
    count_n_frames = -1
    while(True):
        
        ret, frame = video.read()
        now = datetime.now()
        formatted_time = now.strftime("%Y-%m-%d %H:%M:%S") 
        if ret:
            start_time = time.time()
            count_frame += 1
            # prepare
            if count_frame % id_frame == 0:
                count_frame = 0
                frame, list_count_fighter, fight, max_score, count_n_frames = process_frame (frame, video_clip, num_frame, transform,list_count_fighter, model, device, class_names, args, count_n_frames)
                #df.loc[(df['video'] == video_value) & (df['id'] == count_n_frames), f'predict_8'] = fight
                #df.loc[(df['video'] == video_value) & (df['id'] == count_n_frames), f'conf_score_8'] = max_score
                if len(list_count_fighter) == num_frame:
                    # print(torch.tensor(list_count_fighter).type(torch.LongTensor))
                    # print(type(torch.tensor(list_count_fighter).type(torch.LongTensor)))
                    #outputs = model_alert(torch.tensor(list_count_fighter).float()) 
                    #predicted = outputs.round()
                    count_fight = 0
                    for i in list_count_fighter:
                        count_fight += i

                    if count_fight >= num_frame/2:
                    #if predicted == 1:
                        alert = "Bullying"

                        #print("Bully")
                        #df.loc[(df['video'] == video_value) & (df['id'] == count_n_frames), f'predict_{num_frame}'] = 1
                        color = (0,0,255)               
                    else:
                        alert = "Normal"

                        #print("Normal")
                        #df.loc[(df['video'] == video_value) & (df['id'] == count_n_frames), f'predict_{num_frame}'] = 0
                        color = (0,255,0)
                # frames += 1
                # count_frame += 1
                #df.to_csv(csv_file, index=False)
                current_time = time.time()
                elapsed_time = current_time - start_time
                # print("elapsed_time", elapsed_time)
                fps = 1/elapsed_time
                # if elapsed_time >= 1:
                #     fps = frame / elapsed_time
                #     start_time = current_time
                #     frames = 0
                cv2.putText(frame, f"Time: {str(formatted_time)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2) 
                cv2.putText(frame, f"FPS: {fps:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Alert: {alert}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                # save
    #            out.write(frame)
                
                if args.show:
                    # show
                    cv2.namedWindow('key-frame detection', cv2.WINDOW_NORMAL)

                    # Thay đổi kích thước cửa sổ thành (width, height)
                    cv2.resizeWindow('key-frame detection', 1280, 720)

                    # Hiển thị khung hình trong cửa sổ
                    cv2.imshow('key-frame detection', frame)
                    cv2.imshow('key-frame detection', frame)
#------------------------------------original---------------------------------------------------------
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break


        else:
            break

    video.release()
#    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    args = parse_args()
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
    basetransform = BaseTransform(
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

    # run
    run(args=args, d_cfg=d_cfg, model=model, device=device,
        transform=basetransform, class_names=class_names)
