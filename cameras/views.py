# from django.shortcuts import render,redirect
# from django.contrib.auth.decorators import login_required
# from django.http import HttpResponse,HttpResponseRedirect,StreamingHttpResponse,HttpRequest
# from django.template import loader
# from .forms import *
# from .camera import VideoCamera
# import cv2
# import argparse
# import os
# import time
# import numpy as np
# import torch
# from PIL import Image
# import time
# from datetime import datetime
# from dataset.transforms import BaseTransform
# from utils.misc import load_weight
# from config import build_dataset_config, build_model_config
# from models import build_model
# import pandas as pd
# import csv
# from test_video_ava import process_frame
# import torch.backends.cudnn as cudnn
# import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel as DDP
from django.shortcuts import render,redirect
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse,StreamingHttpResponse
from django.template import loader
from django.core.paginator import Paginator
from .models import *
from .forms import *
from .utils import *
import cv2
import os
import time
from alert import SimpleANN

# Create your views here.
def gen(video_clip, frame, list_count_fighter,start_time,formatted_time,formatted_time_img, IDCam):
                print("IDCam----------", IDCam)
                save_size = (1280, 720)
                fps = 6
                id_frame = 30/fps
                num_frame = 16

                alert = "Normal"
                color = (0,255,0)

                orig_h, orig_w = frame.shape[:2]
                frame, list_count_fighter = predict_frame(video_clip, list_count_fighter, num_frame, orig_h, orig_w, frame)


                if len(list_count_fighter) == num_frame:
                    # count_fight= 0
                    # for i in list_count_fighter:
                    #     count_fight += i
                    outputs = model_alert(torch.tensor(list_count_fighter).float()) 
                    predicted = outputs.round()
                    if predicted == 1:
                    #if count_fight >= num_frame/2:
                        alert = "Bullying"
                        
                        #print("Bully")
                        #df.loc[(df['video'] == video_value) & (df['id'] == count_n_frames), f'predict_{num_frame}'] = 1
                        color = (0,0,255)               
                    else:
                        alert = "Normal"

                        #print("Normal")
                        #df.loc[(df['video'] == video_value) & (df['id'] == count_n_frames), f'predict_{num_frame}'] = 0
                        color = (0,255,0)
                #frames += 1
                #count_frame += 1
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
                
    #//----------------------------------------Model---------------------------------------------------------//
                #frame_flip = cv2.flip(frame,1)
                frame_flip = frame
                text = f"Camera {IDCam}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                thickness = 2
                color_cam = (0, 0, 255) 
                (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
                x = frame_flip.shape[1] - text_width - 10 
                y = text_height + 10  
                position = (x, y)
                frame_flip = cv2.putText(frame_flip, text, position, font, font_scale, color_cam, thickness, cv2.LINE_AA)
                link = f"D:/yowov2V7/YOWOv2/video_capture/cam{IDCam}/img_{formatted_time_img}.jpg"
                if alert=="Bullying":
                    cv2.imwrite(link, frame_flip)
                    addLog(IDCam, alert, link)
                ret, jpeg = cv2.imencode('.jpg', frame_flip)
                frame_output = jpeg.tobytes()
                return list_count_fighter, frame_output
                    
                # yield (b'--frame\r\n'
                #         b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
def gen_frame(video, video_clip, list_count_fighter, IDCam):
    print("IDcam-----------", IDCam)
    save_size = (1280, 720)
    fps = 2
    id_frame = 30 / fps
    num_frame = 16
    count_frame = 0
    while True:
        ret, frame = video.read()
        if not ret:
            break
        now = datetime.now()
        formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
        formatted_time_img = now.strftime("%Y%m%d_%H%M%S")

        start_time = time.time()
        count_frame += 1
        if count_frame % id_frame == 0:
            count_frame = 0
            frame_pil = Image.fromarray(frame.astype(np.uint8))
            if len(video_clip) <= 0:
                for _ in range(num_frame):
                    video_clip.append(frame_pil)
            video_clip.append(frame_pil)
            video_clip.pop(0)
            list_count_fighter, frame_output = gen(video_clip, frame, list_count_fighter, start_time, formatted_time, formatted_time_img, IDCam)

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_output + b'\r\n\r\n')


def check_available_sources(cam_test):
    cap = cv2.VideoCapture(cam_test)
    test, _ = cap.read()
    print("i : "+str(cam_test)+" /// result: "+str(test))
    if test:
        return True
    else:
        return False

def Cam0(request):
    if check_available_sources(0):
        path_video = "D:/NO/Django_code/video_test/video_13.mp4"
        video = cv2.VideoCapture(0)
        video_clip = []
        list_count_fighter = []

        return StreamingHttpResponse(gen_frame(video, video_clip, list_count_fighter, IDCam=0),
                                     content_type='multipart/x-mixed-replace; boundary=frame')


def Cam1(request):
    if check_available_sources(1):
        path_video = "D:/NO/Django_code/video_test/test_2.mp4"
        video = cv2.VideoCapture(1)
        video_clip = []
        list_count_fighter = []

        return StreamingHttpResponse(gen_frame(video, video_clip, list_count_fighter, IDCam=1),
                                     content_type='multipart/x-mixed-replace; boundary=frame')
    
# def Cam2(request):
#     if check_available_sources(2):
#         return StreamingHttpResponse(gen(2, 2),
# 					content_type='multipart/x-mixed-replace; boundary=frame')

# def Cam3(request):
#     if check_available_sources(3):
#         return StreamingHttpResponse(gen(3, d_cfg, model, device, transform, class_names),
# 					content_type='multipart/x-mixed-replace; boundary=frame')

# def Cam4(request):
#     if check_available_sources(4):
#         return StreamingHttpResponse(gen(4, d_cfg, model, device, transform, class_names),
# 					content_type='multipart/x-mixed-replace; boundary=frame')

# def Cam5(request):
#     if check_available_sources(5):
#         return StreamingHttpResponse(gen(5, d_cfg, model, device, transform, class_names),
# 					content_type='multipart/x-mixed-replace; boundary=frame')

# def Cam6(request):
#     if check_available_sources(6):
#         return StreamingHttpResponse(gen(6, d_cfg, model, device, transform, class_names),
# 					content_type='multipart/x-mixed-replace; boundary=frame')



@login_required(login_url='homePage')
def home(request):
    return render(request, 'cameras/home.html')

def homePage(request):
    return render(request, 'cameras/homePage.html')


@login_required(login_url='homePage')
def AlertLogs(request):
    logs = Alert_log.objects.all().order_by('-time')
    paginator = Paginator(logs, 50)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    return render(request,'cameras/alertLogs.html', {"page_obj": page_obj})

@login_required(login_url='homePage')
def get_alert_logs(request):
    page_number = request.GET.get('page', 1)
    logs = Alert_log.objects.all().order_by('-time')
    paginator = Paginator(logs, 50)  # Show 50 logs per page

    page_obj = paginator.get_page(page_number)
    logs_list = list(page_obj.object_list.values('camera_number', 'alert', 'time', 'clip_link'))

    return JsonResponse({
        'logs': logs_list,
        'has_next': page_obj.has_next(),
        'has_previous': page_obj.has_previous(),
        'page_number': page_obj.number,
        'total_pages': paginator.num_pages
    })

