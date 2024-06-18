import cv2
from threading import Thread
import imutils
from collections import deque
import time
from test_video_ava import process_frame
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
import argparse
import cv2
import os
import time
import numpy as np
import torch
from PIL import Image
import time
from datetime import datetime
from dataset.transforms import BaseTransform
from utils.misc import load_weight
import torch.backends.cudnn as cudnn

class VideoCamera(object):
	def __init__(self, camID):
		self.video = cv2.VideoCapture(camID)

	def __del__(self):
		self.video.release()

	def get_frame(self):
		success, image = self.video.read()
		
		frame_flip = cv2.flip(image,1)
		ret, jpeg = cv2.imencode('.jpg', frame_flip)
		
		#cv2.imshow('key-frame detection', frame)

		return jpeg.tobytes()