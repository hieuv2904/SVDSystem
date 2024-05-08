from django.shortcuts import render,redirect
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse,HttpResponseRedirect,StreamingHttpResponse,HttpRequest
from django.template import loader
from .forms import *
from .camera import VideoCamera
import cv2

def gen(camera):
	while True:
		frame = camera.get_frame()
		yield (b'--frame\r\n'
				b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def check_available_sources(cams_test):
    available_sources = []
    for i in range(0, cams_test):
        cap = cv2.VideoCapture(i)
        test, _ = cap.read()
        if test:
             available_sources.append(i)
        print("i : "+str(i)+" /// result: "+str(test))
    return available_sources

def Cam0(request):
    cam0 = VideoCamera(0)
    return StreamingHttpResponse(gen(cam0),
					content_type='multipart/x-mixed-replace; boundary=frame')
def Cam1(request):
    cam1 = VideoCamera(6)
    return StreamingHttpResponse(gen(cam1),
					content_type='multipart/x-mixed-replace; boundary=frame')

def Cam2(request):
    cam2 = VideoCamera(2)
    return StreamingHttpResponse(gen(cam2),
					content_type='multipart/x-mixed-replace; boundary=frame')
    # cam2 = VideoCamera(2)
    # cam3 = VideoCamera(3)
    # cam4 = VideoCamera(4)




# Create your views here.
@login_required(login_url='homePage')
def home(request):
    return render(request, 'cameras/home.html')

def homePage(request):
    return render(request, 'cameras/homePage.html')

@login_required(login_url='homePage')
def add_camera(request):
    if request.method=="POST":
        form = Add_camera_Forms(request.POST)
        if form.is_valid():
            form.save()
            return redirect('cameras')
    else:
        form = Add_camera_Forms()

    return render(request,'cameras/add_camera.html', {"New_Camera_Form":form})