from django.shortcuts import render,redirect
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse,HttpResponseRedirect,StreamingHttpResponse,HttpRequest
from django.template import loader
from .forms import *
from .camera import VideoCamera

def gen(camera):
	while True:
		frame = camera.get_frame()
		yield (b'--frame\r\n'
				b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


cnt = 1
queue = []

# Create your views here.
@login_required(login_url='homePage')
def home(request):
    return render(request, 'cameras/home.html')

def homePage(request):
    return render(request, 'cameras/homePage.html')

def add_camera(request):
    if request.method=="POST":
        form = Add_camera_Forms(request.POST)
        if form.is_valid():
            form.save()
            return redirect('cameras')
    else:
        form = Add_camera_Forms()

    return render(request,'cameras/add_camera.html',{"New_Camera_Form":form})