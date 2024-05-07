from django.shortcuts import render

def home(request):
    return render(request, 'cameras/home.html')
# Create your views here.
