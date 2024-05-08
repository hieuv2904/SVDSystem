from django.urls import path
from . import views

urlpatterns = [
    path("", views.homePage, name='homePage'),
    path("cameras/", views.home, name='cameras'),
    path("add_camera/",views.add_camera,name="add_camera"),
    path("cam0/", views.Cam0, name='cam0'),
    path("cam1/", views.Cam1, name='cam1'),
    path("cam2/", views.Cam2, name='cam2'),
]