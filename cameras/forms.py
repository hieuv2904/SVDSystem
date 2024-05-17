from django import forms
from .models import *


# class Add_camera_Forms(forms.ModelForm):
#     class Meta:
#         model = Add_camera
#         fields = "__all__"
class AlertLogForms(forms.ModelForm):
    class Meta:
        model = Alert_log
        fields = ['alert', 'camera_number', 'clip_link']