from django import forms
from .models import *


class Add_camera_Forms(forms.ModelForm):
    class Meta:
        model = Add_camera
        fields = "__all__"