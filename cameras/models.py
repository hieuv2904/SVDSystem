from django.db import models

# # Create your models here.
# class Add_camera(models.Model):
#     name = models.CharField(max_length=30)
#     make = models.CharField(max_length=20)
#     camera_model = models.CharField(max_length=30)
#     zoom = models.IntegerField()
#     ip = models.GenericIPAddressField (null=False)


#     def __str__(self):
#         return self.name
    
class Alert_log(models.Model):
    time = models.DateTimeField(auto_now_add=True, null=True)
    alert = models.CharField(max_length=30)
    camera_number = models.IntegerField()
    clip_link = models.CharField(max_length=300, null=True)