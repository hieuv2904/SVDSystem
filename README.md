# School Violence Detection System
Capstone project in HCMUT

### Model
This system use YOWOv2 trained on our custom school violence dataset

### Alert 
When frame has object , which class "bully", this frame has 1 point. Otherwise , this frame has 0 point.
With 16 continuous frame, we have a list points of 16 frame, then we fit it to the pretrained ANN to predict the output has target 1 (corresponding on bullying behaviour), or 0 (corresponding on not bullying behaviour).

### Web App 
Using Django for system multi-cam.


### Demo video (run test_video_ava.py)
Link Drive : https://drive.google.com/file/d/12mCQVTvjSdURxttv_2uezwtH-QemvgOB/view?usp=drive_link

### References
@article{yang2023yowov2,
  title={YOWOv2: A Stronger yet Efficient Multi-level Detection Framework for Real-time Spatio-temporal Action Detection},
  author={Yang, Jianhua and Kun, Dai},
  journal={arXiv preprint arXiv:2302.06848},
  year={2023}
}
