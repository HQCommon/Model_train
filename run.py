from ultralytics import YOLO
from utils import video_Inf, img_Inf

model = YOLO("best_ncnn_model") 

source_vid = 0 #0 for webcam or camera

video_Inf(model, source_vid)