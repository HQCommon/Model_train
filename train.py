from ultralytics import YOLO
import numpy as np
import cv2 as cv
from utils import video_Inf, img_Inf

# Load a model
model = YOLO("yolo11n.yaml")  # build a new model from YAML
model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
model = YOLO("yolo11n.yaml").load("yolo11n.pt")  # build from YAML and transfer weights

# Train the model
results = model.train(data="D:\\VSC\\Test_train\\use_data\\data.yaml", epochs=100, imgsz=800)

metrics = model.val(data="D:\\VSC\\Test_train\\use_data\\data.yaml", plots=True)
print(metrics.confusion_matrix.to_df())


#            RETRAINING
# Load the partially trained model
# model = YOLO("path/to/last.pt")

# Resume training
# results = model.train(resume=True)
