from ultralytics import YOLO
import cv2 as cv
from utils import video_Inf, img_Inf
# import torch

# Load a model
# model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
# model = YOLO("yolo11n.yaml").load("yolo11n.pt")  # build from YAML and transfer weights

# torch.cuda.empty_cache()

# Train the model
# if __name__ == '__main__':
#     results = model.train(data="D:\\VSC\\Test_train\\Weapon_Datasets\\Handgun_with_LongGun_surv\\v2\\data.yaml", epochs=200, imgsz=1080, batch=.7, device=0, patience=50)

# if __name__ == '__main__':
#     model = YOLO("D:\\VSC\\Test_train\\runs\\detect\\train\\weights\\last.pt")
#     results = model.train(resume=True, batch=.07, device=0, patience=50)

# ------------------------Load trained model------------------------------
model = YOLO("D:\\VSC\\Test_train\\runs\\detect\\train\\weights\\best.pt") # CHANGE THIS FOR PI
# model = YOLO("D:\\VSC\\Test_train\\runs\\detect\\Hanguns\\Handguns_with_LongGuns\\v2\\940x940\\train_200epoch\\weights\\best.pt")

source_vid = 0 #"D:\\VSC\\Test_train\\Vids\\short.mp4"
# source_pic = "D:\\VSC\\Test_train\\Pics\\IMG_2690-scaled.jpeg"

# ------IMAGES OR VIDEOS------
# img_Inf(model, source_pic)
video_Inf(model, source_vid)
# model.predict(source, save=False, imgsz=640, conf=0.4, show=True)


# -------EXPORTING--------
# model.export(format="ncnn", device=0, data="D:\\VSC\\Test_train\\Weapon_Datasets\\Handgun_with_LongGun_surv\\v2\\data.yaml", imgsz=940)
# model.export(format="openvino", device=0, data="D:\\VSC\\Test_train\\Datasets_to_use\\6kCars\\920x920\\data.yaml", imgsz=928)

