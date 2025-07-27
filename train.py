from ultralytics import YOLO
import numpy as np
import cv2 as cv

# Load a COCO-pretrained YOLOv8n model
model = YOLO("yolov8n.pt")

# # Train the model on the COCO8 example dataset for 100 epochs
# results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

# model = YOLO("test_model.pt")
# Run inference with the YOLOv8n model on the 'bus.jpg' image
results = model("D:\\VSC\\OpenCV\\Photos\\face.jpg")
image = cv.imread("D:\\VSC\\OpenCV\\Photos\\face.jpg")

for r in results:
    for i, box in enumerate(r.boxes):
        cords = box.xyxy.numpy().astype(int).tolist()

        if len(cords) == 1:
            pointer = cords[0]
        x1, y1, x2, y2 = pointer

        print(f'x1 = {x1}, y1 = {y1}, x2 = {x2}, y2 = {y2}')

        cv.putText(image, f'Person {i}', (x1, y1 - 5), cv.FONT_HERSHEY_TRIPLEX, .5, (255,255,255), thickness= 1)
        cv.rectangle(image, (x1,y1), (x2,y2), (0,250,0), thickness=2)
        cv.imshow('Person Detect', image)

cv.waitKey(0)


