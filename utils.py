import cv2 as cv 
from ultralytics import YOLO

# FOR IMAGES
def img_Inf(model_use, image):
    model = YOLO(model_use)
    
    image = cv.imread(image)
    results = model(image)

    if len(results[0].boxes) == 0:
        cv.imshow('No Detections', image)
        print("NO DETECTIONS")
        cv.waitKey(0)
        return

    for r in results:
        for i, box in enumerate(r.boxes):
            class_idx = box.cls
            cords = box.xyxy.cpu().numpy().astype(int).tolist()

            if len(cords) == 1:
                pointer = cords[0]
            x1, y1, x2, y2 = pointer

            # print(f'x1 = {x1}, y1 = {y1}, x2 = {x2}, y2 = {y2}')
            print(f'{model.names[int(class_idx)]} = {round(float(box.conf), 3)}')

            cv.putText(image, f"{model.names[int(class_idx)]} || {round(float(box.conf), 3)}", (x1, y1 - 5), cv.FONT_HERSHEY_TRIPLEX, .5, (255,255,255), thickness= 1)
            cv.rectangle(image, (x1,y1), (x2,y2), (0,250,0), thickness=2)
            cv.imshow('Image', image)

    cv.waitKey(0)

# FOR VIDEOS
def video_Inf(model_use, video_path):
    model = YOLO(model_use)
    capture = cv.VideoCapture(video_path)

    while True:
        isTrue, frame = capture.read()
        results = model(frame)

        if len(results[0].boxes) == 0:
            cv.imshow('Video', frame)

        for r in results:
            for i, box in enumerate(r.boxes):
                class_idx = box.cls
                cords = box.xyxy.cpu().numpy().astype(int).tolist()

                if len(cords) == 1:
                    pointer = cords[0]
                x1, y1, x2, y2 = pointer

                # print(f'x1 = {x1}, y1 = {y1}, x2 = {x2}, y2 = {y2}')
                cv.putText(frame, f"{model.names[int(class_idx)]} || {round(float(box.conf), 3)}", (x1, y1 - 5), cv.FONT_HERSHEY_TRIPLEX, .5, (255,0,0), thickness= 1)
                cv.rectangle(frame, (x1,y1), (x2,y2), (0,250,0), thickness=2)
                cv.imshow('Video', frame)

        if cv.waitKey(20) & 0xFF==ord('d'):
            break

    capture.release()
    cv.destroyAllWindows()
