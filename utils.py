import cv2 as cv 
from ultralytics import YOLO

# FOR IMAGES
def img_Inf(results, image):
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

# FOR VIDEOS
def video_Inf(model_use,video_path):
    capture = cv.VideoCapture(video_path)

    while True:
        isTrue, frame = capture.read()
        results = model_use(frame)

        for r in results:
            for i, box in enumerate(r.boxes):
                cords = box.xyxy.numpy().astype(int).tolist()

                if len(cords) == 1:
                    pointer = cords[0]
                x1, y1, x2, y2 = pointer

                print(f'x1 = {x1}, y1 = {y1}, x2 = {x2}, y2 = {y2}')

                cv.putText(frame, f'Person {i}', (x1, y1 - 5), cv.FONT_HERSHEY_TRIPLEX, .5, (255,255,255), thickness= 1)
                cv.rectangle(frame, (x1,y1), (x2,y2), (0,250,0), thickness=2)
                cv.imshow('Person Detect', frame)

        if cv.waitKey(20) & 0xFF==ord('d'):
            break

    capture.release()
    cv.destroyAllWindows()
