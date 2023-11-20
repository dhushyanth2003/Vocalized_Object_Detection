from ultralytics import YOLO
import cv2
import math
import pyttsx3

engine = pyttsx3.init()
engine.setProperty('rate', 150)  

cap = cv2.VideoCapture(0)
cap.set(3, 640)  
cap.set(4, 480) 

model = YOLO("yolo-Weights/yolov8n.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

focal_length = 1000  
known_width = 18 


FOCAL_LENGTH_CONSTANT = 50  

previous_objects = set()

while True:
    success, img = cap.read()

    if not success:
        print("Failed to capture frame. Exiting...")
        break

   
    img = cv2.resize(img, (640, 480))

    results = model(img, stream=True)

    current_objects = set()

   
    for r in results:
        boxes = r.boxes

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            
            confidence = math.ceil((box.conf[0] * 100)) / 100
            print("Confidence --->", confidence)

           
            cls = int(box.cls[0])
            print("Class name -->", classNames[cls])

            
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2

            cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)

            box_width_pixels = x2 - x1
            distance = (known_width * FOCAL_LENGTH_CONSTANT) / box_width_pixels
            print("Distance:", distance, "cm")

            current_objects.add((classNames[cls], distance))

    if current_objects != previous_objects:
        engine.stop() 
        text_to_speak = ", ".join([f"{obj[0]} at {obj[1]:.2f} cm" for obj in current_objects])
        print("Objects detected:", text_to_speak)

       
        engine.say(text_to_speak)
        engine.runAndWait()

    previous_objects = current_objects

    cv2.imshow('Webcam', img)
    if cv2.waitKey(30) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
engine.stop()
engine.runAndWait()
