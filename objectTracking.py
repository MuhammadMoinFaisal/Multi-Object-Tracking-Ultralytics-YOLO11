#Import All the Required Libraries
#from ultralytics import YOLO

#Load the YOLO11 Model
#model = YOLO("yolo11n.pt")

#Tracking with default tracker bot-sort
#results = model.track(source = "Resources/Videos/video7.mp4", show = True, save=True)
#Tracking with Byte-Track
#results = model.track(source = "Resources/Videos/video8.mp4", show=True, save=True, tracker = "bytetrack.yaml", conf = 0.20, iou = 0.3)

#---------------------------------------------------#
#Python Script using OpenCV-Python (cv2) and YOLO11 to run Object Tracking on Video Frames and on Live Webcam Feed

#Import All the Required Libraries
import cv2
from ultralytics import YOLO

#Load the YOLO11 Model
model = YOLO("yolo11n.pt")

#Create a Video Capture Object
cap = cv2.VideoCapture("Resources/Videos/video5.mp4")

#Loop through Video Frames
while True:
    ret, frame = cap.read()
    if ret:
        #Run YOLO11 Tracking on the Video Frames
        results = model.track(frame, persist=True)
        #Visualize the results on the frame
        annotated_frame = results[0].plot()
        #Display the annotated frame
        cv2.imshow("YOLO11 Tracking", annotated_frame)
        #Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()







