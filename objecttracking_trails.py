#Plotting Tracks Over Time

#Import All the Required Libraries
import cv2
from ultralytics import YOLO
from collections import defaultdict
import numpy as np

#Load the YOLO Model
model = YOLO("yolo11n.pt")

#Create a Video Capture Object
cap = cv2.VideoCapture("Resources/Videos/video7.mp4")

#Store the Track History
track_history  =defaultdict(lambda  : [])

#Loop through the Video Frames
while True:
    ret, frame = cap.read()
    if ret:
        #Run YOLO11 tracking on the frame
        results = model.track(source=frame, persist=True)
        if results[0].boxes.id is not None:
            #Get the bounding box coordinates and the track IDs
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            #Visualize the results on the frame
            annotated_frame = results[0].plot()
            #Plot the tracks
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                track = track_history[track_id]
                track.append((float(x), float(y))) #x, y center point
                if len(track) > 30:
                    track.pop(0)
                #Draw the Tracking Lines
                points = np.hstack(track).astype(np.int32).reshape((-1,1,2))
                cv2.polylines(annotated_frame, [points], isClosed=False, color = (230,0,0), thickness=10)
            #Display the annotated frame
            cv2.imshow("YOLO11 Tracking", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('w'):
                break
    else:
        break
cap.release()
cv2.destroyAllWindows()
