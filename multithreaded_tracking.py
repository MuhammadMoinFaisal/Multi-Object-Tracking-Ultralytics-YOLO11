#MultiThreaded Tracking Provides the Capability to run Object Tracking on Multiple Video Streams
#Import All the Required Libraries
import threading
import cv2
from ultralytics import YOLO
#Define Model Names and Video Source
MODEL_NAMES = ["yolo11n.pt", "yolo11n-seg.pt"]

SOURCES = ["Resources/Videos/video5.mp4", "Resources/Videos/video8.mp4"]

def run_tracker_in_thread(model_name, file_name):
    #Run YOLO Tracker in its own thread for concurrent processing
    model = YOLO(model_name)
    results = model.track(source = file_name, save = True, stream = True, show=True)
    for r in results:
        pass

#Create and Start Tracker Threads using a for loop
tracker_threads = []
for video_file, model_name in zip(SOURCES, MODEL_NAMES):
    thread = threading.Thread(target=run_tracker_in_thread, args = (model_name, video_file), daemon=True)
    tracker_threads.append(thread)
    thread.start()

#Wait for all tracker threads to finish
for thread in tracker_threads:
    thread.join()

#Clean Up and Close Windows
cv2.destroyAllWindows()