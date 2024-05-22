import cv2
import argparse
import supervision as sv
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
import requests
from collections import Counter

ZONE_POLYGON = np.array([
    [0, 0],
    [1280 // 2, 0],
    [1280 // 2, 720],
    [0, 720]
])

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 live")
    parser.add_argument(
        "--webcam-resolution",
        default=[1280, 720],
        nargs=1,
        type=int
    )
    args = parser.parse_args()
    return args

class KeyEvent:
    def __init__(self):
        self.key = None

    def __call__(self, event):
        self.key = event.key

def show_frame(frame, key_event):
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.draw()
    plt.pause(interval=0.1)
    plt.gcf().canvas.mpl_connect('key_press_event', key_event)
    

def main():
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    model = YOLO("yolov8s.pt")

    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )
    zone = sv.PolygonZone(polygon=ZONE_POLYGON)
    zone_annotator = sv.PolygonZoneAnnotator(
        zone=zone,
        color=sv.Color.RED,
        thickness=2,
        text_thickness=4,
        text_scale=2
    )

    key_event = KeyEvent()
    counter = 0
    last_print_time = cv2.getTickCount()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        result = model(frame)[0]
        
        result=model(frame,agnostic_nms=True)[0]
        detections = sv.Detections.from_ultralytics(result)
       
        detections = detections[(detections.class_id == 63) | (detections.class_id == 67)]
        labels = [
            f"{model.model.names[detections.class_id[i]]} {detections.confidence[i]:0.2f}"
            for i in range(len(detections))
        ]

        
        current_time = cv2.getTickCount()
        elapsed_time_ms = (current_time - last_print_time) / cv2.getTickFrequency() * 1000
        if elapsed_time_ms >= 5000:
            if len(detections) > 8:
                dic = count_occurrences(detections.data['class_name'])
                # print(dic)
                last_print_time = cv2.getTickCount()

                payload = {
                "data":dic
                }
                print(payload)
                response = requests.post("https://gemini.up.railway.app/api/gemini/realtimeupdate",json=payload)

                # Check for successful response (may not always be 200)
                if response.status_code == 200:
                # Access the response data (assuming JSON format)
                    print(response.status_code)
                else:
                    print(response.status_code)
        
        frame = box_annotator.annotate(
            scene=frame,
            detections=detections,
            labels=labels
        )

        zone.trigger(detections=detections)
        frame = zone_annotator.annotate(scene=frame)
         

        show_frame(frame, key_event)
        if key_event.key == 'escape':
            break

    cap.release()
    # plt.close()

def count_occurrences(values):
  """
  This function takes an array of strings and returns a dictionary
  where keys are unique values and values are their counts.
  """
  counts = Counter(values)
  return dict(counts)




if __name__ == "__main__":
    main()
