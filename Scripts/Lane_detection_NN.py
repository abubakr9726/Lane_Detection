import numpy as np
import cv2
from cvlib.object_detection import YOLO
from tensorflow import keras
import time
import os
from Configuration import config

data_array = config()
INPUT_FILE, LABELS_FILE, CONFIG_FILE, WEIGHTS_FILE, lane_detection_model, \
CONFIDENCE_THRESHOLD, nms_thresh, vid_output, GPU_enable, write_conf,\
Lane_detection, Object_detection = data_array[0], data_array[1], data_array[2], \
                                   data_array[3], data_array[4], data_array[5], data_array[6], \
                                   data_array[7], data_array[8], data_array[9], data_array[10], data_array[11]


yolo = YOLO(WEIGHTS_FILE, CONFIG_FILE, LABELS_FILE)
model = keras.models.load_model(lane_detection_model)

#checking output directories, if not present then creating the output directories
if not os.path.exists ("../Output_Result"):
	os.mkdir ("../Output_Result")
result = cv2.VideoWriter(vid_output, cv2.VideoWriter_fourcc(*'MJPG'), 10, (1280, 720))
# used to record the time when we processed last frame

class Lanes():
  def __init__(self):
    self.recent_fit = []
    self.avg_fit = []

def road_lanes(image):
  h, w = image.shape[:2]
  small_img = cv2.resize(image, (160, 80))
  small_img = np.array(small_img)
  small_img = small_img[None, :, :, :]

  prediction = model.predict(small_img)[0] * 255

  lanes.recent_fit.append(prediction)

  if len(lanes.recent_fit) > 5:
    lanes.recent_fit = lanes.recent_fit[1:]

  lanes.avg_fit = np.mean(np.array([i for i in lanes.recent_fit]), axis = 0)


  blanks = np.zeros_like(lanes.avg_fit).astype(np.uint8)
  lane_drawn = np.dstack((blanks, lanes.avg_fit, blanks))

  lane_image = cv2.resize(lane_drawn, (w, h)).astype(np.uint8)
  result = cv2.addWeighted(image, 1, lane_image, 1, 0)
  return result

lanes = Lanes()

def main():
    prev_frame_time = 0
    cap = cv2.VideoCapture(INPUT_FILE)

    while True:
        ret, frame = cap.read()
        if ret:

            # time when we finish processing for this frame
            new_frame_time = time.time()

            if Object_detection:
                bbox, label, conf = yolo.detect_objects(image=frame, confidence=CONFIDENCE_THRESHOLD,
                                                    nms_thresh=nms_thresh, enable_gpu=GPU_enable)
                yolo.draw_bbox(frame, bbox, label, conf, write_conf=write_conf)

            if Lane_detection:
                out_frame = road_lanes(frame)
            else:
                out_frame = frame
            fps = str(int(1 / (new_frame_time - prev_frame_time)))
            prev_frame_time = new_frame_time
            # putting the FPS count on the frame
            cv2.putText(out_frame, fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)
            cv2.imshow('Output', out_frame)
            result.write(out_frame)
            keyboard = cv2.waitKey(1)
            if keyboard == 'q' or keyboard == 27:
                break
        else:
            break

# ---------------------------------------------------
# main condition
if __name__ == "__main__":
    main()
# ---------------------------------------------------
