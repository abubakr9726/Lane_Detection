import numpy as np
import cv2
from cvlib.object_detection import YOLO
from PIL import Image
from moviepy.editor import VideoFileClip
from tensorflow import keras
model = keras.models.load_model('model.h5')

INPUT_FILE = "test_videos/challenge.mp4";
LABELS_FILE = "coco.names";
CONFIG_FILE = "yolov4-tiny.cfg";
WEIGHTS_FILE = "yolov4-tiny.weights";
CONFIDENCE_THRESHOLD = 0.35
factor = 40
yolo = YOLO(WEIGHTS_FILE, CONFIG_FILE, LABELS_FILE)

result = cv2.VideoWriter('filename.avi',
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, (1280, 720))

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


vid_output = 'lanes_output_clip.mp4'

lanes = Lanes()





def main():

    cap = cv2.VideoCapture(INPUT_FILE)

    while True:
        ret, frame = cap.read()
        if ret:


            bbox, label, conf = yolo.detect_objects(frame)

            img1 = yolo.draw_bbox(frame, bbox, label, conf, write_conf=True)

            out_frame = road_lanes(frame)
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




























































































# using System.Collections.Generic;
# using UnityEngine;
# using OpenCVForUnity.CoreModule;
# using OpenCVForUnity.ImgprocModule;
# using OpenCVForUnity.UnityUtils;
# using OpenCVForUnity.DnnModule;
# using System.Linq;
#
# public class Lanes
# {
#     public List<Mat> recent_fit = new List<Mat>();
#     public Mat avg_fit = new Mat();
# }
#
# public class RoadLanes : MonoBehaviour
# {
#     public string modelPath;
#     private Net model;
#     private Lanes lanes;
#
#     // Start is called before the first frame update
#     void Start()
#     {
#         lanes = new Lanes();
#         model = Dnn.readNetFromTensorflow(modelPath);
#     }
#
#     // Update is called once per frame
#     void Update()
#     {
#         Mat frame = new Mat();
#         Mat outFrame = new Mat();
#
#         if (WebCamTextureToMatHelper.IsInitialized())
#         {
#             Mat rgbaMat = webCamTextureToMatHelper.GetMat();
#             Imgproc.cvtColor(rgbaMat, frame, Imgproc.COLOR_RGBA2RGB);
#             outFrame = roadLanes(frame);
#             Imgproc.cvtColor(outFrame, outFrame, Imgproc.COLOR_RGB2RGBA);
#             Utils.fastMatToTexture2D(outFrame, texture);
#         }
#     }
#
#     private Mat roadLanes(Mat image)
#     {
#         Mat smallImg = new Mat();
#         Imgproc.resize(image, smallImg, new Size(160, 80));
#         smallImg.convertTo(smallImg, CvType.CV_32F);
#         smallImg /= 255;
#
#         List<Mat> channels = new List<Mat>();
#         Core.split(smallImg, channels);
#         Mat inputBlob = Dnn.blobFromImages(channels);
#
#         model.setInput(inputBlob);
#         Mat prediction = model.forward();
#         Core.transpose(prediction, prediction);
#         prediction = prediction.reshape(0, 80);
#
#         Core.multiply(prediction, new Scalar(255), prediction);
#         prediction.convertTo(prediction, CvType.CV_8UC1);
#
#         lanes.recent_fit.Add(prediction);
#
#         if (lanes.recent_fit.Count > 5)
#         {
#             lanes.recent_fit.RemoveAt(0);
#         }
#
#         Mat[] recent_fit_array = lanes.recent_fit.ToArray();
#         Core.hconcat(recent_fit_array, lanes.avg_fit);
#
#         Core.mean(lanes.avg_fit, lanes.avg_fit);
#
#         List<Mat> blanks = Enumerable.Repeat(Mat.zeros(lanes.avg_fit.size(), lanes.avg_fit.type()), 2).ToList();
#         List<Mat> lane_drawn_channels = new List<Mat>() { blanks[0], lanes.avg_fit, blanks[1] };
#         Mat lane_drawn = new Mat();
#         Core.merge(lane_drawn_channels, lane_drawn);
#
#         Imgproc.resize(lane_drawn, lane_drawn, new Size(1280, 720));
#         lane_drawn.convertTo(lane_drawn, CvType.CV_8UC3);
#
#         Mat result = new Mat();
#         Core.addWeighted(image, 1, lane_drawn, 1, 0, result);
#
#         return result;
#     }
# }
