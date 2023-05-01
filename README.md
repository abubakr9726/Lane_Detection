# Lane Detection
![](/Readme Data/icon.png)


**Description**

1. This is a Python script for a computer vision application that performs object detection and lane detection on a video stream. It uses the cvlib library for object detection and      TensorFlow Keras for lane detection.

2. The script first loads configuration settings from a separate Configuration file. It then creates a YOLO object detector using the weights and configuration files specified in the configuration settings. It also loads a lane detection model using the path specified in the configuration settings.

3. The script then defines a class called "Lanes" that keeps track of recent lane detection results and calculates an average result. It defines a "road_lanes" function that takes an image as input and uses the lane detection model to predict the lane positions in the image. It then draws the detected lanes onto the image and returns the result.

4. The main loop of the script reads frames from the video stream using OpenCV. For each frame, it performs object detection and lane detection if specified in the configuration settings. It then draws the detected objects and lanes onto the frame and displays the result. It also records the result to a video file specified in the configuration settings.

5. The script uses the cv2.VideoWriter class to create a video writer object for recording the result to a file. It also uses OpenCV to display the output frame by frame.

6. Finally, the script checks for a key press to terminate the application and exits the main loop if the "q" key or the "Esc" key is pressed.
