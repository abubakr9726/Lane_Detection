
import numpy as np
import cv2




def grayscale(img):
    # return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image




def draw_lines(img, lines, color=[0, 0, 255], thickness=7):
    # list to get positives and negatives values

    x_bottom_pos = []
    x_upperr_pos = []
    x_bottom_neg = []
    x_upperr_neg = []

    y_bottom = 540
    y_upperr = 315

    # y1 = slope*x1 + b
    # b = y1 - slope*x1
    # y = slope*x + b
    # x = (y - b)/slope

    slope = 0
    b = 0

    # get x upper and bottom to lines with slope positive and negative
    for line in lines:
        for x1, y1, x2, y2 in line:
            # test and filter values to slope
            if ((y2 - y1) / (x2 - x1)) > 0.5 and ((y2 - y1) / (x2 - x1)) < 0.8:

                slope = ((y2 - y1) / (x2 - x1))
                b = y1 - slope * x1

                x_bottom_pos.append((y_bottom - b) / slope)
                x_upperr_pos.append((y_upperr - b) / slope)

            elif ((y2 - y1) / (x2 - x1)) < -0.5 and ((y2 - y1) / (x2 - x1)) > -0.8:

                slope = ((y2 - y1) / (x2 - x1))
                b = y1 - slope * x1

                x_bottom_neg.append((y_bottom - b) / slope)
                x_upperr_neg.append((y_upperr - b) / slope)

    # creating a new 2d array with means
    lines_mean = np.array(
        [[int(np.mean(x_bottom_pos)), int(np.mean(y_bottom)), int(np.mean(x_upperr_pos)), int(np.mean(y_upperr))],
         [int(np.mean(x_bottom_neg)), int(np.mean(y_bottom)), int(np.mean(x_upperr_neg)), int(np.mean(y_upperr))]])

    # Drawing the lines
    for i in range(len(lines_mean)):
        cv2.line(img, (lines_mean[i, 0], lines_mean[i, 1]), (lines_mean[i, 2], lines_mean[i, 3]), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img


# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

def process_image(image,
                  kernel_size=5,
                  low_threshold=100, high_threshold=250,
                  rho=1, theta=np.pi / 180, threshold=30,
                  min_line_len=100, max_line_gap=200):
    # turn in grayscale
    gray = grayscale(image)

    # apply blur
    blur_gray = gaussian_blur(gray, kernel_size)

    # finding edges
    edges = canny(image, low_threshold, high_threshold)

    # setting vertices
    vertices = np.array([[(0, image.shape[0]), (450, 315), (490, 315),
                          (image.shape[1], image.shape[0])]], dtype=np.int32)

    # creating a region of interest
    masked_edges = region_of_interest(edges, vertices)

    # finding lines
    lines = hough_lines(masked_edges, rho, theta, threshold, min_line_len, max_line_gap)

    # merge with original image
    lines_edges = weighted_img(lines, image, α=0.8, β=1., λ=0.)

    return lines_edges


def process_challenge(image,
                      kernel_size=7,
                      low_threshold=150, high_threshold=250,
                      rho=1, theta=np.pi / 90, threshold=15,
                      min_line_len=40, max_line_gap=150):
    # turn in grayscale
    gray = grayscale(image)

    # apply blur
    blur_gray = gaussian_blur(gray, kernel_size)

    # finding edges
    edges = canny(image, low_threshold, high_threshold)

    # setting vertices with a shorter range
    vertices = np.array([[(100, image.shape[0]), (470, 325), (490, 325),
                          (image.shape[1] - 100, image.shape[0])]], dtype=np.int32)

    # creating a region of interest
    masked_edges = region_of_interest(edges, vertices)

    # finding lines
    lines = hough_lines(masked_edges, rho, theta, threshold, min_line_len, max_line_gap)

    # merge with original image
    lines_edges = weighted_img(lines, image, α=0.8, β=1., λ=0.)

    return lines_edges


# def draw_lines(img, lines, color=[0, 0, 255], thickness=7):
#     # list to get positives and negatives values
#
#     x_bottom_pos = []
#     x_upperr_pos = []
#     x_bottom_neg = []
#     x_upperr_neg = []
#
#     y_bottom = 720
#     y_upperr = 450
#
#     # y1 = slope*x1 + b
#     # b = y1 - slope*x1
#     # y = slope*x + b
#     # x = (y - b)/slope
#
#     slope = 0
#     b = 0
#
#     # get x upper and bottom to lines with slope positive and negative
#     for line in lines:
#         for x1, y1, x2, y2 in line:
#             # test and filter values to slope
#             if ((y2 - y1) / (x2 - x1)) > 0.5 and ((y2 - y1) / (x2 - x1)) < 0.8:
#
#                 slope = ((y2 - y1) / (x2 - x1))
#                 b = y1 - slope * x1
#
#                 x_bottom_pos.append((y_bottom - b) / slope)
#                 x_upperr_pos.append((y_upperr - b) / slope)
#
#             elif ((y2 - y1) / (x2 - x1)) < -0.5 and ((y2 - y1) / (x2 - x1)) > -0.8:
#
#                 slope = ((y2 - y1) / (x2 - x1))
#                 b = y1 - slope * x1
#
#                 x_bottom_neg.append((y_bottom - b) / slope)
#                 x_upperr_neg.append((y_upperr - b) / slope)
#
#     # creating a new 2d array with means
#     lines_mean = np.array(
#         [[int(np.mean(x_bottom_pos)), int(np.mean(y_bottom)), int(np.mean(x_upperr_pos)), int(np.mean(y_upperr))],
#          [int(np.mean(x_bottom_neg)), int(np.mean(y_bottom)), int(np.mean(x_upperr_neg)), int(np.mean(y_upperr))]])
#
#     # Drawing the lines
#     for i in range(len(lines_mean)):
#         cv2.line(img, (lines_mean[i, 0], lines_mean[i, 1]), (lines_mean[i, 2], lines_mean[i, 3]), color, thickness)





file_path = "test_videos/solidYellowLeft.mp4"

cap = cv2.VideoCapture(file_path)

while True:
    ret, frame = cap.read()
    if ret:
        out_frame = process_image(frame)

        cv2.imshow('Output', out_frame)

        keyboard = cv2.waitKey(10)
        if keyboard == 'q' or keyboard == 27:
            break
    else:
        break