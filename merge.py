import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO


def bbox_area(bbox):
    x1, y1, x2, y2 = bbox
    return abs((x2 - x1) * (y2 - y1))
def undistort(image):
    mtx = np.loadtxt('./cam_mat.txt')
    dist = np.loadtxt('./dist_coeffs.txt')
    h,  w = image.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    return cv2.undistort(image, mtx, dist, None, newcameramtx)

def calculate_euler_angles(A, B, C, D, P):
    # Calculate the center of the square (M) as the average of the four corners
    M = (A + B + C + D) / 4

    # Calculate vector from P to M
    delta_x, delta_y, delta_z = M - P

    # Calculate Yaw angle
    yaw_angle = np.degrees(np.arctan2(delta_y, delta_x))

    # Calculate Pitch angle
    pitch_angle = np.degrees(np.arctan2(delta_z, np.sqrt(delta_x**2 + delta_y**2)))

    # Calculate the normal vector of the square's plane
    AB = B - A
    AC = C - A
    normal_vector = np.cross(AB, AC)
    normal_vector = normal_vector / np.linalg.norm(normal_vector)  # Normalize

    # Assume drone's right vector is along the X-axis (for simplicity)
    drone_right_vector = np.array([1, 0, 0])

    # Calculate Roll angle
    roll_angle = np.degrees(np.arccos(np.dot(drone_right_vector, normal_vector)))

    return yaw_angle, pitch_angle, roll_angle

def contourer(path_to_im,predictions):

    #image = undistort(cv2.imread(path_to_im))

    image = cv2.imread(path_to_im)

    pred = predictions[0]
    bboxes =pred.boxes.xyxy.tolist()
    if not bboxes:
        print("No bounding boxes found in the prediction.")
        return
    bboxes_t =[]
    for bbox in bboxes:
        bbox = list(map(int,bbox))
        bboxes_t.append(bbox)
    bboxes=bboxes_t

    largest_bbox = max(bboxes, key=bbox_area)
    x1, y1, x2, y2 = largest_bbox

    # Ensure coordinates are within image bounds
    height, width = image.shape[:2]
    x1 = max(0, min(x1, width - 1))
    x2 = max(0, min(x2, width - 1))
    y1 = max(0, min(y1, height - 1))
    y2 = max(0, min(y2, height - 1))

    # Create a mask the same size as the image, initialized to the average value of the border
    top_border = image[y1-2:y1-1, x1:x2+1]       # Top edge
    bottom_border = image[y2+1:y2+2, x1:x2+1]    # Bottom edge
    left_border = image[y1+1:y2, x1:x1+1]      # Left edge, excluding corners
    right_border = image[y1+1:y2, x2-1:x2]     # Right edge, excluding corners
    border_pixels = np.concatenate([top_border.transpose(1,0,2), bottom_border.transpose(1,0,2), left_border, right_border])
    avg_colr = [np.mean(border_pixels[:, :, channel]) for channel in range(3)]
    mask = np.stack([np.full((image.shape[0], image.shape[1]), avg_colr[channel]) for channel in range(3)], axis=-1).astype(np.uint8)

    # Copy the region defined by the largest bounding box onto the mask
    mask[y1:y2, x1:x2] = image[y1:y2, x1:x2]

    box_width = x1 - x2
    hei_height = y1 - y2

    #gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(mask, (13,13), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    lower_purple = np.array([120, 50, 50])  # Adjusted lower bound
    upper_purple = np.array([160, 255, 255])  # Adjusted upper bound
    mask = cv2.inRange(hsv, lower_purple, upper_purple)
    thresh = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Two pass dilate with horizontal and vertical kernel
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5))
    dilate = cv2.dilate(thresh, horizontal_kernel, iterations=2)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 9))
    dilate = cv2.dilate(dilate, vertical_kernel, iterations=2)
    dilate = cv2.bitwise_not(dilate)

    contours, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop through the contours and find the one that resembles the white spot
    for cnt in contours:

        # Approximate the contour to a polygon
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # Check if the approximated contour has 4 points (quadrilateral)
        if len(approx) == 4:
            # Draw the quadrilateral on a copy of the original image
            output_image = cv2.cvtColor(dilate, cv2.COLOR_GRAY2BGR)  # Convert to BGR for color drawing
            cv2.drawContours(output_image, [approx], 0, (0, 255, 0), 2)  # Draw in green

            # Display the quadrilateral and the region of interest
            cv2.imwrite(f'output/filename_{file.name[-7:]}', output_image)
            corners = approx.reshape(4, 2)  # Reshape to get each corner as (x, y) pair
            print("Corner coordinates:")
            for i, corner in enumerate(corners):
                print(f"Corner {i + 1}: {corner}")
            return corners




model = YOLO('best.pt')
jgps = [
    "./res/captured_image_76.jpg",
    "./res/captured_image_71.jpg",
    "./res/captured_image_79.jpg"
]
for file in jgps:
    file= Path(file)
    if file.suffix == '.jpg':
        print(file)
        output = contourer(file,model.predict(file))
exit