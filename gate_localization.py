import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import time


def calculate_euler_angles(A, B, C, D, P):

    M = (A + B + C + D) / 4

    # Calculate vector from P to M
    delta_x, delta_y = M - P
    #delta_x, delta_y, delta_z  = M - P

    # Calculate Yaw angle
    yaw_angle = np.degrees(np.arctan2(delta_y, delta_x))

    # Calculate Pitch angle
    #pitch_angle = np.degrees(np.arctan2(delta_z, np.sqrt(delta_x ** 2 + delta_y ** 2)))
    pitch_angle =0
    # Calculate the normal vector of the square's plane
    AB = B - A
    AC = C - A
    normal_vector = np.cross(AB, AC)
    normal_vector = normal_vector / np.linalg.norm(normal_vector)  # Normalize

    # Assume drone's right vector is along the X-axis (for simplicity)
    drone_right_vector = np.array([1, 0, 0])

    # Calculate Roll angle
    roll_angle = np.degrees(np.arccos(np.dot(drone_right_vector, normal_vector)))

    return yaw_angle, pitch_angle, roll_angle, M

def roll_correct(a,_roll):
    a[0] = a[0]*np.cos(_roll)
    a[1] = a[1]*np.sin(_roll)
    return a

def get_distance(points, intrinsic):
    corrected_points = [roll_correct(point, roll) for point in points]
    p1, p2, p3, p4 = sorted(points, key=lambda x: x[1])

    yaw, pitch, roll, center = calculate_euler_angles(p1,p2,p3,p4)

    l1,l2,_ = intrinsic
    f_x,_,c_x = l1
    _,f_y,c_y = l2

    #correcting for roll
    f_x_corr = f_x *np.cos(roll)+f_y*np.sin(roll)
    f_y_corr = -f_y *np.sin(roll)+f_x*np.cos(roll)
    c_x_corr = c_x *np.cos(roll)+c_y*np.sin(roll)
    c_y_corr = -c_y *np.sin(roll)+c_x*np.sin(roll)

    x,y = roll_correct(center,roll)

    [x1, y1], [x2, y2] = [p1, p2].sort(key=lambda x: x[0])
    [x3, y3], [x4, y4] = [p3, p4].sort(key=lambda x: x[0])

    #Point orientation is:
    #p1 p2
    #p3 p4

    #get pixel height in midpoint
    y_mid = min(y1,y2) + abs(y1-y2)/2 - (min(y3,y4) + abs(y3-y4)/2)
    #corret for pitch
    y_mid_cor = y_mid/np.cos(pitch)
    dist = f_y_corr*150/y_mid_cor
    #left right
    X = (x-c_x_corr)/f_x_corr
    #forward
    Y = np.sqrt(dist**2-X**2)


    return dist,(X,Y),yaw


def bbox_area(bbox):
    x1, y1, x2, y2 = bbox
    return abs((x2 - x1) * (y2 - y1))
def undistort(image):
    mtx = np.loadtxt('./cam_mat.txt')
    dist = np.loadtxt('./dist_coeffs.txt')
    h,  w = image.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    return cv2.undistort(image, mtx, dist, None, newcameramtx)

def contourer(path_to_im,predictions):

    #image = undistort(cv2.imread(path_to_im))

    image = cv2.imread(path_to_im)
    t1 = time.time()
    pred = predictions[0]
    bboxes =pred.boxes.xyxy.tolist()
    if not bboxes:
        print("No bounding boxes found in the prediction.")
        return False, False
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
    try:
        border_pixels = np.concatenate([top_border.transpose(1,0,2), bottom_border.transpose(1,0,2), left_border, right_border])
    except ValueError as e:
        print(e)
        return False, False
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
            x, y, w, h = cv2.boundingRect(approx)
            if w > 0.99*width and h > 0.99*height:
                print("Green outline is the size of the image.")
                return False, False
            # Draw the quadrilateral on a copy of the original image
            output_image = cv2.cvtColor(dilate, cv2.COLOR_GRAY2BGR)  # Convert to BGR for color drawing
            cv2.drawContours(output_image, [approx], 0, (0, 255, 0), 2)  # Draw in green
            print((time.time() - t1)*1000)
            # Display the quadrilateral and the region of interest
            #cv2.imwrite(f'output/filename_{file.name[-7:]}', output_image)
            cv2.imwrite(f'./res/output/{path_to_im.name[:-3]}_processed.jpg', output_image)
            corners = approx.reshape(4, 2)  # Reshape to get each corner as (x, y) pair
            print("Corner coordinates:")
            for i, corner in enumerate(corners):
                print(f"Corner {i + 1}: {corner}")

            return corners, output_image




model = YOLO('best.pt')

for file in Path.cwd().joinpath("./res").iterdir():
    if file.suffix == '.jpg':
        if file is None:
            continue
        res = contourer(file, model.predict(file))
        if res is not None:
            output, img = res
            if output is not False:
                 get_distance(output, )
        #         jpg = cv2.imread(file)
        #         if img.shape != jpg.shape:
        #             jpg = cv2.resize(jpg, (img.shape[1], img.shape[0]))
        #
        #         # Concatenate the images horizontally
        #         combined_image = cv2.hconcat([img, jpg])
        #
        #         # Display the concatenated image
        #         cv2.imshow("Combined Image", combined_image)
        #
        #         cv2.waitKey(0)
        #         cv2.destroyAllWindows()
exit
#1, 255,241,265,110