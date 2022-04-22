import cv2
import numpy as np
import skimage.exposure

# usage
# put this script and the image face.jpg in the same directory /dir
# run these 2 commands inside bash
# cd /dir
# python change_skin_v1.py

# script_name= change_skin_v1.py


# you can change  the 3 parameters: alpha, skincolor_low, skincolor_high


# path file
path_face = "./face.jpg"
result_partial = "./result_partial.png"
result_final = "./result_partial.png"

# blending parameter
alpha = 0.7

# Define lower and uppper limits of what we call "skin color"
skincolor_low = np.array([0, 10, 60])
skincolor_high = np.array([180, 150, 255])

# specify desired bgr color (brown) for the new face.
# this value is approximated
desired_color_brg = (255, 0, 0)

# read face
img_main_face = cv2.imread(path_face)

# face.jpg has by default the BGR format, convert BGR to HSV
hsv = cv2.cvtColor(img_main_face, cv2.COLOR_BGR2HSV)

# create the HSV mask
mask = cv2.inRange(hsv, skincolor_low, skincolor_high)

# Change image to brown where we found pink
img_main_face[mask > 0] = desired_color_brg
cv2.imwrite(result_partial, img_main_face)

# blending block start

# alpha range for blending is  0-1


# load images for blending
src1 = cv2.imread(result_partial)
src2 = cv2.imread(path_face)

if src1 is None:
    print("Error loading src1")
    exit(-1)
elif src2 is None:
    print("Error loading src2")
    exit(-1)

# actually  blend_images
result_final = cv2.addWeighted(src1, alpha, src2, 1 - alpha, 0.0)
cv2.imwrite('./result_final.png', result_final)
