import cv2
import os
import numpy as np

IMAGES_DIR = 'Playground/data/images'
EXAMPLES_DIR = IMAGES_DIR + '/examples'
OUTPUT_DIR = 'Playground/output/basic_color_segmentation'


def segment(img_name):   
    img = cv2.imread(IMAGES_DIR + '/examples/' + img_name)

    if img is None:
        print(f"Skipping {img_name} — could not load as image")
        return
    
    img_thresholded = cv2.inRange(img, (60, 60, 60), (140, 140, 140))

    os.makedirs(OUTPUT_DIR + '/thresholded', exist_ok=True)
    cv2.imwrite(OUTPUT_DIR + '/thresholded/' + img_name, img_thresholded)

    kernel = np.ones((10,10),np.uint8)
    img_opening = cv2.morphologyEx(img_thresholded, cv2.MORPH_OPEN, kernel)

    os.makedirs(OUTPUT_DIR + '/kerneled', exist_ok=True)
    cv2.imwrite(OUTPUT_DIR + '/kerneled/' + img_name, img_opening)

    contours, hierarchy = cv2.findContours(img_opening.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    print (len(contours))

    for contour in contours:
        (x,y),radius = cv2.minEnclosingCircle(contour)
        center = (int(x),int(y))
        radius = int(radius)
        cv2.circle(img,center,radius,(0,255,0),2)

    os.makedirs(OUTPUT_DIR + '/labelled', exist_ok=True)
    cv2.imwrite(OUTPUT_DIR + '/labelled/' + img_name, img)


for img_name in os.listdir(EXAMPLES_DIR):
    segment(img_name) 
