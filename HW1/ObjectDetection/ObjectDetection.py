# import libraries
import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox

print(cv2.__version__)

image_path = 'tmp.JPG'  
im = cv2.imread(image_path) 


# object detection
bbox, label, conf = cv.detect_common_objects(im)

print(bbox, label, conf)

im = draw_bbox(im, bbox, label, conf) 


cv2.imwrite('result.jpg', im) 
