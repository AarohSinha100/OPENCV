import cv2 as cv
import numpy as np

######################################################################
######-----------MASKING-------------#################################
######################################################################

import cv2 as cv
import numpy as np

img = cv.imread("CAT_IMG.jpg")

#Rescaling the image
def rescaleFrame(frame, scale):
    width = int(img.shape[1]*scale)
    height = int(img.shape[0]*scale)

    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation = cv.INTER_AREA)

resized_image = rescaleFrame(img, 0.12)
cv.imshow("Resized Image",resized_image)

blank = np.zeros(resized_image.shape[:2],dtype='uint8') #The dimensions of the mask has to be of the same size as that of the image
cv.imshow("Blank Image",blank)

#What we wanna do is to create a circle on this blank image and treat it as a mask

mask = cv.circle(blank.copy(), (resized_image.shape[1]//2, int(resized_image.shape[0]//2)-200),140,255,-1) #Note - blank.copy() allows us to use the same blank image as many times as we want without creating new ones
cv.imshow("Mask",mask)

masked_image = cv.bitwise_and(resized_image,resized_image, mask=mask)
cv.imshow('Masked Image',masked_image)


cv.waitKey(0)
'''

'''
##########MASKINGGGGGGGGGG##############

import cv2 as cv
import numpy as np
#import matplotlib.pyplot as plt

img = cv.imread("CAT_IMG.jpg")

#Rescaling the image
def rescaleFrame(frame, scale):
    width = int(img.shape[1]*scale)
    height = int(img.shape[0]*scale)

    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation = cv.INTER_AREA)

cat = rescaleFrame(img, 0.12)

blank = np.zeros(cat.shape[:2],dtype='uint8') #The dimensions of the mask has to be of the same size as that of the image
circle = cv.circle(blank.copy(), (cat.shape[1]//2, int(cat.shape[0]//2)-200),140,255,-1) #Note - blank.copy() allows us to use the same blank image as many times as we want without creating new ones



#Let's convert the image as gray
gray_cat = cv.cvtColor(cat, cv.COLOR_BGR2GRAY)

masked_img = cv.bitwise_and(cat,gray_cat,mask=circle)
cv.imshow("masked",masked_img)

#Grayscale histogram - will show the pixel intensiies in the image
#The image is a list so we wanna pass the list
gray_hist = cv.calcHist([masked_img], [0], None, [256],[0,256]) #(which image from list, 0 for grayscale, mask if we want, histsize, range for possible pizel values)
#Lets plot this on matplotlib
'''
plt.figure()
plt.title('Grayscale Histogram')
plt.xlabel('Bins')
plt.ylabel('Number of pixels')
plt.plot(gray_hist)
plt.xlim([0,256])
plt.show()
'''
cv.imshow("Image",gray_cat)
cv.waitKey(0)