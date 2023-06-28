import cv2 as cv

#REAL IMAGE
image = cv.imread("programming_big.jpg")

#FUNCTION TO RESCALE THE IMAGE
def rescale_frame(frame, scale):
    width = int(frame.shape[1]*scale)
    height = int(frame.shape[0]*scale)

    dimensions = (width, height)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

#RESCALED IMAGE
rescaled = rescale_frame(image, 0.1)

#SHOW BOTH IMAGES
cv.imshow("real_image",image)
cv.imshow("rescaled_image",rescaled)
cv.waitKey(0)
