import cv2 as cv

img = cv.imread("images/programming_big.jpg")

def rescaleFrame(frame, scale):
    width = int(frame.shape[1]*scale)
    height = int(frame.shape[0]*scale)

    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation = cv.INTER_AREA)



resized_image = rescaleFrame(img, 0.2)

#______1.GrayScale Images
gray = cv.cvtColor(resized_image, cv.COLOR_BGR2GRAY)


#______2.Blur
blur = cv.GaussianBlur(resized_image, (7,7), cv.BORDER_DEFAULT)
# cv.imshow('Blur',blur)

#______3.Edge Cascade
canny = cv.Canny(resized_image, 125,175)


#______4.Dialating the image
dialated = cv.dilate(canny, (3,3), iterations=10)

#_______5.Eroding
eroded = cv.erode(dialated, (3,3), iterations=1)


# cv.imshow("image", canny)



#____6.Resize
resized = cv.resize(img, (100,100),interpolation=cv.INTER_AREA)


#____7.Cropping
cropped = resized_image[50:200, 200:400]


cv.imshow("cropped",cropped)
cv.waitKey(0)
