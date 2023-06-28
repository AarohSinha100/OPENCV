import cv2 as cv

# Resizing and rescaling

img = cv.imread("images/programming_big.jpg")

## Rescaling the frame
def rescaleFrame(frame, scale):
    width = int(frame.shape[1]*scale)
    height = int(frame.shape[0]*scale)
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)


capture = cv.VideoCapture("videos/pexels-nino-souza-1536322-1920x1080-30fps.mp4")
def changeRes(width, height, capture):
    # Works for only videos
    capture.set(3, width)
    capture.set(4,height)




#Seeing the resized image
resized_image = rescaleFrame(img, 0.3)
cv.imshow('image_resized', resized_image)



#Seeing the resized video
capture = cv.VideoCapture("videos/pexels-nino-souza-1536322-1920x1080-30fps.mp4")

while True:
    isTrue, frame = capture.read()
    frame_resized = rescaleFrame(frame, 0.74) #This funxtion rescales the video

    #cv.imshow('Video',frame)
    cv.imshow('Rescaled_Video',frame_resized)
    if cv.waitKey(20) & 0xFF==ord('q'):
        break

capture.release()
cv.destroyAllWindows()
