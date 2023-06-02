import cv2 as cv2
import numpy as np

#########################################################################
#
#########------CONTOURS AND SHAPE DETECTION----------####################
#
#########################################################################
def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver


def getContours(img):
    objectType = ""
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours: #as contours return a list of contour. For each contour we are going to find the area first
        area = cv2.contourArea(cnt)
        print(area)
        # CHECK FOR THE MINIMUM AREA TO GIVE IT FOR TH THRESHOLD
        if area > 500:
            #Now let's draw the contours so we can see them clearly
            cv2.drawContours(shapesContour, cnt, -1, (255,100,100),3) #where to draw, the shape, -1 for full drawing, color channel

            #Let's calculate the cirve length - will help us to approximate the corner of our shapes
            peri = cv2.arcLength(cnt, True)
            print(peri)
            approx = cv2.approxPolyDP(cnt, 0.02*peri,True)

            print(approx) #This will give us the corner point of each of the shapes
            print(len(approx)) #This gives us idea of what the shape is
            # 3-triangle, 4-rectangle, >4 - circle

            object_corners = len(approx)

            if object_corners == 3:
                objectType = "Triangle"
            elif object_corners==4:
                # #If in certain rangle then square else rectangle
                # aspRatio = width/float(height)
                # if aspRatio > 0.95 and aspRatio < 1.05:
                #     objectType=="Square"
                # else:
                #     objectType="Rectangle"
                objectType="RECTANGLE"
            elif object_corners>4:
                objectType=="circle"

            #Now create a bounded box around the object corners
            x, y, width, height = cv2.boundingRect(approx) #Gives us x and y of th eshapes
            cv2.rectangle(shapesContour,(x,y),(x+width, y+height),(0,255,0),2) #Now we have bounding boxes across each of the shapes we are detecting
            cv2.putText(shapesContour, objectType,
                        (x+(width//2)-10,y+(height//2)-10), #where to print it (center)
                        cv2.FONT_HERSHEY_COMPLEX,
                        0.5,#scale
                        (0,255,255),2) #color and font scale







path = "shapes.png"
shapes = cv2.imread(path)

shapesContour = shapes.copy() #copy to put contour drawing from th eoriginal image

# Presprocessing image - converting to grayscale and adding bit of blur
shapes_gray = cv2.cvtColor(shapes, cv2.COLOR_BGR2GRAY)
shapes_blur = cv2.GaussianBlur(shapes_gray,(7,7),1)
shapes_blank = np.zeros_like(shapes)

###################-----FINDING EDGES BY CANNY-----------##################
shapes_canny = cv2.Canny(shapes_blur,50,50)

getContours(shapes_canny)

shapes_stacked = stackImages(0.8,([shapes,shapes_gray,shapes_blur],
                                  [shapes_canny,shapesContour,shapes_blank]))




cv2.imshow("Stacked image",shapes_stacked)


# cv.imshow("Shapes Image",shapes)
cv2.waitKey(0)