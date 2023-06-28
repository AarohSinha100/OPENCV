import cv2 as cv
import numpy as np

######### DRAWAING ON IMAGES ###########



#Creating a blank image
blank = np.zeros((500,500,3),dtype='uint8') #datatype o fimage
cv.imshow("Blank",blank)

 #_____1.Paint the image certain color________
blank[:] = 0,255,0
cv.imshow("Green",blank)
blank[:] = 0,0,255
cv.imshow("Red",blank)

#------>Coloring certain portion
blank[200:300, 300:400] = 104,100,25
cv.imshow("ColoredPart",blank)

# ________2.Draw a rectangle
rectangle = cv.rectangle(blank, (50,0), (400,250), (141,155,121),thickness=cv.FILLED) #Thickness = CV.FILLED means not border but the whole image is filled
cv.imshow("Rectangle",rectangle)

#___________3.Draw a circle
circle = cv.circle(blank, (250,250),40,(0,0,255),thickness=-1) #blank, center, radius, color, thckness
cv.imshow("Circle",circle)

#___________4.Draw a line
lines = cv.line(blank, (0,0), (450,450), (100,100,100),thickness=4) #blank, start, end, rgb color
cv.imshow("line",lines)

#___________5. Write text on image
text = cv.putText(blank, "Hello My name is Aaroh", (45,225), cv.FONT_HERSHEY_COMPLEX,1.0, (0,255,0), 2) #blank, coordinates, font, scale of font, color, thickness
cv.imshow("TEXT",text)



#cv.imshow("PROGRAMMER IMAGE RESCALED",programmer_image)
cv.waitKey(0)
