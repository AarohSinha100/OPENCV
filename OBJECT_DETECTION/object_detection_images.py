import cv2
import tensorflow_hub as hub
import cv2
import numpy
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


print("TensorFlow version:", tf.__version__)


# Loading the image and preprocessing it to expected format
# for the tensorflow model

width = 1028
height = 1028

# Load image using opencv
img = cv2.imread("pexels-emre-akyol-17255476.jpg")

# Resize the image to specific width and height
resized = cv2.resize(img,(width, height))

# Convert image to RGB
rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
# cv.imshow("test_image", rgb)
# cv.imshow("bgr", resized)

cv2.waitKey(0)

# Converting it into uint8
rgb_tensor = tf.convert_to_tensor(rgb, dtype='uint8')
# Add dims
rgb_tensor = tf.expand_dims(rgb_tensor, axis=0)
# print(rgb_tensor)


# Loading the csv file labels
labels = pd.read_csv("labels.csv", sep=';', index_col='ID')
labels = labels['OBJECT (2017 REL.)']
# print(labels)

print("CHECK 1")

# Loading the model and labels
# Loading the model directly from tensorflow hub
detector = hub.load("https://tfhub.dev/tensorflow/efficientdet/lite2/detection/1")

print("CHECK 2")

# Creating predictions
boxes, scores, classes, num_detections = detector(rgb_tensor)

# print(boxes)
# print(classes)
# print(scores)
# print(num_detections)
print("CHECK 3")

# Processing outputs
print(classes)
prediction_labels = classes.numpy().astype('int')[0]
predictions_labels = [labels[i] for i in prediction_labels]
prediction_boxes = boxes.numpy()[0].astype('int')
prediction_scores = scores.numpy()[0]

# print(prediction_labels)

# Putting the boxes and labels on the image
for score, (ymin,xmin,ymax,xmax), label in zip(prediction_scores, prediction_boxes, prediction_labels):
    if score < 0.5:
        continue

    score_txt = f'{100 * round(score)}%'
    img_boxes = cv2.rectangle(rgb,(xmin, ymax),(xmax, ymin),(0,255,0),2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img_boxes, str(labels[label]),(xmin, ymax-10), font, 1.5, (255,0,0), 2, cv2.LINE_AA)
    cv2.putText(img_boxes,score_txt,(xmax, ymax-10), font, 1.5, (0,0,255), 2, cv2.LINE_AA)
    print(labels[label])

    plt.imshow(img_boxes)
    plt.show()
