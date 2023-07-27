import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

print("CHECK 1")

# Getting the model and labels
detector = hub.load("https://tfhub.dev/tensorflow/efficientdet/lite2/detection/1")
labels = pd.read_csv("labels.csv", sep=';', index_col='ID')
labels = labels['OBJECT (2017 REL.)']

capture = cv2.VideoCapture(0)

width = 512
height = 512

while True:
    # Capture frame by frame
    ret, frame = capture.read()

    frame_ = cv2.resize(frame, (width, height))
    frame_rgb = cv2.cvtColor(frame_, cv2.COLOR_BGR2RGB)
    rgb_tensor = tf.convert_to_tensor(frame_rgb, dtype='uint8')
    rgb_tensor_exp = tf.expand_dims(rgb_tensor, axis=0)

    boxes, scores, classes, num_detection = detector(rgb_tensor_exp)

    prediction_labels = classes.numpy().astype('int')[0]
    predictions_labels = [labels[i] for i in prediction_labels]
    prediction_boxes = boxes.numpy()[0].astype('int')
    prediction_scores = scores.numpy()[0]

    # loop throughout the detection and place a box around it
    for score, (ymin, xmin, ymax, xmax), label in zip(prediction_scores, prediction_boxes, prediction_labels):
        if score < 0.5:
            continue

        score_txt = f'{100 * round(score)}%'

        # Create a copy of the frame_rgb to draw on
        img_boxes = frame_rgb.copy()

        # Draw the bounding box
        cv2.rectangle(img_boxes, (xmin, ymax), (xmax, ymin), (0, 255, 0), 2)

        # Draw the label and score text
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img_boxes, str(labels[label]), (xmin, ymax - 10), font, 1.5, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img_boxes, score_txt, (xmax, ymax - 10), font, 1.5, (0, 0, 255), 2, cv2.LINE_AA)

        print(labels[label])

        # Display the result frame
        cv2.imshow("Video", img_boxes)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release when done
capture.release()
cv2.destroyAllWindows()
