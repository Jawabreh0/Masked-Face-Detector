import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# load the trained model
model = load_model("MobileNetV2_model")

# define the input image size and the confidence threshold
IMG_SIZE = (224, 224)
CONF_THRESH = 0.5

# load the input image and preprocess it
image = cv2.imread("/home/jawabreh/Desktop/MFDD/dataset/test/Masked/3.jpg")
image = cv2.resize(image, IMG_SIZE)
image = img_to_array(image)
image = preprocess_input(image)

# make predictions on the input image using the trained model
predictions = model.predict(np.expand_dims(image, axis=0))[0]

# extract the probabilities for each class
mask_prob = predictions[0]
without_mask_prob = predictions[1]

# determine the predicted class and display the result
if mask_prob > without_mask_prob and mask_prob > CONF_THRESH:
    label = "Mask"
    color = (0, 255, 0)  # green
elif without_mask_prob > mask_prob and without_mask_prob > CONF_THRESH:
    label = "No Mask"
    color = (0, 0, 255)  # red
else:
    label = "Uncertain"
    color = (255, 255, 0)  # yellow

# draw the label and bounding box on the input image
cv2.putText(image, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
cv2.imshow("Output", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
