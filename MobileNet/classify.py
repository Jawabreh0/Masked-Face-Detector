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

# initialize counters for TP, TN, FP, FN
TP = 0
TN = 0
FP = 0
FN = 0

# loop over the test images
for i in range(1, 2000):
    # load the input image and preprocess it
    image_path = f"/home/jawabreh/Desktop/MFDD/dataset/test/Masked/{i}.jpg"
    image = cv2.imread(image_path)
    image = cv2.resize(image, IMG_SIZE)
    image = img_to_array(image)
    image = preprocess_input(image)

    # make predictions on the input image using the trained model
    predictions = model.predict(np.expand_dims(image, axis=0))[0]

    # extract the probabilities for each class
    mask_prob = predictions[0]
    without_mask_prob = predictions[1]

    # determine the predicted class
    if mask_prob > without_mask_prob and mask_prob > CONF_THRESH:
        pred_label = "Mask"
    elif without_mask_prob > mask_prob and without_mask_prob > CONF_THRESH:
        pred_label = "No Mask"
    else:
        pred_label = "Uncertain"

    # extract the ground truth label from the image path
    true_label = "Masked" if "Masked" in image_path else "Unmasked"

    # update the counters for TP, TN, FP, FN
    if pred_label == "Mask" and true_label == "Masked":
        TP += 1
    elif pred_label == "No Mask" and true_label == "Unmasked":
        TN += 1
    elif pred_label == "Mask" and true_label == "Unmasked":
        FP += 1
    else:
        FN += 1

# compute accuracy, precision, and f1 score
accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1_score = 2 * precision * recall / (precision + recall)

# print the results
print(f"TP: {TP}")
print(f"TN: {TN}")
print(f"FP: {FP}")
print(f"FN: {FN}")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1_score}")
