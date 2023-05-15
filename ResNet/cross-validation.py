import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# load the trained model
model = load_model("ResNet50V2_model")

# define the input image size and the confidence threshold
IMG_SIZE = (224, 224)
CONF_THRESH = 0.5

# define the directory where the test data is stored
test_dir = "/home/jawabreh/Desktop/MFDD/dataset/test"

# define the names of the subdirectories containing the masked and unmasked images
masked_dir = "Masked"
unmasked_dir = "Unmasked"

# define the name of the Excel file to write the results to
excel_file = "results.xlsx"

# initialize lists to store the ground truth labels and the predicted labels
ground_truth_labels = []
predicted_labels = []
file_names = []

# iterate over all the masked images in the test directory
for filename in os.listdir(os.path.join(test_dir, masked_dir)):
    # load the input image and preprocess it
    image = cv2.imread(os.path.join(test_dir, masked_dir, filename))
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
        label = "Mask"
    else:
        label = "Unmasked"

    # add the ground truth and predicted labels to their respective lists
    ground_truth_labels.append("Mask")
    predicted_labels.append(label)
    file_names.append(filename)

# iterate over all the unmasked images in the test directory
for filename in os.listdir(os.path.join(test_dir, unmasked_dir)):
    # load the input image and preprocess it
    image = cv2.imread(os.path.join(test_dir, unmasked_dir, filename))
    image = cv2.resize(image, IMG_SIZE)
    image = img_to_array(image)
    image = preprocess_input(image)

    # make predictions on the input image using the trained model
    predictions = model.predict(np.expand_dims(image, axis=0))[0]

    # extract the probabilities for each class
    mask_prob = predictions[0]
    without_mask_prob = predictions[1]

    # determine the predicted class
    if without_mask_prob > mask_prob and without_mask_prob > CONF_THRESH:
        label = "Unmasked"
    else:
        label = "Mask"

    # add the ground truth and predicted labels to their respective lists
    ground_truth_labels.append("Unmasked")
    predicted_labels.append(label)
    file_names.append(filename)

# write the results to an Excel file
with open(excel_file, "w") as f:
    f.write("Ground Truth Label\tPredicted Label\tFile Name\n")
    for i in range(len(ground_truth_labels)):
        f.write("{}\t{}\t{}\n".format(ground_truth_labels[i], predicted_labels[i], file_names[i]))

print("Results written to {}".format(excel_file))
