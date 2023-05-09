from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os

# initialize the initial learning rate, number of epochs to train for,
# and batch size
INIT_LR = 1e-4
EPOCHS = 20
BS = 32

# paths to input datasets
train_dataset_path = "/home/jawabreh/Desktop/Test_MFD/data/train"
test_dataset_path = "/home/jawabreh/Desktop/Test_MFD/data/test"

# grab the list of images in our training dataset directory, then initialize
# the list of data (i.e., images) and class images
print("[INFO] loading training images...")
train_image_paths = list(paths.list_images(train_dataset_path))
train_data = []
train_labels = []

# loop over the training image paths
for imagePath in train_image_paths:
    # extract the class label from the filename
    label = imagePath.split(os.path.sep)[-2]

    # load the input image (224x224) and preprocess it
    image = load_img(imagePath, target_size=(224, 224))
    image = img_to_array(image)
    image = preprocess_input(image)

    # update the train_data and train_labels lists, respectively
    train_data.append(image)
    train_labels.append(label)

# convert the train_data and train_labels to NumPy arrays
train_data = np.array(train_data, dtype="float32")
train_labels = np.array(train_labels)

# perform one-hot encoding on the train_labels
lb = LabelBinarizer()
train_labels = lb.fit_transform(train_labels)
train_labels = to_categorical(train_labels)

# grab the list of subdirectories in our testing dataset directory
test_subdirs = os.listdir(test_dataset_path)

# initialize the list of data (i.e., images) and class images
print("[INFO] loading testing images...")
test_data = []
test_labels = []

# loop over the subdirectories
for subdir in test_subdirs:
    subdir_path = os.path.join(test_dataset_path, subdir)
    if not os.path.isdir(subdir_path):
        continue

    # grab the list of images in the current subdirectory
    image_paths = list(paths.list_images(subdir_path))

    # loop over the image paths
    for imagePath in image_paths:
        # load the input image (224x224) and preprocess it
        image = load_img(imagePath, target_size=(224, 224))
        image = img_to_array(image)
        image = preprocess_input(image)

        # update the test_data and test_labels lists, respectively
        test_data.append(image)
        test_labels.append(subdir)

# convert the test_data and test_labels to NumPy arrays
test_data = np.array(test_data, dtype="float32")
test_labels = np.array(test_labels)

# perform one-hot encoding on the test_labels
lb = LabelBinarizer()
test_labels = lb.fit_transform(test_labels)
test_labels = to_categorical(test_labels)

base_model = ResNet50V2(weights="imagenet", include_top=False,
input_tensor=Input(shape=(224, 224, 3)))

head_model = base_model.output
head_model = AveragePooling2D(pool_size=(7, 7))(head_model)
head_model = Flatten(name="flatten")(head_model)
head_model = Dense(128, activation="relu")(head_model)
head_model = Dropout(0.5)(head_model)
head_model = Dense(2, activation="softmax")(head_model)

model = Model(inputs=base_model.input, outputs=head_model)

for layer in base_model.layers:
    layer.trainable = False

print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR)
model.compile(loss="binary_crossentropy", optimizer=opt,
metrics=["accuracy"])

print("[INFO] training head...")
H = model.fit(
ImageDataGenerator().flow(train_data, train_labels, batch_size=BS),
steps_per_epoch=len(train_data) // BS,
validation_data=(test_data, test_labels),
validation_steps=len(test_data) // BS,
epochs=EPOCHS)

print("[INFO] evaluating network...")
pred_idxs = model.predict(test_data, batch_size=BS)
pred_idxs = np.argmax(pred_idxs, axis=1)

print(classification_report(test_labels.argmax(axis=1), pred_idxs,
target_names=lb.classes_))

print("[INFO] saving mask detector model...")
model.save("mask_detector.h5", save_format="h5")

N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.show()




