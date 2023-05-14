import os
import pandas as pd
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model

# load the saved model
model = load_model("MobileNetV2_model")

# define the test dataset path
test_path = "/home/jawabreh/Desktop/MFDD/dataset/test"

# initialize the list of data and class images
data = []
results = []

# loop over the image paths
for root, dirs, files in os.walk(test_path):
    for file in files:
        # get the class label from the directory name
        label = os.path.basename(root)
        
        # load the input image (224x224) and preprocess it
        image_path = os.path.join(root, file)
        image = load_img(image_path, target_size=(224, 224))
        image = img_to_array(image)
        image = preprocess_input(image)
        data.append(image)
        
        # get the prediction results for the image
        prediction = model.predict(image.reshape(1, 224, 224, 3))[0]
        class_name = "Masked" if prediction[0] > prediction[1] else "Unmasked"
        confidence_rate = max(prediction)
        
        # append the prediction result, file name, and confidence rate to a list
        results.append([label, class_name, file, confidence_rate])

# save the results in an Excel sheet using pandas
df = pd.DataFrame(results, columns=["Ground Truth", "Classification Result", "File Name", "Confidence Rate"])
df.to_excel("results.xlsx", index=False)
