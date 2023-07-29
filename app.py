# Importing Necessary Libraries

import numpy as np
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D  # maxpooling wrt to whole input area
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from numpy.linalg import norm
import os
from tqdm import tqdm  # to create a smart progress bar for the loops
import pickle

model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# print(model.summary())

def extract_features(img_path,model):                         # creating function for extracting features from the image
    img = image.load_img(img_path,target_size=(224,224))
    img_array = image.img_to_array(img)                       # converts image to array format
    expanded_img_array = np.expand_dims(img_array, axis=0)    # converts image shape to (1,224,224,3)
    preprocessed_img = preprocess_input(expanded_img_array)   # gives input to ResNet50
    result = model.predict(preprocessed_img).flatten()        # predicts output
    normalized_result = result/norm(result)                   # scaling down values between 0 & 1 by l2 normalization

    return normalized_result

filenames = []

for file in os.listdir('images'):                            # importing images into a list
    filenames.append(os.path.join('images',file))

feature_list = []

for file in tqdm(filenames):                                 # importing image features into a list
    feature_list.append(extract_features(file,model))

pickle.dump(feature_list,open('embeddings.pkl','wb'))        # saving features' list into a pickle file
pickle.dump(filenames,open('filenames.pkl','wb'))            # saving filesnames list into a pickle file