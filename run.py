# Randomly picking 10 images from UTKFace and predicting upon
import random
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
from tensorflow.keras.models import load_model


model = load_model('age_gender.h5')

path = "UTKFace/"
files = os.listdir(path)
random.shuffle(files)
files = files[:10]

for file in files:
    plt.figure(figsize=(5,5))
    img = cv2.imread(path+file)
    pred = model.predict(np.expand_dims(img,axis=0))
    age = int(pred[0][0])
    gen = pred[1]
    gender = lambda gen: 'male' if gen < 0.5 else 'female' # the value lies between 0 and 1
    gender= gender(gen)

    # write age and gender on the image

    img = cv2.putText(img, "Age: "+str(int(pred[0][0])), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    img = cv2.putText(img, "gender: "+ str(gender), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    # convert to rgb
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # plt.imshow(
    plt.show()

