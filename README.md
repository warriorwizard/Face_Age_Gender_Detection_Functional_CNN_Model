# Face_Age_Gender_Detection_Functional_CNN_Model
uses functional model api of keras to build age and gender detection from scratch

## Using Transfer learning to train the model 


```
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import *
resnet = ResNet50(include_top= False, weights= 'imagenet',input_shape = (200,200,3))
resnet.trainable=  False
output = resnet.layers[-1].output

flatten = Flatten()(output)

dense1 = Dense(512, activation='relu')(flatten)
dense2 = Dense(512,activation='relu')(flatten)

dense3 = Dense(512,activation='relu')(dense1)
dense4 = Dense(512,activation='relu')(dense2)

output1 = Dense(1,activation='linear',name='age')(dense3)
output2 = Dense(1,activation='sigmoid',name='gender')(dense4)
model = Model(inputs=resnet.input,outputs=[output1,output2])
# Early stopping and Reduce learning rate if the model is not improving
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau

earlystop = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', patience=5, verbose=1, factor=0.2, min_lr=0.00001)
model.compile(optimizer = 'Adam', loss = {'age': 'mae', 'gender': 'binary_crossentropy'},  metrics={'age': 'mae', 'gender': 'accuracy'},loss_weights={'age':1,'gender':99})

```
