# -*- coding: utf-8 -*-
"""
Created on Sun May 17 17:27:55 2020

@author: kingslayer
"""

##### FASHION CLASSIFIER ######


#Importin the data

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


fashion_train_df=pd.read_csv("Fashion-mnist_train.csv")
fashion_test_df=pd.read_csv("Fashion-mnist_test.csv")


training=np.array(fashion_train_df,dtype="float32")
testing=np.array(fashion_test_df,dtype="float32")

import random
i=random.randint(1,60000)
plt.imshow(training[i,1:].reshape(28,28))
label=training[i,0]



#Training the model

X_train=training[:,1:]/255
y_train=training[:,0]

X_test=testing[:,1:]/255
y_test=testing[:,0]

X_train=X_train.reshape(X_train.shape[0],*(28,28,1))
X_test=X_test.reshape(X_test.shape[0],*(28,28,1))


import keras
from keras.models import Sequential
from keras.layers import Convolution2D,Flatten,Dense,MaxPooling2D,Dropout

classifier=Sequential()

classifier.add(Convolution2D(64,3,3,input_shape=(28,28,1),activation="relu"))

classifier.add(MaxPooling2D(2,2))

classifier.add(Flatten())

classifier.add(Dense(output_dim=64,activation="relu"))
classifier.add(Dropout(0.2))

classifier.add(Dense(output_dim=64,activation="relu"))
classifier.add(Dropout(0.2))

classifier.add(Dense(output_dim=10,activation="sigmoid"))

classifier.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])




classifier.fit(X_train,y_train,batch_size=300,epochs=30,validation_data=(X_test,y_test))

y_pred=classifier.predict_classes(X_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

sns.heatmap(cm,annot=True)


from sklearn.metrics import classification_report
num=10
target=[f"class:{n}" for n in range(num)]
print(classification_report(y_test,y_pred,target_names=target))
