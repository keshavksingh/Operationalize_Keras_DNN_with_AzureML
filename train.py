# Spark configuration and packages specification. The dependencies defined in
# this file will be automatically provisioned for each run that uses Spark.

import sys
import os
import argparse
from azureml.logging import get_azureml_logger
from sklearn.model_selection import train_test_split

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop,Adam
from keras.models import Model
import numpy as np
import pandas as pd
train = pd.read_csv('ProductPurchase.csv')
train.head()
train.columns
X_train=train[['TimeSpentOnWeb','TimeSpentOnProductPage']]
Y_train=train[['ProductPurchased']]
Col_To_Transform=['ProductPurchased']
Y_With_Dummies=pd.get_dummies(columns=Col_To_Transform,data=Y_train)
X_train = X_train[['TimeSpentOnWeb','TimeSpentOnProductPage']].astype(float)
Y_train = Y_With_Dummies[['ProductPurchased_A','ProductPurchased_B','ProductPurchased_C']].astype(float)

XX_train, XX_test, yy_train, yy_test = train_test_split(X_train,Y_train,test_size=0.25, random_state=11)

model = Sequential()
model.add(Dense(16, activation='relu', input_shape=(2,)))
model.add(Dense(8, activation='relu', input_shape=(16,)))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])
history = model.fit(XX_train, yy_train,
                    epochs=1000,batch_size=5,
                    verbose=1,
                    validation_data=(XX_test, yy_test))
score = model.evaluate(XX_test, yy_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

from keras.models import load_model
model.save("PredictPurchase_model.h5")
m = load_model("PredictPurchase_model.h5")

preddf=pd.read_csv('PredictProductPurchase.csv')
y_pred = m.predict_classes(preddf,verbose=1)

y_pred = model.predict_classes(preddf,verbose=1)
y_pred
predictions=pd.DataFrame(data=y_pred,columns=['PredictedProduct'])
predictions['PredictedProduct']= predictions['PredictedProduct'].apply(lambda x: 'Product_C' if x ==2 else ('Product_B' if x==1 else 'Product_A'))
print(predictions)

# initialize the logger
logger = get_azureml_logger()

# add experiment arguments
parser = argparse.ArgumentParser()
# parser.add_argument('--arg', action='store_true', help='My Arg')
args = parser.parse_args()
print(args)

# This is how you log scalar metrics
# logger.log("MyMetric", value)

# Create the outputs folder - save any outputs you want managed by AzureML here
os.makedirs('./outputs', exist_ok=True)