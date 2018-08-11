import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D, Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from keras.optimizers import SGD
import keras


#preparing process...
train_df = pd.read_csv(r'train.csv')
test_df  = pd.read_csv(r'test.csv')

train_df.head()

feat_cols = train_df.columns.tolist()[1:]


                       
x_train =  train_df[feat_cols].values.reshape(-1,28,28,1)/255
y_train =  train_df.iloc[:,0].values

x_test =   test_df[feat_cols].values.reshape(-1,28,28,1)/255
y_test =   test_df.iloc[:,0].values

x_train = x_train.reshape(-1, 784).astype('float32')
x_test = x_test.reshape(-1, 784).astype('float32')

y_train = keras.utils.to_categorical(y_train)
y_test =  keras.utils.to_categorical(y_test)


#x_train,x_validate,y_test,y_validate = train_test_split(x_train,y_train,test_size=0.2,random_state=12345)
model = Sequential()
model.add(Dense(784, activation='relu', input_dim=784))
model.add(Dense(10, activation='softmax'))
#compilation process....

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=20,
          batch_size=128,verbose=1)
score = model.evaluate(x_test, y_test, batch_size=128)
                       
