import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D, Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
#preparinf process...
train_df = pd.read_csv(r'data\train.csv')
test_df  = pd.read_csv(r'data\test.csv')

train_df.head()

train_data = np.arrays(train_df,dtype='float32')
test_data = np.array(test_df,dtype='float32')
                       
x_train = train_data[:,1:]
y_train = train_data[:,0]

x_test = train_data[:,1:]
y_test = train_data[:,0]

x_train,x_validate,y_test,y_validate = train_test_split(x_train,y_train,test_size=0.2,random_state=12345)
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=784))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
#compilation process....

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=20,
          batch_size=128,validation_data=(x_validate,y_validate))
score = model.evaluate(x_test, y_test, batch_size=128)
                       
