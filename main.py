import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


train_df = pd.read_csv(r'data\')
test_df  = pd.read_csv(r'data\')

train_df.head()


train_data = np.arrays(train_df,dtype='float32')
test_data = np.array(test_df,dtype='float32')
                       
x_train = train_data[:,1:]
y_train = train_data[:,0]

x_test = train_data[:,1:]
y_test = train_data[:,0]

x_train,x_validate,y_test,y_validate = train_test_split(x_train,y_train,test_size=0.2,random_state=12345)
