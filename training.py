import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import signal
from keras.utils import to_categorical
from glob import glob

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
def make_model1(emo_n,ch,size):
    model = Sequential()
    model.add(Conv2D(12,kernel_size=(3,3),activation='relu',input_shape=(size,size,ch)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(18,kernel_size=(3,3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(24,kernel_size=(3,3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(48,activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(32,activation='relu'))
    model.add(Dense(emo_n,activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model

N = 200
size = 64
emo_n = 7
l = 7
split1 = int(0.8*N)*l
split2 = int(0.9*N)*l
ch = 3
X = np.zeros((N*l,size,size,ch))
Y = np.zeros((N*l))
emo = ['smile','laugh','wink','smug','sad','null','none']
label = [0,1,2,3,4,5,6,7,8,8]

none = glob('dataset_/none/*.jpg')
none = np.array([x.split('/')[-1] for x in none])
rnd = np.random.choice(none,N,replace=False)
idx = np.array(['%d.jpg'%(x+1) for x in np.arange(N)])
file = np.zeros((l,N)).astype(str)
for e in range(emo_n-1):
    file[e,:] = np.random.permutation(idx)
file[l-1,:] = rnd[:N]
for i in range(N):
    for e in range(l):
        img = cv2.resize(cv2.imread('dataset_/%s/%s'%(emo[e],file[e,i])),(size,size),cv2.INTER_LINEAR)
        img = (img-np.mean(img,axis=(0,1)))/np.std(img,axis=(0,1))
        X[e+i*l,:,:,:] = img
        Y[e+i*l] = label[e]
Y = to_categorical(Y,emo_n)

X_train = X[:split1]
Y_train = Y[:split1]
X_val = X[split1:split2]
Y_val = Y[split1:split2]
X_test = X[split2:]
Y_test = Y[split2:]

model = make_model1(emo_n,ch,size)
his = model.fit(X_train,Y_train,batch_size=32,epochs=80,shuffle=True,validation_data=(X_val,Y_val))

y = model.predict(X_test)
y = np.argmax(y,axis=1)
yr = np.argmax(Y_test,axis=1)
print(np.mean(y==yr))

#model.save('model%d.h5'%("任意"))
