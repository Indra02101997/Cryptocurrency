import time
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import *
import numpy as np

EPOCHS=10
BATCH_SIZE=64
NAME="RNNWITHCRYPTOCURRENCY{}".format(int(time.time()))

xtrain=np.array(np.load('D://Personal Files/Python Programs/DeepLearningSentdex/Cryptocurrency/train_x.npy'))
ytrain=np.array(np.load('D://Personal Files/Python Programs/DeepLearningSentdex/Cryptocurrency/train_y.npy'))
xtest=np.array(np.load('D://Personal Files/Python Programs/DeepLearningSentdex/Cryptocurrency/validation_x.npy'))
ytest=np.array(np.load('D://Personal Files/Python Programs/DeepLearningSentdex/Cryptocurrency/validation_y.npy'))

model=Sequential()
model.add(LSTM(128,input_shape=xtrain.shape[1:],activation='relu',return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(LSTM(128,activation='relu',return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(LSTM(128,activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dense(32,activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(2,activation='softmax'))

opt=tf.keras.optimizers.Adam(lr=0.001,decay=1e-6)

model.compile(loss='sparse_categorical_crossentropy',optimizer=opt,metrics=['accuracy'])

tensorboard=TensorBoard(log_dir='D://Personal Files/Python Programs/DeepLearningSentdex/Cryptocurrency/logs/{}'.format(NAME))

model.fit(xtrain,ytrain,batch_size=BATCH_SIZE,epochs=1,validation_data=(xtest,ytest),callbacks=[tensorboard])

score=model.evaluate(xtest,ytest,verbose=0)

print('Test Loss = ',score[0])
print('Test Accuracy = ',score[1])

model.save('D://Personal Files/Python Programs/DeepLearningSentdex/Cryptocurrency/{}'.format(NAME))


          
