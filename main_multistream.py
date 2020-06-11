from keras.applications.resnet50 import ResNet50
from keras.applications.vgg19 import VGG19
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from keras.preprocessing import image
from keras.models import Model, load_model
from keras.layers import Flatten, Dense, Input, Reshape, Lambda, Concatenate, dot
from keras import backend as K
import pickle
import numpy as np
import matplotlib.pyplot as plt
import keras
from func_utils import *
import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"
os.environ['OMP_NUM_THREADS']='6'
batch_size = 8
epochs = 30

# Load data
print('...loading training data')
f = open('dataHand.pkl', 'rb')
data = pickle.load(f)
f.close()

f = open('dataR1.pkl', 'rb')
data_S = pickle.load(f)
f.close()

f = open('dataR2.pkl', 'rb')
data_M = pickle.load(f)
f.close()

f = open('data_age.pkl', 'rb')
age = pickle.load(f)
f.close()

f = open('data_gender.pkl','rb')
gender = pickle.load(f)
f.close()


data = np.asarray(data, dtype=np.float16)
data_S = np.asarray(data_S, dtype=np.float16)
data_M = np.asarray(data_M, dtype=np.float16)
print (data.shape)
print (data_S.shape)
print (data_M.shape)

age = np.asarray(age)
gender = np.asarray(gender)
Ones = np.ones((data.shape[1],data.shape[2]))
Zeros = np.zeros((data.shape[1],data.shape[2]))
#for i in range(data.shape[0]):
#    if gender[i]==0:
#        data[i,:,:,2]=Zeros
#        data_S[i,:,:,2]=Zeros
#        data_M[i,:,:,2]=Zeros
#    elif gender[i]==1:
#        data[i,:,:,2]=Ones
#        data_S[i,:,:,2]=Ones
#        data_M[i,:,:,2]=Ones



data /= 255.
data_S/=255.
data_M/=255.
gender =2*( gender-0.5)
x_final = []
xs_final = []
xm_final = []
y_final = []
gender_final = []

# Shuffle images and split into train, validation and test sets
random_no = np.random.choice(data.shape[0], size=data.shape[0], replace=False)
for i in random_no:
    x_final.append(data[i,:,:,:])
    xs_final.append(data_S[i,:,:,:])
    xm_final.append(data_M[i,:,:,:])
    y_final.append(age[i])
    gender_final.append(gender[i])

x_final = np.asarray(x_final)
xs_final = np.asarray(xs_final)
xm_final = np.asarray(xm_final)
y_final = np.asarray(y_final)
gender_final = np.asarray(gender_final)





print (y_final[:50])
print (gender_final[:50])
k = 500 # Decides split count
x_test = x_final[:k,:,:,:]
xs_test = xs_final[:k,:,:,:]
xm_test =xm_final[:k,:,:,:]
y_test = y_final[:k]
gender_test = gender_final[:k]
x_valid = x_final[k:2*k,:,:,:]
xs_valid = xs_final[k:2*k,:,:,:]
xm_valid = xm_final[k:2*k,:,:,:]
y_valid = y_final[k:2*k]
gender_valid = gender_final[k:2*k]
x_train = x_final[2*k:,:,:,:]
xs_train = xs_final[2*k:,:,:,:]
xm_train = xm_final[2*k:,:,:,:]
y_train = y_final[2*k:]
gender_train = gender_final[2*k:]

del data
del data_S
del data_M
del x_final
del xs_final
del xm_final


print ('x_train shape:'+ str(x_train.shape))
print ('y_train shape:'+ str(y_train.shape))
print ('gender_train shape:'+ str(gender_train.shape))
print ('x_valid shape:'+ str(x_valid.shape))
print ('y_valid shape:'+ str(y_valid.shape))
print ('gender_valid shape:' + str(gender_valid.shape))
print ('x_test shape:'+ str(x_test.shape))
print ('y_test shape:'+ str(y_test.shape))

# Using VGG19 with pretrained weights from Imagenet 
base_model_1 = InceptionV3(weights='imagenet', include_top=False)
base_model_2 = ResNet50(weights='imagenet', include_top=False)
base_model_3 = VGG19(weights='imagenet', include_top=False)
for layer in base_model_1.layers:
    layer.name = layer.name + str('_1')
for layer in base_model_2.layers:
    layer.name = layer.name + str('_2')
for layer in base_model_3.layers:
    layer.name = layer.name + str('_3')

input_x = Input(shape=(560,560,3),name='input1')
input_xs = Input(shape=(560,560,3),name='input2')
input_xm =  Input(shape=(560,560,3),name='input3')
input_gender = Input(shape=(1,),dtype='float32',name='input4')
output_x = base_model_1(input_x)
output_xs = base_model_2(input_xs)
output_xm = base_model_3(input_xm)
gender_embedding = Dense(32)(input_gender)
#x = keras.layers.MaxPooling2D(pool_size=(5,5))(output)
#x = keras.layers.Conv2D(512,kernel_size=(3,3))(x)
x = keras.layers.Conv2D(128,kernel_size=(1,1))(output_x)
xs = keras.layers.Conv2D(128,kernel_size=(1,1))(output_xs)
xm = keras.layers.Conv2D(128,kernel_size=(1,1))(output_xm)

x = keras.layers.MaxPooling2D(pool_size=(3,3))(x)
xs = keras.layers.MaxPooling2D(pool_size=(3,3))(xs)
xm = keras.layers.MaxPooling2D(pool_size=(3,3))(xm)


x1=Flatten()(x)
x2=Flatten()(xs)
x3=Flatten()(xm)
merge_feature = Concatenate(axis=1)([x1,x2,x3,gender_embedding])
print (K.int_shape(merge_feature))
predictions = Dense(1)(merge_feature)

model = Model(inputs=[input_x,input_xs,input_xm,input_gender], outputs=predictions)
for i,layer in enumerate(model.layers):
    print (i,layer.name)

Adam=keras.optimizers.Adam(lr=0.0003,beta_1=0.9,beta_2=0.999)
model.compile(optimizer=Adam, loss='mean_absolute_error', metrics=['MAE'])

# Save weights after every epoch
checkpoint =keras.callbacks.ModelCheckpoint(filepath='weights/weights.{epoch:02d}-{val_loss:.2f}.hdf5',save_weights_only=True,period=30)
history=model.fit([x_train,xs_train,xm_train,gender_train],y_train,batch_size=batch_size,epochs=50,verbose=1,validation_data=([x_valid,xs_valid,xm_valid,gender_valid],y_valid), callbacks = [checkpoint])
score = model.evaluate([x_test,xs_test,xm_test,gender_test], y_test, batch_size=batch_size)
print('Test loss:', score[0])
print('Test MAE:', score[1])

##Visulization
#GAPAttention(model,weights,'/raid/chenchao/code/BoneAge/BoneAge/data/train/')
#ShowAttentionV1(base_model,'/raid/chenchao/code/BoneAge/BoneAge/data/train/')

#for layer in base_model.layers[:16]:
#    layer.trainable=False
#for layer in base_model.layers:
#    print (layer.name,layer.trainable)
Adam=keras.optimizers.Adam(lr=0.0001,beta_1=0.9,beta_2=0.999)
model.compile(optimizer=Adam, loss='mean_absolute_error', metrics=['MAE'])
history = model.fit([x_train,xs_train,xm_train,gender_train],y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=([x_valid,xs_valid,xm_valid,gender_valid],y_valid), callbacks = [checkpoint])
score = model.evaluate([x_test,xs_test,xm_test,gender_test], y_test, batch_size=batch_size)
print('Test loss:', score[0])
print('Test MAE:', score[1])

#ShowAttentionV1(base_model,'/raid/chenchao/code/BoneAge/BoneAge/data/train/')

Adam=keras.optimizers.Adam(lr=0.00001,beta_1=0.9,beta_2=0.999)
model.compile(optimizer=Adam, loss='mean_absolute_error', metrics=['MAE'])
history =model.fit([x_train,xs_train,xm_train,gender_train],y_train,batch_size=batch_size,epochs=10,verbose=1,validation_data=([x_valid,xs_valid,xm_valid,gender_valid],y_valid), callbacks = [checkpoint])
score = model.evaluate([x_test,xs_test,xm_test,gender_test], y_test, batch_size=batch_size)
print('Test loss:', score[0])
print('Test MAE:', score[1])

#ShowAttentionV1(base_model,'/raid/chenchao/code/BoneAge/BoneAge/data/train/')


model.save_weights("model.h5")
with open('history.pkl', 'wb') as f:
	pickle.dump(history.history, f)
f.close()


