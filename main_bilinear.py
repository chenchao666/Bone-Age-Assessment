from keras.applications.resnet50 import ResNet50
from keras.applications.vgg19 import VGG19
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Flatten, Dense, Input, Lambda, Reshape, Permute
from keras import backend as K
import pickle
import numpy as np
import matplotlib.pyplot as plt
import keras
import os
os.environ["CUDA_VISIBLE_DEVICES"]="4"
batch_size = 32
epochs = 30
##function
def matmul(x):
    return x[0]*x[1]*x[2]


# Load data
print('...loading training data')
f = open('data.pkl', 'rb')
x = pickle.load(f)
f.close()

f = open('data_age.pkl', 'rb')
y = pickle.load(f)
f.close()

f = open('data_gender.pkl','rb')
gender = pickle.load(f)
f.close()


x = np.asarray(x, dtype=np.float32)
y = np.asarray(y)
gender = np.asarray(gender)

x /= 255.
gender =2*( gender-0.5)
x_final = []
y_final = []
gender_final = []

# Shuffle images and split into train, validation and test sets
random_no = np.random.choice(x.shape[0], size=x.shape[0], replace=False)
for i in random_no:
    x_final.append(x[i,:,:,:])
    y_final.append(y[i])
    gender_final.append(gender[i])

x_final = np.asarray(x_final)
y_final = np.asarray(y_final)
gender_final = np.asarray(gender_final)
print (y_final[:50])
print (gender_final[:50])
k = 500 # Decides split count
x_test = x_final[:k,:,:,:]
y_test = y_final[:k]
gender_test = gender_final[:k]
x_valid = x_final[k:2*k,:,:,:]
y_valid = y_final[k:2*k]
gender_valid = gender_final[k:2*k]
x_train = x_final[2*k:,:,:,:]
y_train = y_final[2*k:]
gender_train = gender_final[2*k:]

print ('x_train shape:'+ str(x_train.shape))
print ('y_train shape:'+ str(y_train.shape))
print ('gender_train shape:'+ str(gender_train.shape))
print ('x_valid shape:'+ str(x_valid.shape))
print ('y_valid shape:'+ str(y_valid.shape))
print ('gender_valid shape:' + str(gender_valid.shape))
print ('x_test shape:'+ str(x_test.shape))
print ('y_test shape:'+ str(y_test.shape))

# Using VGG19/ResNet-50/InceptionV3 with pretrained weights from Imagenet 
base_model = InceptionV3(weights='imagenet', include_top=False)
for i,layer in enumerate(base_model.layers):
    print (i,layer.name)
input = Input(shape=(800,600,3),name='input1')
input_gender = Input(shape=(1,),dtype='float32',name='input2')
output = base_model(input)
gender_embedding=Dense(16)(input_gender)
#gender_embedding=Dense(12)(gender_embedding)
#x = keras.layers.MaxPooling2D(pool_size=(2,2))(output)
#x = keras.layers.Conv2D(512,kernel_size=(3,3))(x)
x = keras.layers.Conv2D(512,kernel_size=(1,1))(output)
x = keras.layers.Conv2D(128,kernel_size=(1,1))(x)
#x = keras.layers.MaxPooling2D(pool_size=(3,3))(x)
print (K.int_shape(x))
x = Reshape((391,128))(x)
x=keras.layers.dot(inputs=[x,x],axes=1)
#x = Permute((2,1))(x)
#x_1 = Reshape((25,128,1,1))(x)
#x_2 = Reshape((25,1,128,1))(x)
#x_3 = Reshape((25,1,1,128))(x)
#x_4=Lambda(matmul)([x_1,x_2,x_3])
#x =Lambda(lambda x: K.mean(x,axis=1))(x_4)
print (K.int_shape(x))
x=Flatten()(x)
f = keras.layers.Concatenate(axis=1)([x,gender_embedding])
print (K.int_shape(f)) 
#x = Dense(256, activation='relu')(x)
predictions = Dense(1)(f)

model = Model(inputs=[input,input_gender], outputs=predictions)
Adam=keras.optimizers.Adam(lr=0.0003,beta_1=0.9,beta_2=0.999)
model.compile(optimizer=Adam, loss='mean_absolute_error', metrics=['MAE'])

#model.load_weights('model.h5')
# Save weights after every epoch
checkpoint =keras.callbacks.ModelCheckpoint(filepath='weights/weights.{epoch:02d}-{val_loss:.2f}.hdf5',save_weights_only=True,period=30)
history =model.fit([x_train,gender_train],y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=([x_valid,gender_valid],y_valid), callbacks = [checkpoint])
score = model.evaluate([x_test,gender_test], y_test, batch_size=batch_size)
print('Test loss:', score[0])
print('Test MAE:', score[1])


for layer in base_model.layers[:250]:
    layer.trainable=False
for layer in base_model.layers:
    print (layer.name,layer.trainable)
Adam=keras.optimizers.Adam(lr=0.00001,beta_1=0.9,beta_2=0.999)
model.compile(optimizer=Adam, loss='mean_absolute_error', metrics=['MAE'])
history = model.fit([x_train,gender_train], y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=([x_valid,gender_valid],y_valid), callbacks = [checkpoint])
score = model.evaluate([x_test,gender_test], y_test, batch_size=batch_size)
print('Test loss:', score[0])
print('Test MAE:', score[1])


for layer in base_model.layers[:310]:
    layer.trainable=False
for layer in base_model.layers:
    print (layer.name,layer.trainable)
Adam=keras.optimizers.Adam(lr=0.00001,beta_1=0.9,beta_2=0.999)
model.compile(optimizer=Adam, loss='mean_absolute_error', metrics=['MAE'])
history = model.fit([x_train,gender_train], y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=([x_valid,gender_valid],y_valid), callbacks = [checkpoint])
score = model.evaluate([x_test,gender_test], y_test, batch_size=batch_size)
print('Test loss:', score[0])
print('Test MAE:', score[1])
model.save_weights("model.h5")


with open('history.pkl', 'wb') as f:
	pickle.dump(history.history, f)
f.close()


