import imageio
from random import randint
from glob import glob
import pandas as pd
import numpy as np
from keras.layers.core import Flatten, Dropout
from keras.layers import Input, Dense, Lambda, Layer
from keras import backend as K
from keras import applications
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint

# data generator for neural network
# forms correct and incorrect pairings of images with text descriptions and labels them as correct (1) or incorrect (0)

def generator(batch_size, df):
    
    batch_img = np.zeros((batch_size, 299, 299, 3))
    batch_txt = np.zeros((batch_size, 500))
    batch_labels = np.zeros((batch_size,1))
    
    video_ids = df['video']
    video_txt = df['txt_enc']
    
    length = len(df) -1
    
    while True:
        for i in range(batch_size//6):
            
            i = i*6
            
            #correct
            sample = randint(0,length)
            file = video_ids.iloc[sample]
            
            filenames = glob('/home/ubuntu/models/Data/images/' + file + '/*.jpg')
            correct_txt = video_txt.iloc[sample]
            
            im = imageio.imread(filenames[0])

            batch_img[i-6] = im
            batch_txt[i-6] = correct_txt
            batch_labels[i-6] = 1
            
            #correct             
            im = imageio.imread(filenames[-1])
            
            batch_img[i-5] = im
            batch_txt[i-5] = correct_txt
            batch_labels[i-5] = 1
            
            #correct            
            im = imageio.imread(filenames[-10])
            
            batch_img[i-4] = im
            batch_txt[i-4] = correct_txt
            batch_labels[i-4] = 1
            
            #incorrect 
            file = video_ids.iloc[randint(0,length)]
            filename = glob('/home/ubuntu/models/Data/images/' + file + '/*.jpg')[0]
            
            im = imageio.imread(filename)

            batch_img[i-3] = im
            batch_txt[i-3] = correct_txt
            batch_labels[i-3] = 0
            
            #incorrect
            file = video_ids.iloc[randint(0,length)]
            filename = glob('/home/ubuntu/models/Data/images/' + file + '/*.jpg')[0]
            
            im = imageio.imread(filename)

            batch_img[i-2] = im
            batch_txt[i-2] = correct_txt
            batch_labels[i-2] = 0
            
            #incorrect
            file = video_ids.iloc[randint(0,length)]
            filename = glob('/home/ubuntu/models/Data/images/' + file + '/*.jpg')[0]
            
            im = imageio.imread(filename)

            batch_img[i-1] = im
            batch_txt[i-1] = correct_txt
            batch_labels[i-1] = 0
            
        yield [batch_txt, batch_img], batch_labels
 
# siamese neural network 

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def contrastive_loss(y_true, y_pred):
    margin = 1
    return K.mean(y_true * K.square(y_pred) +
                  (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

def create_img_encoder(input_dim, xcept):
    x = Sequential()
    x.add(xcept)
    x.add(Dense(500, activation="relu"))
    x.add(Dropout(0.5))
    x.add(Dense(512, activation="relu"))
    return x

def create_txt_encoder(input_dim):
    x = Sequential()
    x.add(Dense(500, input_shape = (500,), activation="relu"))
    x.add(Dropout(0.5))
    x.add(Dense(512, activation="relu"))
    return x

def compute_accuracy(predictions, labels):
    return labels[predictions.ravel() < 0.5].mean()

xcept = applications.xception.Xception(include_top=True, weights='imagenet', input_tensor=None, input_shape=(299, 299, 3))

for layer in xcept.layers[:-8]:
    layer.trainable = False

input_txt = Input (shape = (500,))
input_img = Input(shape=(299, 299, 3))

txt_enc = create_txt_encoder(input_txt)
img_enc = create_img_encoder(input_img, xcept)

encoded_txt = txt_enc(input_txt)
encoded_img = img_enc(input_img)

distance = Lambda(euclidean_distance,
                  output_shape=eucl_dist_output_shape)([encoded_txt, encoded_img])

model = Model([input_txt, input_img], distance)
model.load_weights('/home/ubuntu/models/siamese/old/siamese.31.h5')
siamesecheckpoint = ModelCheckpoint("/home/ubuntu/models/siamese/new/siamese.{epoch:02d}.h5", verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=0)

rms = RMSprop(lr=0.00001)
model.compile(loss=contrastive_loss, optimizer=rms)

model.summary()

df_train = txt_final[:2500]
df_test = txt_final[2500:]

model.fit_generator(generator(30, df_train), steps_per_epoch= 1000, validation_data= generator(30, df_test), validation_steps=300, epochs=250, verbose=1, callbacks=[siamesecheckpoint])
