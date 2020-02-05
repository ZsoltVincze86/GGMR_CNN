# GGMR_CNN for surfer map image classification - Model building and training



from keras.models import Sequential 
from keras.layers import Convolution2D 
from keras.layers import MaxPooling2D 
from keras.layers import Flatten
from keras.layers import Dense 
from keras.layers import Dropout
from keras import optimizers
from keras.callbacks import ModelCheckpoint

# --------Initialising the CNN----------
classifier = Sequential() 
classifier.size=128                 # size of the image for input

# ------- Convolution----------
classifier.add(Convolution2D(32, (3,3), input_shape = (classifier.size, classifier.size, 3), activation = 'relu'))
classifier.add(Convolution2D(32, (3,3), activation = 'relu')) 
classifier.add(Convolution2D(32, (3,3), activation = 'relu')) 
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Dropout(0.25))       # adding dropout reguralization to model

# adding a 2nd convolutional block
classifier.add(Convolution2D(64, (3,3), activation = 'relu'))
classifier.add(Convolution2D(64, (3,3), activation = 'relu'))
classifier.add(Convolution2D(64, (3,3), activation = 'relu')) 
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Dropout(0.25)) 

# adding a 3rd convolutional block
classifier.add(Convolution2D(128, (3,3), activation = 'relu')) 
classifier.add(Convolution2D(128, (3,3), activation = 'relu'))
classifier.add(Convolution2D(128, (3,3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Dropout(0.25))

#---------Flattening-------------
classifier.add(Flatten())

#---------Full Connection-----------
classifier.add(Dense(512, activation = 'relu')) 
classifier.add(Dropout(0.5)) 
classifier.add(Dense(1, activation = 'sigmoid')) 

# --------Compiling the CNN----------
opt = optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)
classifier.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = ['accuracy'])

#---------Training-------
#using Keras image augmentation method for increase the image numer for the network training. 
#It generates a lot of slightly different pictures from the originals, so it prevents model overfitting
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,                 #the value of each pixel goes between 0 and 1 from 0 and 255
        shear_range=0.2,
        zoom_range=0.2, 
        horizontal_flip=False,
        rotation_range=0)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set', 
                                                 target_size=(classifier.size, classifier.size), 
                                                 batch_size=32, 
                                                 class_mode='binary') 

test_set = test_datagen.flow_from_directory('dataset/valid_set',
                                            target_size=(classifier.size, classifier.size),
                                            batch_size=32, 
                                            class_mode='binary')

mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

classifier.fit(training_set,
               steps_per_epoch=1600,
               epochs=50, 
               validation_data=test_set,
               validation_steps=336,
               callbacks=[mc]) 

