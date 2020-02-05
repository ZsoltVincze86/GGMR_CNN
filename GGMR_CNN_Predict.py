# GGMR_CNN for surfer map image classification - Image prediction


from keras.models import load_model

# load model
classifier = load_model('GGMR_CNN.h5')

# summarize model to see if it loaded properly
classifier.summary() 

# prediction
import numpy as np
from keras.preprocessing import image
import os
directory = 'test_images'
for file in os.listdir(directory):
    if ".jpg" in file:
        surfer_map = image.load_img(directory+'/'+file, target_size=(128,128))
        surfer_map = image.img_to_array(surfer_map)
        surfer_map = np.expand_dims(surfer_map, axis=0)
        result = classifier.predict(surfer_map)
        if result>0:
            prediction = 'NO DAMAGE'
        else: prediction = 'A HOLE'
        print (f'There is {prediction} on the map: {file}')





