import numpy as np
import pandas as pd
import tensorflow as tf
import json
import keras

from sklearn.model_selection import train_test_split
from keras_unet.metrics import iou, iou_thresholded
from PIL import Image
import cv2

print(tf.__version__)



from data import DataGenerator, MNISTDataGenerator, DataGeneratorUNET
# Parameters
params = {'dim': (160, 160),
          'batch_size': 8,
          
          'n_channels': 1,
          'shuffle': True}


# Datasets
# Opening JSON file
f = open('/content/drive/MyDrive/BoneSegm/mask_labels_carpal_bones/dict.json')
data_dicts = json.load(f)

model = keras.models.load_model("/content/drive/MyDrive/BoneSegm/model/model_3")
model.summary()


# initialize LeNet and then write the network architecture
# visualization graph to disk

tf.keras.utils.plot_model(model, to_file="/content/drive/MyDrive/BoneSegm/model.png", show_shapes=True)

for index, img_path in enumerate(data_dicts['partition']['validation']):
  img_path_complete = '/content/drive/MyDrive/BoneSegm/mask_labels_carpal_bones/' + img_path
  
  img = Image.open(img_path_complete).convert('RGB')
  src1 = img
  img = img.convert('L')

  newsize = (512, 512)
  img = img.resize(newsize)
  img.save("/content/drive/MyDrive/BoneSegm/results/original_images/"+ str(index) + ".png")
  

  img = np.array(img) / 255

  colors = [(255, 0, 0), (0,255,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255), (100, 200, 0), (230, 0, 60)]

  #X = np.empty((self.batch_size, *self.dim, self.n_channels))
  img = np.expand_dims(img, 0)
  y_pred = model.predict(img)

  for index_bone in range(1, y_pred.shape[3]):
    img_array = y_pred[0, :, :, index_bone] * 255

    print(np.max(img_array), np.min(img_array)) 

    img = Image.fromarray(img_array * 0.3)
    orig_img_size = src1.size
    img = img.resize(orig_img_size)
    img_RGB = img.convert('RGB')

    fusion_mask = img.convert('L')
      
    
    
    pixels = img_RGB.load()
    # Convert scapho to red
    for i in range(img_RGB.size[0]): # for every pixel:
      for j in range(img_RGB.size[1]):
          red_value = pixels[i,j][0]
          if red_value > 50:
            pixels[i,j] =  colors[index_bone]
    
    img_RGB.putalpha(fusion_mask)
    src1.paste(img_RGB, (0,0), img_RGB)

  src1.save("/content/drive/MyDrive/BoneSegm/results/combined/" + str(index) + ".png")
  """
  img_array_background = y_pred[0, :, :, 0] * 255
  img_array_scapho = y_pred[0, :, :, 1] * 255
  img_array_luna = y_pred[0, :, :, 2] * 255
  fusion_mask_scapho = img_array_scapho / 255.0
  #img_array = np.asarray(img_array)

  img = Image.fromarray(img_array_scapho)

  img_background =  Image.fromarray(img_array_background)
  img_scapho =  Image.fromarray(img_array_scapho * 0.3)
  img_luna =  Image.fromarray(img_array_luna * 0.3)

  

  img_background.convert('L').save("/content/drive/MyDrive/BoneSegm/results/predicted_masks/background" + str(index) + ".png")
  img_scapho.convert('L').save("/content/drive/MyDrive/BoneSegm/results/predicted_masks/scapho" + str(index) + ".png")
  img_luna.convert('L').save("/content/drive/MyDrive/BoneSegm/results/predicted_masks/luna" + str(index) + ".png")
  

  img_background = img_background.convert('RGB')
  img_scapho = img_scapho.convert('RGB')
  img_luna = img_luna.convert('RGB')
  
  fusion_mask_scapho = img_scapho.convert('L')
  fusion_mask_luna = img_luna.convert('L')
  
  pixels = img_scapho.load()
  # Convert scapho to red
  for i in range(img_scapho.size[0]): # for every pixel:
    for j in range(img_scapho.size[1]):
        red_value = pixels[i,j][0]
    
        pixels[i,j] = (red_value, 0 ,0)

  pixels = img_luna.load()
  # Convert scapho to red
  for i in range(img_scapho.size[0]): # for every pixel:
    for j in range(img_scapho.size[1]):
        blue_value = pixels[i,j][2]
    
        pixels[i,j] = (0, 0, blue_value)


 
  img_scapho.putalpha(fusion_mask_scapho)
  img_luna.putalpha(fusion_mask_luna)


  

  

  src1.paste(img_scapho, (0,0), img_scapho)
  src1.paste(img_luna, (0,0), img_luna)


  src1.save("/content/drive/MyDrive/BoneSegm/results/combined/" + str(index) + ".png")
  """



