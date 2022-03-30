import numpy as np
import tensorflow as tf
import json
import keras
from PIL import Image
import config
import model
import segmentation_models as sm
sm.set_framework('tf.keras')

from get_server_inference import getPredictionFromSagemakerEndpoint, getPredictionFromEC2
from helper_functions import createImageWithMaskLabels

# Datasets
# Opening JSON file
f = open(config.dirs['dict_partition'])
partition = json.load(f)

# Option A: load pretraind model
model = model.getModel(config.training_params['model_name'],
                 img_size=config.dl_params['dim'], 
                 num_classes=config.dl_params['n_classes']+1,  # Background has to be added as additional  class
                 num_channels=config.dl_params['n_channels'])

model.load_weights(config.dirs['save_model'])
get_predict = model.predict


# Option B: Get server inference from deployed model
# get_predict = getPredictionFromSagemakerEndpoint
# get_predict = getPredictionFromEC2


for index, img_path in enumerate(partition['validation']):
  img_path_complete = config.dirs['image_source'] + "/" + img_path
  
  # Load and preprocess image
  img = Image.open(img_path_complete).convert('RGB')
  src1 = img
  img = img.convert('L')
  newsize = config.dl_params["dim"]
  img = img.resize(newsize)
  img = np.array(img) / 255
  img = np.expand_dims(img, 0)

  # Get preds
  y_pred = model.predict(img)

  # Draw predicted masks on original images and save image 
  src1 = createImageWithMaskLabels(src1, y_pred)
  src1.save(config.dirs['image_results'] + "/predicted_mask_" + str(img_path))


  """
  for index_bone in range(1, y_pred.shape[3]):
    img_array = y_pred[0,:, :, index_bone] * 255
    img = Image.fromarray(img_array * 0.3)
    orig_img_size = src1.size
    img = img.resize(orig_img_size)
    img_RGB = img.convert('RGB')
    fusion_mask = img.convert('L')
    pixels = img_RGB.load()
    # Convert to corresponding color
    for i in range(img_RGB.size[0]): # for every pixel:
      for j in range(img_RGB.size[1]):
          red_value = pixels[i,j][0]
          if red_value > 50:
            pixels[i,j] =  colors[index_bone]
    
    img_RGB.putalpha(fusion_mask)
    src1.paste(img_RGB, (0,0), img_RGB)
  """

