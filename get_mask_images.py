import numpy as np
import tensorflow as tf
import json
import keras
from PIL import Image
import config
import model
import segmentation_models as sm
sm.set_framework('tf.keras')
from data import DataGeneratorUNET_OHE2
from helper_functions import getLoss, getMetrics
from get_server_inference import getPredictionFromSagemakerEndpoint, getPredictionFromEC2
from helper_functions import createImageWithMaskLabels, preprocessImagePIL

# Datasets
# Opening JSON file
f = open(config.dirs['dict_partition'])
partition = json.load(f)
f = open(config.dirs['dict_labels'])
labels = json.load(f)
# Create dataloaders 
test_generator = DataGeneratorUNET_OHE2(partition['validation'], 
                                           labels, 
                                           config.dirs['image_source'], 
                                           **config.dl_params)


# Option A: load pretraind model
model = model.getModel(config.training_params['model_name'],
                 img_size=config.dl_params['dim'], 
                 num_classes=config.dl_params['n_classes']+1,  # Background has to be added as additional  class
                 num_channels=config.dl_params['n_channels'])

model.load_weights(config.dirs['save_model'])
# Compile model
loss = getLoss(config.training_params['loss'])
metrics = getMetrics(config.training_params['metrics'])
model.compile(optimizer= config.training_params['optimizer'],
              loss=loss,
              metrics=metrics)
results = model.evaluate(test_generator)
print("Loss, IOU, F1:", results)


def get_predict(img):
  return model.predict(img)

# Option B: Get server inference from deployed model
# get_predict = getPredictionFromSagemakerEndpoint
# get_predict = getPredictionFromEC2

for index, img_path in enumerate(partition['validation']):
  img_path_complete = config.dirs['image_source'] + "/" + img_path
  
  # Load and preprocess image
  img = Image.open(img_path_complete).convert('RGB')
  src1 = img   # Image has to be rgb to draw colored masks in

  # Preprocess image 
  img, __ = preprocessImagePIL(img, 
                        n_channels= config.dl_params['n_channels'],
                        dim = config.dl_params['dim'],
                        autocontrast=True, 
                        random_crop_coeff = None
                        )
  
  img = np.expand_dims(img, axis=0)
  print(img.shape)

  # Get preds
  y_pred = model.predict(img)

  print(y_pred.shape)
  y_pred = y_pred[0,:,:,:]
  print(y_pred.shape)
  # Draw predicted masks on original images and save image 
  src1 = createImageWithMaskLabels(src1, y_pred, config.others['color_list'], adjust_to_orig_img_size=True)
  src1.save(config.dirs['image_results'] + "/predicted_mask_" + str(img_path))



