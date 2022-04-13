import numpy as np
import tensorflow as tf
import json
import keras
from PIL import Image
import config
import model as model_importer
from helper_functions import createImageWithMaskLabels, preprocessImagePIL
from pathlib import Path

# Datasets
# Opening JSON file
f = open(config.dirs['dict_partition'])
partition = json.load(f)

for index, model_name in enumerate(config.eval_params['models_list'], start=0):
  # Option A: load pretraind model
  model = model_importer.getModel(model_name,
                  img_size=config.dl_eval_params['dim'], 
                  num_classes=config.dl_eval_params['n_classes']+1,  
                  num_channels=config.dl_eval_params['n_channels'])
  directory_model = config.dirs["results_path"] + "/models/" + model_name
  model.load_weights(directory_model).expect_partial()

  img_drc = config.dirs['results_path'] + "/pred_masks_" + model_name
  print(img_drc)
  Path(img_drc).mkdir(parents=True, exist_ok=True)

  for index, img_path in enumerate(partition['validation']):
    img_path_complete = config.dirs['image_source'] + "/" + img_path
 
    # Load and preprocess image
    img = Image.open(img_path_complete).convert('RGB')
    src1 = img   # Image has to be rgb to draw colored masks in

    # Preprocess image 
    img = preprocessImagePIL(img, 
                          dim = config.dl_eval_params['dim'],
                          autocontrast=config.dl_eval_params['autocontrast'], 
                          random_crop_params = config.dl_eval_params['random_crop_coeff'],
                          padding=config.dl_eval_params['padding']
                          )

    src1 = Image.fromarray(img[:,:,0]*255).convert('RGB')
    img = np.expand_dims(img, axis=0)

    # Get preds
    y_pred = model.predict(img)
    y_pred = y_pred[0,:,:,:]
 
    # Draw predicted masks on original images and save image 
    src1 = createImageWithMaskLabels(src1, y_pred, config.others['color_list'], adjust_to_orig_img_size=False)
    src1.save(img_drc + "/"+str(index)+".png")



