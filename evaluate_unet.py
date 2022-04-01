import numpy as np
import tensorflow as tf
import json
import keras
from datetime import datetime
from PIL import Image
import config
import model as model_importer
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
                                           **config.dl_eval_params)


parameters_string = config.eval_params['loss']
for metric_func in config.eval_params['metrics']:
  parameters_string = parameters_string + " / " + metric_func

drc_eval_results = config.dirs["results_path"] + "/" + "eval_results.txt"
text_writer = open(drc_eval_results,"a")

date_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
text_writer.write("Results from " + date_string + "\n")
text_writer.write("Eval loss and metrics: " + parameters_string + "\n")

for index, model_name in enumerate(config.eval_params['models_list'], start=0):
  # Option A: load pretraind model
  model = model_importer.getModel(model_name,
                  img_size=config.dl_eval_params['dim'], 
                  num_classes=config.dl_eval_params['n_classes']+1,  
                  num_channels=config.dl_eval_params['n_channels'])
  directory_model = config.dirs["results_path"] + "/models/" + model_name
  model.load_weights(directory_model)

  # Compile model
  loss = getLoss(config.eval_params['loss'])
  metrics = getMetrics(config.eval_params['metrics'])
  model.compile(#optimizer= config.training_params['optimizer'],
                loss=loss,
                metrics=metrics)

  results = model.evaluate(test_generator)
  print("Model: ", model_name, "with Loss, IOU, F1:", results)

  results_string = model_name + ": "
  for data_element in results: 
      results_string =  results_string  + str(data_element) + " / "

  text_writer = open(drc_eval_results,"a")
  text_writer.write(results_string + "\n")
  text_writer.close() 




