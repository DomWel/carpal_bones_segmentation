import numpy as np
import tensorflow as tf
from keras.models import Sequential
from data import DataGeneratorUNET_OHE, DataGeneratorUNET_OHE2
import json
import keras
from tensorflow.keras import layers
import config
import model as model_generator
import segmentation_models as sm
sm.set_framework('tf.keras')

sm.framework()

# Dataloader generation
# Opening JSON file

f = open(config.dirs['dict_partition'])
partition = json.load(f)

f = open(config.dirs['dict_labels'])
labels = json.load(f)


training_generator = DataGeneratorUNET_OHE2(partition['train'], 
                                           labels, 
                                           config.dirs['image_source'], 
                                           **config.dl_params)
validation_generator = DataGeneratorUNET_OHE2(partition['validation'], 
                                             labels, 
                                             config.dirs['image_source'],
                                             **config.dl_params)

# Build model
model = model_generator.get_model(config.dl_params['dim'], config.dl_params['n_classes']+1)
#model = sm.Unet('vgg16', input_shape=(None, None, 1), encoder_weights=None, classes=9)
model.summary()
model.compile(optimizer= config.training_params['optimizer'],
              #loss=config.training_params['loss'],
              loss=sm.losses.bce_jaccard_loss,
              metrics=[sm.metrics.iou_score, sm.metrics.f1_score])

#model.save(config.dirs['save_model'])

# Train model on dataset
model.fit(training_generator,
          validation_data=validation_generator,
          use_multiprocessing=False,
          verbose=1, 
          epochs = config.training_params['epochs'])

model.save(config.dirs['save_model'])
