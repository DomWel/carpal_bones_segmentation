import numpy as np
import tensorflow as tf
from keras.models import Sequential
from data import DataGeneratorUNET_OHE
import json
import keras
from tensorflow.keras import layers
import config
import model as model_generator

# Dataloader generation
# Opening JSON file
f = open(config.dirs['dict_source'])
data_dicts = json.load(f)
partition = data_dicts['partition']
labels = data_dicts

training_generator = DataGeneratorUNET_OHE(partition['train'], 
                                           labels, 
                                           config.dirs['image_source'], 
                                           **config.dl_params)
validation_generator = DataGeneratorUNET_OHE(partition['validation'], 
                                             labels, 
                                             config.dirs['image_source'],
                                             **config.dl_params)

# Build model
model = model_generator.get_model(config.dl_params['dim'], config.dl_params['n_classes'])
model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss=config.training_params['loss'],
              metrics=[tf.keras.metrics.MeanIoU(num_classes=8)])

# Train model on dataset
model.fit(training_generator,
          validation_data=validation_generator,
          use_multiprocessing=False,
          verbose=1, 
          epochs = config.training_params['epochs'])

model.save(config.dirs['model_save'])
