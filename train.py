from data import DataGeneratorUNET_OHE2
import json
import config
import model
from helper_functions import getLoss, getMetrics


# Load dataset dict files
f = open(config.dirs['dict_partition'])
partition = json.load(f)

f = open(config.dirs['dict_labels'])
labels = json.load(f)

# Create dataloaders 
training_generator = DataGeneratorUNET_OHE2(partition['train'], 
                                           labels, 
                                           config.dirs['image_source'], 
                                           **config.dl_params)
validation_generator = DataGeneratorUNET_OHE2(partition['validation'], 
                                             labels, 
                                             config.dirs['image_source'],
                                             **config.dl_params)

# Build model
model = model.getModel(config.training_params['model_name'],
                 img_size=config.dl_params['dim'], 
                 num_classes=config.dl_params['n_classes']+1,  # Background has to be added as additional  class
                 num_channels=config.dl_params['n_channels'])
model.summary()

# Compile model
loss = getLoss(config.training_params['loss'])
metrics = getMetrics(config.training_params['metrics'])
model.compile(optimizer= config.training_params['optimizer'],
              loss=loss,
              metrics=metrics)

# Train model on dataset
model.fit(training_generator,
          validation_data=validation_generator,
          use_multiprocessing=False,
          verbose=1, 
          epochs = config.training_params['epochs'])

# Save model in protobuff format
model.save(config.dirs['save_model'])
