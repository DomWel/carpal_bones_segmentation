import numpy as np
import tensorflow as tf
from keras.models import Sequential
from data import DataGenerator, MNISTDataGenerator, DataGeneratorUNET
import json
from keras_unet.models import vanilla_unet
from keras_unet.metrics import iou, iou_thresholded
import keras
from tensorflow.keras import layers



# Parameters
params = {'dim': (512, 512),
          'batch_size': 16,
          
          'n_channels': 3,
          'shuffle': True}

print('Still in model build up')


# Datasets
# Opening JSON file
f = open('/content/drive/MyDrive/BoneSegm/mask_labels_carpal_bones/dict.json')
data_dicts = json.load(f)

partition = data_dicts['partition']
labels = data_dicts['labels_dict_Os lunatum']
counter = 0
for key in partition['train']: 
  counter = counter +1

print('Elems in Partition train: ', counter)

counter = 0
for key in partition['validation']: 
  counter = counter +1

print('Elems in Partition validation: ', counter)

# Generators
training_generator = DataGeneratorUNET(partition['train'], labels, **params)
validation_generator = DataGeneratorUNET(partition['validation'], labels, **params)



# Build model
#model = vanilla_unet(input_shape=(512, 512, 1))


def get_model(img_size, num_classes):
    inputs = keras.Input(shape=img_size + (3,))

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [256, 128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)

    # Define the model
    model = keras.Model(inputs, outputs)
    return model


# Free up RAM in case the model definition cells were run multiple times
keras.backend.clear_session()

# Build model
model = get_model((512, 512), 2)
model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss='sparse_categorical_crossentropy'
              
)
"""

model.compile(optimizer="rmsprop",
              loss="sparse_categorical_crossentropy",
              metrics=tf.keras.metrics.MeanIoU(2)
)
"""
# Train model on dataset
model.fit(training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=False,
                    verbose=1, epochs = 50)

model.save("/content/drive/MyDrive/BoneSegm/model/model_3")





y_pred = model.predict(validation_generator)