dl_params = dict(
    dim = (512, 512),
    batch_size = 16,
    n_channels = 1, 
    n_classes = 8, 
    shuffle = True, 
)

training_params = dict(
  epochs = 100,
  loss = "binary_crossentropy",
  optimizer = "rmsprop"  # or "adam" (tf.keras.optimizers.Adam(learning_rate=1e-3))
)

dirs = dict(
  #save_model = "/content/drive/MyDrive/carpal_bones_segmentation/results/models/model1",
  save_model = "/content/drive/MyDrive/carpal_bones_segmentation/results/models/model1",
  image_source = "/tmp/carpal_bones_segmentation/carpal_bones_segmented_incl_pisiforme",
  dict_partition = "/tmp/carpal_bones_segmentation/carpal_bones_segmented_incl_pisiforme/partition",
  dict_labels = "/tmp/carpal_bones_segmentation/carpal_bones_segmented_incl_pisiforme/labels", 
  image_results = "/content/drive/MyDrive/carpal_bones_segmentation/results"
)

credentials = dict(
  ACCESS_KEY = 'AKIASWQJDSISFUUWUTCV',
  SECRET_KEY = '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'
)