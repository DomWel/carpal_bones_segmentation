dl_params = dict(
    dim = (512, 512),
    batch_size = 16,
    n_channels = 1, 
    n_classes = 8, 
    shuffle = True, 
)

training_params = dict(
  epochs = 50,
  loss = "binary_crossentropy"
)

dirs = dict(
  save_model = "./model",
  image_source = "/content/drive/MyDrive/BoneSegm/mask_labels_carpal_bones/",
  dict_source = "/content/drive/MyDrive/BoneSegm/mask_labels_carpal_bones/dict.json",
  image_results = "/content/drive/MyDrive/carpal_bones_segmentation/results"
)

credentials = dict(
  ACCESS_KEY = 'AKIASWQJDSISFUUWUTCV',
  SECRET_KEY = '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'
)