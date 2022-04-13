dl_train_params = dict(
    dim = (512, 512),
    batch_size = 16,
    n_channels = 1, 
    n_classes = 8, 
    shuffle = True, 
    random_crop_coeff = None,  # Turn off random crop: random_crop_coeff = None
    autocontrast = True,
    padding = True          # If false images will be resized to "dim" without regard of the original ratio
)

dl_eval_params = dict(
    dim = (512, 512),
    batch_size = 16,
    n_channels = 1, 
    n_classes = 8, 
    shuffle = False, 
    random_crop_coeff = None,  # Turn off random crop: random_crop_coeff = None
    autocontrast = True, 
    padding=True
)

training_params = dict(
  models_list = ["inceptionv3", "vgg16", "resnet18", "custom_model_from_keras_examples"],
  epochs = 100,
  loss = "bce_jaccardi_loss", # or: "binary_crossentropy"
  optimizer = "rmsprop",  # or "adam" (tf.keras.optimizers.Adam(learning_rate=1e-3))
  metrics = ["iou_score", "f1_score"]
)

eval_params = dict(
  models_list = ["inceptionv3", "vgg16", "resnet18", "custom_model_from_keras_examples"],
  loss = "bce_jaccardi_loss", # or: "binary_crossentropy"
  metrics = ["iou_score", "f1_score"]
)

dirs = dict(
  results_path = "/content/drive/MyDrive/carpal_bones_segmentation/results_wo_rc_padding",
  image_source = "/tmp/carpal_bones_segmentation/corrected_dataset",
  dict_partition = "/tmp/carpal_bones_segmentation/corrected_dataset/partition",
  dict_labels = "/tmp/carpal_bones_segmentation/corrected_dataset/labels", 
)

server_ec2 = dict(
  host = "XXXXXXXXXXXXX",
  port="8501",
  model_name ="model"
)

sagemaker_endpoint = dict(
  service_name='sagemaker-runtime',
  region_name='eu-west-1', 
  EndpointName='XXXXXXXXXXXX',
  ContentType='application/json',
  ACCESS_KEY = 'xxxxxxxxxxxxxx',
  SECRET_KEY = 'xxxxxxxxxxxxxx'
)

others = dict(
  # List of colors to draw in segmentation mask
  color_list = [(255, 0, 0), (0,255,0), (0,0,255), (255,255,0), 
          (255,0,255), (0,255,255), (100, 200, 0), (230, 0, 60),
          (34, 78, 150), (45,15,160)]
)