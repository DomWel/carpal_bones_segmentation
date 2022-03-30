dl_params = dict(
    dim = (512, 512),
    batch_size = 16,
    n_channels = 1, 
    n_classes = 8, 
    shuffle = True, 
)

training_params = dict(
  epochs = 100,
  loss = "bce_jaccardi_loss", # or: "binary_crossentropy"
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
  ACCESS_KEY = 'xxxxxxxxxxxxxx',
  SECRET_KEY = 'xxxxxxxxxxxxxx'
)

server_ec2 = dict(
  host = "ec2-54-76-152-56.eu-west-1.compute.amazonaws.com",
  port="8501",
  model_name ="model"
)

sagemaker_endpoint = dict(
  service_name='sagemaker-runtime',
  region_name='eu-west-1', 
  EndpointName='unet-carpal-bones-serverless-ep-2022-03-29-16-07-40',
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