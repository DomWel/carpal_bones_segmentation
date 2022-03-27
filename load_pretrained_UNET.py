import numpy as np
import tensorflow as tf
import json
import keras
from PIL import Image
import config

# Datasets
# Opening JSON file
f = open(config.dirs['dict_source'])
data_dicts = json.load(f)

model = keras.models.load_model(config.dirs['model_save'])
#model.summary()
#tf.keras.utils.plot_model(model, to_file="/content/drive/MyDrive/BoneSegm/model.png", show_shapes=True)

# Cretate result images (= oringinal images + predicted masks)
for index, img_path in enumerate(data_dicts['partition']['validation']):
  img_path_complete = '/content/drive/MyDrive/BoneSegm/mask_labels_carpal_bones/' + img_path
  
  img = Image.open(img_path_complete).convert('RGB')
  src1 = img
  img = img.convert('L')

  newsize = (512, 512)
  img = img.resize(newsize)
  img.save("/content/drive/MyDrive/BoneSegm/results/original_images/"+ str(index) + ".png")
  

  img = np.array(img) / 255

  colors = [(255, 0, 0), (0,255,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255), (100, 200, 0), (230, 0, 60)]

  #X = np.empty((self.batch_size, *self.dim, self.n_channels))
  img = np.expand_dims(img, 0)
  y_pred = model.predict(img)

  for index_bone in range(1, y_pred.shape[3]):
    img_array = y_pred[0, :, :, index_bone] * 255
    img = Image.fromarray(img_array * 0.3)
    orig_img_size = src1.size
    img = img.resize(orig_img_size)
    img_RGB = img.convert('RGB')
    fusion_mask = img.convert('L')
    pixels = img_RGB.load()
    # Convert to corresponding color
    for i in range(img_RGB.size[0]): # for every pixel:
      for j in range(img_RGB.size[1]):
          red_value = pixels[i,j][0]
          if red_value > 50:
            pixels[i,j] =  colors[index_bone]
    
    img_RGB.putalpha(fusion_mask)
    src1.paste(img_RGB, (0,0), img_RGB)

  src1.save(config.dirs['image_results'] + "/" + str(index) + ".png")




