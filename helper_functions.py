from PIL import Image, ImageOps
import segmentation_models as sm
sm.set_framework('tf.keras')
import numpy as np
import random

def createImageWithMaskLabels(orig_img, masks_np, colors):
  orig_img = orig_img.convert('RGB')
  for index in range(1, masks_np.shape[2]):
    img_array = masks_np[:, :, index] * 255
    img = Image.fromarray(img_array * 0.3)
    img_RGB = img.convert('RGB')
    fusion_mask = img.convert('L')
    pixels = img_RGB.load()
    # Convert to corresponding color
    for i in range(img_RGB.size[0]): # for every pixel:
      for j in range(img_RGB.size[1]):
          red_value = pixels[i,j][0]
          if red_value > 50:
            pixels[i,j] =  colors[index]
    img_RGB.putalpha(fusion_mask)
    orig_img.paste(img_RGB, (0,0), img_RGB)
  return orig_img

def getLoss(loss_name):
  if loss_name == "bce_jaccardi_loss":
    loss = sm.losses.bce_jaccard_loss
  else: 
    loss = loss_name
  return loss

def getMetrics(metrics_list):
  metrics_func = []
  if 'iou_score' in metrics_list:
    metrics_func.append(sm.metrics.iou_score)
  if 'f1_score' in metrics_list:
    metrics_func.append(sm.metrics.f1_score)
  return metrics_func

def preprocessImagePIL(img, n_channels=1, dim=(512,512), random_crop_coeff=None, autocontrast=True):
  if n_channels == 1:
    img = img.convert('L')
  
  if autocontrast:
    img = ImageOps.autocontrast(img)

  # Random crop
  random_crop_params=[]
  if random_crop_coeff != None:
    random_crop_size = random.randrange(random_crop_coeff*dim[0], dim[0])
    x1 = random.randrange(0, dim[0] - random_crop_size)
    y1 = random.randrange(0, dim[1] - random_crop_size)
    img = img.crop((x1, y1, x1 + random_crop_size, y1 + random_crop_size))
    random_crop_params=[random_crop_size, x1, y1]

  img = img.resize(dim)
  img = np.array(img) / 255
  img = np.expand_dims(img, 2)
  return img, random_crop_params  # The random crop params are needed to crop mask with the same values

  