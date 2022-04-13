from PIL import Image, ImageOps
import segmentation_models as sm
sm.set_framework('tf.keras')
import numpy as np
import random
import config

def createImageWithMaskLabels(orig_img, masks_np, colors, adjust_to_orig_img_size=False):
  orig_img = orig_img.convert('RGB')
  for index in range(1, masks_np.shape[2]):
    img_array = masks_np[:, :, index] * 255
    img = Image.fromarray(img_array * 0.3)
    if adjust_to_orig_img_size:
      img = img.resize(orig_img.size)
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
  for metric in metrics_list:
    if metric == 'iou_score':
      metrics_func.append(sm.metrics.iou_score)
    elif metric == 'f1_score':
      metrics_func.append(sm.metrics.f1_score)
    else:
      metrics_func.append(metric)
  return metrics_func

def preprocessImagePIL(img, convert_grayscale=True, 
                      dim=(512,512), random_crop_params=None, 
                      autocontrast=True, padding=True, norm=True):
  if convert_grayscale:
    img = img.convert('L')
  
  if autocontrast:
    img = ImageOps.autocontrast(img)

  if random_crop_params != None:
    img = randomCrop(img, random_crop_params)

  if padding:
      img = padImg(img, target_size=dim[0])
  else: 
      img = img.resize(dim)
  
  if norm == True:
    img = np.array(img) / 255
    
  img = np.expand_dims(img, 2)
  return img  

def padImg(img, target_size=512):
    width, height = img.size
    if width == height:
        if width != target_size:
          img = img.resize((target_size, target_size))
        return img
    elif width > height:
        result = Image.new(img.mode, (width, width))
        result.paste(img, (0, 0))
        if width != target_size:
          result = result.resize((target_size, target_size))
        return result
    else:
        result = Image.new(img.mode, (height, height))
        result.paste(img, (0, 0))
        if height != target_size:
          result = result.resize((target_size, target_size))
        return result

def randomCrop(img, random_params):
    x_orig = img.size[0]
    y_orig = img.size[1]

    random_crop_factor = random_params[0]
    crop_width = int(x_orig * random_crop_factor)
    crop_height = int(y_orig * random_crop_factor)

    random_x_pos = int(random_params[1] * (x_orig - crop_width))
    random_y_pos = int(random_params[2] * (y_orig - crop_height))

    img = img.crop((random_x_pos, random_y_pos, random_x_pos + crop_width, random_y_pos + crop_height))
    img = img.resize((x_orig, y_orig))
    return img
  