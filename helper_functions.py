from PIL import Image
import segmentation_models as sm
sm.set_framework('tf.keras')

def createImageWithMaskLabels(orig_img, masks_np, colors):
  colors = [(255, 0, 0), (0,255,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255), (100, 200, 0), (230, 0, 60), (60,230,0), (250, 0, 0)]
  orig_img = orig_img.convert('RGB')
  for index in range(1):
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
  