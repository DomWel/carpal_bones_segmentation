import numpy as np
import csv, boto3, json, time
import numpy as np
from data import DataGeneratorUNET_OHE
from PIL import Image
import sys 
import config
import requests

# List of colors to draw in segmentation mask
colors = [(255, 0, 0), (0,255,0), (0,0,255), (255,255,0), 
          (255,0,255), (0,255,255), (100, 200, 0), (230, 0, 60),
          (34, 78, 150), (45,15,160)]

sm_rt = boto3.client(service_name='sagemaker-runtime', 
                     region_name='eu-west-1', 
                     aws_access_key_id=config.credentials['ACCESS_KEY'],
                     aws_secret_access_key=config.credentials['SECRET_KEY']
                     )

def get_prediction(img):
    img = img.tolist()
    request = {               
                    "inputs": [img]
    }

    data = json.dumps(request)
    tic = time.time()
     
    response = sm_rt.invoke_endpoint(
                EndpointName='unet-carpal-bones-serverless-ep-2022-03-29-16-07-40',
                Body=data,
                ContentType='application/json'
    )

    tac = time.time()
    print("Total server response time: ", tac - tic)

    response = response["Body"].read()
    response = json.loads(response)
    return response


def get_rest_url(model_name, host="ec2-54-76-152-56.eu-west-1.compute.amazonaws.com", port="8501", verb="predict"):  
     url = "http://{0}:{1}/v1/models/{2}:predict".format(host,port,model_name)
     
     return url
     
def rest_request(data, url):
    payload = json.dumps({"instances": [data]})
    response = requests.post(url=url, data=payload)
    return response

f = open('/content/drive/MyDrive/BoneSegm/mask_labels_carpal_bones/dict.json')
data_dicts = json.load(f)

partition = data_dicts['partition']


for index, img_path in enumerate(data_dicts['partition']['validation']):
  img_path_complete = '/content/drive/MyDrive/BoneSegm/mask_labels_carpal_bones/' + img_path
  img = Image.open(img_path_complete).convert('RGB')
  src1 = img
  img = img.convert('L')
  newsize = (512, 512)
  img = img.resize(newsize)
  img.save("/content/drive/MyDrive/BoneSegm/results_endpoint/original_images/"+ str(index) + ".png")  
  img = np.array(img) / 255
  img = np.expand_dims(img, 2)
 
  print(img.shape)

  #y_pred_dict = get_prediction(img)
  url = get_rest_url("model")
  
  
  img = img.tolist()
  request = {               
                  "inputs": [img]
  }


  #data = json.dumps(request)
  tic = time.time()
  response = rest_request(img, url)
  toc = time.time()
  print("Total server response time: ", toc - tic)
  response = response.json()


  
  y_preds = response['predictions']




  y_preds =  np.asarray(y_preds)
  
  for index_bone in range(1, y_preds.shape[3]):
    img_array = y_preds[0, :, :, index_bone] * 255

    img = Image.fromarray(img_array * 0.3)
    orig_img_size = src1.size
    img = img.resize(orig_img_size)
    img_RGB = img.convert('RGB')
    fusion_mask = img.convert('L')      
    pixels = img_RGB.load()

    for i in range(img_RGB.size[0]): 
      for j in range(img_RGB.size[1]):
          red_value = pixels[i,j][0]
          if red_value > 50:
            pixels[i,j] =  colors[index_bone]
    
    img_RGB.putalpha(fusion_mask)
    src1.paste(img_RGB, (0,0), img_RGB)

  src1.save(config.dirs['image_results'] + "/" + str(index) + ".png")
  

