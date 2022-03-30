import numpy as np
import csv, boto3, json, time
import numpy as np
from data import DataGeneratorUNET_OHE
from PIL import Image
import sys 
import config
import requests

def getPredictionFromSagemakerEndpoint(img):
    # Connect to client for endpoint inference
    sm_rt = boto3.client(service_name=config.sagemaker_endpoint['service_name'], 
                      region_name=config.sagemaker_endpoint['region_name'], 
                      aws_access_key_id=config.sagemaker_endpoint['ACCESS_KEY'],
                      aws_secret_access_key=config.sagemaker_endpoint['SECRET_KEY']
                      )

    # JSON serialize img data
    ## Using this method is difficult, serverless endpoint only accepts 
    ## payloads of maximum 5 mb size
    img = img.tolist()
    request = {"inputs": [img]}
    data = json.dumps(request)

    # Send request and measure response time
    tic = time.time()  
    response = sm_rt.invoke_endpoint(
                EndpointName=config.sagemaker_endpoint['EndpointName'],
                Body=data,
                ContentType=config.sagemaker_endpoint['ContentType']
    )
    tac = time.time()
    print("Total server response time: ", tac - tic)

    # Read response 
    response = response["Body"].read()
    response = json.loads(response)
    y_preds = response["ouputs"]

    return y_preds


def getPredictionFromEC2(img):
  # Create url for request
  model_name = config.server_ec2["model_name"]
  host = config.server_ec2["host"]
  port= config.server_ec2["port"]
  url = "http://{0}:{1}/v1/models/{2}:predict".format(host,port,model_name)

  # Create request data
  img = img.tolist()
  payload = json.dumps({"instances": [img]})

  # Send request and measure response time
  tic = time.time()
  response = requests.post(url=url, data=payload)
  tac = time.time()
  print("Total server response time: ", tac - tic)

  # Read server response
  response = response.json()
  y_preds = response['predictions']

  return y_preds


f = open(config.dirs["dict_partition"])
partition = json.load(f)

for index, img_path in enumerate(partition['validation']):
  img_path_complete = config.dirs['image_source'] + img_path
  img = Image.open(img_path_complete).convert('RGB')
  src1 = img
  img = img.convert('L')
  newsize = (512, 512)
  img = img.resize(newsize)
  img = np.array(img) / 255
  img = np.expand_dims(img, 2)
 
  y_preds = getPredictionFromEC2(img)
  # or: y_preds = getPredictionFromSagemakerEndpoint(img) if sagemaker endppint is up and running
  y_preds =  np.asarray(y_preds)
  
  # Paste the predicted masks into the original image, class0 = background is left out 
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
            pixels[i,j] =  config.others['color_list'][index_bone]
    
    img_RGB.putalpha(fusion_mask)
    src1.paste(img_RGB, (0,0), img_RGB)

  src1.save(config.dirs['image_results'] + "/predicted_mask_for_" + str(img_path))
  

