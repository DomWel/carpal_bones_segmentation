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




  

