import numpy as np
import csv, boto3, json, time
import numpy as np
from data import DataGenerator, MNISTDataGenerator, DataGeneratorUNET, DataGeneratorUNET_OHE
from PIL import Image
import sys 

ACCESS_KEY = 'AKIASWQJDSISFUUWUTCV'
SECRET_KEY = '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'

sm_rt = boto3.client(service_name='sagemaker-runtime', 
                     region_name='eu-west-1', 
                     aws_access_key_id=ACCESS_KEY,
                     aws_secret_access_key=SECRET_KEY
                     )

def get_prediction(img):

    print("Img shape: ", img.shape)

    img = img.tolist()

    request = {               
                    "inputs": [img]
    }

    data = json.dumps(request)

    print('Byte size', sys.getsizeof(data))

    tic = time.time()
     
    response = sm_rt.invoke_endpoint(
                EndpointName='tensorflow-inference-2022-03-13-20-43-50-992',
                Body=data,
                ContentType='application/json'
    )

    tac = time.time()
    print("Total server response time: ", tac - tic)

    response = response["Body"].read()
    response = json.loads(response)

    return response

# Load img
# Datasets
# Opening JSON file
f = open('/content/drive/MyDrive/BoneSegm/mask_labels_carpal_bones/dict.json')
data_dicts = json.load(f)

partition = data_dicts['partition']
labels = data_dicts['labels_dict_Os lunatum']



for index, img_path in enumerate(data_dicts['partition']['validation']):
  img_path_complete = '/content/drive/MyDrive/BoneSegm/mask_labels_carpal_bones/' + img_path
  img_path_mask = '/content/drive/MyDrive/BoneSegm/mask_labels_carpal_bones/' + labels[img_path] 
  
  
  img = Image.open(img_path_complete).convert('RGB')
  src1 = img
  img = img.convert('L')

  
  newsize = (512, 512)
  img = img.resize(newsize)
  img.save("/content/drive/MyDrive/BoneSegm/results_endpoint/original_images/"+ str(index) + ".png")



  colors = [(255, 0, 0), (0,255,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255), (100, 200, 0), (230, 0, 60)]

  #X = np.empty((self.batch_size, *self.dim, self.n_channels))
  
  img = np.array(img) / 255
  img = np.expand_dims(img, 2)
 
  print("Shape img: ", img.shape)

  y_pred_dict = get_prediction(img)

  y_preds = y_pred_dict['outputs']
  y_preds =  np.asarray(y_preds)

  print('Y_preds shape: ', y_preds.shape)
  
  for index_bone in range(1, y_preds.shape[3]):
    img_array = y_preds[0, :, :, index_bone] * 255

    print(np.max(img_array), np.min(img_array)) 

    img = Image.fromarray(img_array * 0.3)
    orig_img_size = src1.size
    img = img.resize(orig_img_size)
    img_RGB = img.convert('RGB')

    fusion_mask = img.convert('L')
      
    fusion_mask.save("/content/drive/MyDrive/BoneSegm/results/predicted_masks/" + str(index) + "_" + str(index_bone) + ".png")
    
    
    pixels = img_RGB.load()
    # Convert scapho to red
    for i in range(img_RGB.size[0]): # for every pixel:
      for j in range(img_RGB.size[1]):
          red_value = pixels[i,j][0]
          if red_value > 50:
            pixels[i,j] =  colors[index_bone]
    
    img_RGB.putalpha(fusion_mask)
    src1.paste(img_RGB, (0,0), img_RGB)

  src1.save("/content/drive/MyDrive/BoneSegm/results/combined/" + str(index) + ".png")
  


#response_dict = get_prediction(img)


#preds = response_dict['outputs']
#preds =  np.asarray(preds)
#idx = np.argmax(preds[0])
