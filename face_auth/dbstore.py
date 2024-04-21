from pymongo import MongoClient
import numpy as np,torch
import cv2 as cv
import os
import imagehash
from PIL import Image

def set_variables(a, b):
    global path, user_name
    path, user_name = a, b

def see_variable():
    global path, user_name
    return path, user_name
def database_storage(port_number=27017):
    client = MongoClient(port=port_number)
    db = client.image_data
    col = db[user_name]
    tpath = {'train': os.path.join(path+'\\data',user_name+'\\train\\'+user_name),
             'test': os.path.join(path+'\\data',user_name+'\\test\\'+user_name)}
    elem = {'train': os.listdir(tpath['train']), 'test': os.listdir(tpath['test'])}
    for i in ['train', 'test']:
        for j in elem[i]:
            im = cv.imread(os.path.join(tpath[i], j))
            hash = imagehash.average_hash(Image.fromarray(im))
            col.insert_one({'filename': str(j), 'kind': i, 'data': str(hash)})
    client.close()
    return 'success'

def model_storage(port_number=27017):
    client = MongoClient(port=port_number)
    db = client.image_data
    col = db['models']
    a =np.array(torch.load(os.path.join(path+'\\data\\'+user_name,user_name+"_best_model_params.pt")))
    col.insert_one({'filename':user_name+'_model','user_name':user_name,'model_data':a.tobytes()})
    client.close()
    return 'success'

