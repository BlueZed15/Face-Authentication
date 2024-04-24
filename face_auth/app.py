import os
import torch
from torchvision import transforms
import cv2 as cv
from flask import Flask,jsonify
import face_auth,dbstore,train_user

app=Flask(__name__)

@app.route("/capture/<user_name>",methods={'GET'})
def capture(user_name):
    face_auth.set_variables(r"\\",user_name)
    face_auth.cap()
    return jsonify({'message':'capture successful'}),200


@app.route("/train_face_model/<user_name>",methods={'GET'})
def train(user_name):
    train_user.set_variables(r"\\", user_name)
    train_user.train_face_model()
    dbstore.set_variables(r"\\", user_name)
    dbstore.database_storage()
    dbstore.model_storage()
    return jsonify({'message_1':'model training successful for '+user_name,
                    'message_2':'data and model stored in mongodb'}),200


@app.route("/authenticate/<user_name>",methods={'GET'})
def classify_face(user_name):
    global check
    check=False
    def patimg(event, x, y, flags, param):
        global check
        if event == cv.EVENT_LBUTTONDOWN:
            check = True
        if event == cv.EVENT_RBUTTONDOWN:
            check = False

    transformer_classify = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    path=r"\\"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model= torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 1)
    model=model.to(device)

    vid = cv.VideoCapture(0)
    winame = 'Face Authentication'
    while True:
        _, image= vid.read()
        ima = image.copy()
        cv.namedWindow(winame, cv.WINDOW_NORMAL)
        cv.resizeWindow(winame, 800, 600)
        cv.rectangle(image, (175, 400), (500, 75), color=(255, 255, 255), thickness=1)
        ima = ima[75:400, 175:500]
        cv.putText(image, text="Position your face inside the frame", lineType=cv.FILLED, color=(255, 200, 170),
                   thickness=2, fontScale=1, org=(50, 30), fontFace=cv.FONT_HERSHEY_SIMPLEX)
        cv.setMouseCallback(winame,patimg)
        cv.imshow(winame,image)
        if check == True:
            break
        if cv.waitKey(10) == (ord('E') | ord('e')):
            break
    check=False
    vid.release()
    cv.destroyAllWindows()

    elem=os.listdir(path+'\\data\\models')
    cnt=0
    for i in elem:
        uname=i.split('_')[0]
        best_model_params_path = os.path.join(path+'\\data\\models',i)
        model.load_state_dict(torch.load(best_model_params_path))
        model.eval()
        im = transformer_classify(ima).unsqueeze(0)
        with torch.no_grad():
            output = model(im)
            probabilities = torch.nn.functional.sigmoid(output[0])
        op = 'true' if probabilities[0] > 0.70 else 'false'
        cnt+=1
        if op=='true':
            return jsonify({'user_name':uname,'message':'authenticated'}),200
        if cnt==len(elem):
            return jsonify({'message':'not authenticated'})


if __name__=='__main__':
    app.run(debug=True)


