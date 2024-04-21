import os
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import time

check = False
def patimg(event, x, y, flags, param):
    global check
    if event == cv.EVENT_LBUTTONDOWN:
        check = True
    if event == cv.EVENT_RBUTTONDOWN:
        check = False

def sav(ima, cnt, spath, uname):
    time.sleep(0.2)
    cv.imwrite(os.path.join(spath, uname) + '\\train\\' +uname+'\\' + str(cnt) + ".png", ima)
    print('success')

def set_variables(a, b):
    global path, user_name
    path, user_name = a, b

def see_variable():
    global path, user_name
    return path, user_name

def cap():
    global check
    vid = cv.VideoCapture(0)
    winame = 'Face Detector'

    if not os.path.exists(os.path.join(path, 'data')):
        os.mkdir(os.path.join(path, 'data'))
    if not os.path.exists(os.path.join(path, 'data\\' + user_name)):
        os.mkdir(os.path.join(path, 'data\\'+user_name))
    save_path = os.path.join(path, 'data')
    if not os.path.exists(os.path.join(save_path, user_name+'\\train')):
        os.mkdir(os.path.join(save_path,user_name+'\\train'))
    if not os.path.exists(os.path.join(save_path, user_name+'\\train\\'+user_name)):
        os.mkdir(os.path.join(save_path,user_name+'\\train\\'+user_name))
    cnt = 0

    while True:
        _, im = vid.read()
        ima = im.copy()
        cv.namedWindow(winame, cv.WINDOW_NORMAL)
        cv.resizeWindow(winame, 800, 600)
        cv.rectangle(im, (175, 400), (500, 75), color=(255, 255, 255), thickness=1)
        ima = ima[75:400, 175:500]
        cv.putText(im, text="Position your face inside the frame", lineType=cv.FILLED, color=(255, 200, 170),
                   thickness=2, fontScale=1, org=(50, 30), fontFace=cv.FONT_HERSHEY_SIMPLEX)
        cv.setMouseCallback(winame,patimg)
        cv.imshow(winame,im)
        if check == True:
            sav(ima, cnt, save_path, user_name)
            cnt+=1
        if cv.waitKey(10) == (ord('E') | ord('e')):
            check=False
            break
    vid.release()
    cv.destroyAllWindows()

    train_path = os.path.join(save_path, user_name + '\\train\\')
    test_path = os.path.join(save_path, user_name + '\\test\\')

    if not os.path.exists(os.path.join(save_path, user_name+'\\test')):
        os.mkdir(os.path.join(save_path,user_name+'\\test'))
    if not os.path.exists(test_path+'\\'+user_name):
        os.mkdir(test_path+'\\'+user_name)
    im_elem = os.listdir(train_path+'\\'+user_name)
    b=np.random.default_rng().choice(range(len(im_elem)), size=round(len(im_elem) * 0.2), replace=False)
    for i in b:
        print(os.path.join(train_path + user_name, str(i) + '.png'))
        os.rename(os.path.join(train_path+user_name,str(i) + '.png'),
                  os.path.join(test_path+user_name, str(i) + '.png'))


