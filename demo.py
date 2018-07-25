import argparse
import matplotlib.pyplot as plt

import chainer
import cv2 as cv
import numpy as np
import os
import pickle

from chainercv.datasets import voc_bbox_label_names
from chainercv.links import FasterRCNNVGG16
from chainercv import utils
from chainercv.visualizations import vis_bbox
from chainer import serializers
import chainer.links as L

import Mynet
import VGG_chainer

def unpickle(file):
    # file.decode('utf-8')
    fo = open(file, 'rb')
    # fo.decode('utf-8')
    dict = pickle.load(fo,encoding="latin1")
    fo.close()
    return dict

def get_cifar100(folder):
    test_fname  = os.path.join(folder,'test')
    data_dict = unpickle(test_fname)
    test_data = data_dict['data']
    test_fine_labels = data_dict['fine_labels']
    test_coarse_labels = data_dict['coarse_labels']

    bm = unpickle(os.path.join(folder, 'meta'))
    clabel_names = bm['coarse_label_names']
    flabel_names = bm['fine_label_names']
    return test_data, np.array(test_coarse_labels), np.array(test_fine_labels), clabel_names, flabel_names

def to_matplotlib_format(img):
    return cv.cvtColor(img, cv.COLOR_BGR2RGB)

def predict(model, x_data):
    x_data = np.reshape(x_data,(1,3,32,32))
    if(x_data.dtype!='float32'):
        x_data = x_data.astype(np.float32)
        x_data = x_data/256
    #x = chainer.Variable(x_data.astype(np.float32))
    y = model.predictor(x_data)
    return np.argmax(y.data, axis = 1), np.max(y.data, axis = 1)

def diff_frame(video,model,flabels):
    fgbg = cv.createBackgroundSubtractorKNN()
    cap = cv.VideoCapture(video)
    
    font = cv.FONT_HERSHEY_PLAIN

    while(1):
        #動画からフレーム取得
        ret, frame = cap.read()
        frame2 = to_matplotlib_format(frame)
        frame3 = frame
        #胴体検知のマスクをかける
        fgmask = fgbg.apply(frame)
        fgmask = cv.GaussianBlur(fgmask,(17,17),0)#GaussianBlurをかけることで，細かいコンタ成分を消すことができる
        ret,thresh = cv.threshold(fgmask,127,255,cv.THRESH_BINARY)#閾値を入れることで，2値化する
        cv.imshow('thresh',thresh)
        #contourを検出する
        image, contours, hierarchy = cv.findContours(thresh,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
        #img_coutor = cv.drawContours(frame, contours, -1, (255,0,0), 3)#全コンタを青で描く
        img_rect = frame3
        #cv.imshow('contours',img_coutor)
        detected = []
        for c in contours:
            if cv.contourArea(c) < 200:
                continue
            # rectangle area
            x, y, w, h = cv.boundingRect(c)
            # crop the image
            # cropped = forcrop[y:(y + h), x:(x + w)]
            # cropped = resize_image(cropped, (210, 210))
            # crops.append(cropped)
            # draw contour
            part = frame2[y:(y+h),x:(x+w)]
            detected.append(part)
            img_rect = cv.rectangle(img_rect, (x, y), (x + w, y + h), (0, 255, 0), 3)  #rectangle contour
            part = cv.resize(part,(32,32))
            part = to_matplotlib_format(part)
            ans, val = predict(model,part)
            cv.putText(img_rect,flabels[ans[0]],(x-5,y-5),font,1,(255,255,0))
        cv.imshow('rectangle',img_rect)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cv.destroyAllWindows()
           
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu',type=int, default=-1)
    parser.add_argument('--pretrained-model',default='voc07')
    parser.add_argument('video')
    args = parser.parse_args()
    if args.pretrained_model == 'trained_model':
        #model = L.Classifier(Mynet.MyNet(100))
        model = L.Classifier(VGG_chainer.VGG(100))
        serializers.load_npz('trained_model',model)
        print('VGG is defined')
    else:
        model = FasterRCNNVGG16(
            n_fg_class=len(voc_bbox_label_names),
            pretrained_model=args.pretrained_model)
#    model = L.Classifier(VGG_chainer.VGG(100))
#    serializers.load_npz('trained_model',model) 
#    print('VGG is defined')
    data_path = "./cifar-100-python"
    test_data, test_clabels, test_flabels, clabels,flabels = get_cifar100(data_path)
    diff_frame(args.video,model,flabels)


if __name__ =='__main__':
    main()
