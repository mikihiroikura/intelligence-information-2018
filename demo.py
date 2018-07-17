import argparse
import matplotlib.pyplot as plt

import chainer
import cv2 as cv
import numpy as np

from chainercv.datasets import voc_bbox_label_names
from chainercv.links import FasterRCNNVGG16
from chainercv import utils
from chainercv.visualizations import vis_bbox
from chainer import serializers
import chainer.links as L

import Mynet

def diff_frame(video):
    fgbg = cv.createBackgroundSubtractorKNN()
    cap = cv.VideoCapture(video)

    while(1):
        #動画からフレーム取得
        ret, frame = cap.read()
        #胴体検知のマスクをかける
        fgmask = fgbg.apply(frame)
        fgmask = cv.GaussianBlur(fgmask,(17,17),0)#GaussianBlurをかけることで，細かいコンタ成分を消すことができる
        ret,thresh = cv.threshold(fgmask,127,255,cv.THRESH_BINARY)#閾値を入れることで，2値化する
        cv.imshow('thresh',thresh)
        #contourを検出する
        image, contours, hierarchy = cv.findContours(thresh,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
        img_coutor = cv.drawContours(frame, contours, -1, (255,0,0), 3)#全コンタを青で描く
        img_rect = img_coutor
        cv.imshow('contours',img_coutor)
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
            img_rect = cv.rectangle(img_rect, (x, y), (x + w, y + h), (0, 255, 0), 3)  #rectangle contour
            part = frame[y:(y+h),x:(x+w)]
            detected.append(part)
        cv.imshow('rectangle',img_rect)
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break

def diff_frame2(video,model):
    fgbg = cv.createBackgroundSubtractorKNN()
    cap = cv.VideoCapture(video)

    while(1):
        #動画からフレーム取得
        ret, frame = cap.read()
        bboxes, labels, scores = model.predict([frame])
        bbox, label, score = bboxes[0], labels[0], scores[0]

        vis_bbox(
            frame, bbox, label, score, label_names=voc_bbox_label_names)
        cv.imshow('frame',frame)
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu',type=int, default=-1)
    parser.add_argument('--pretrained-model',default='voc07')
    parser.add_argument('video')
    args = parser.parse_args()
    if args.pretrained_model == 'trained_model':
        model = L.Classifier(Mynet.MyNet(100))
        serializers.load_npz('trained_model',model)
        print('MyNet is defined')
    else:
        model = FasterRCNNVGG16(
            n_fg_class=len(voc_bbox_label_names),
            pretrained_model=args.pretrained_model)
    diff_frame(args.video)


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--gpu',type=int, default=-1)
#     parser.add_argument('image1')
#     parser.add_argument('image2')
#     args = parser.parse_args()
#     diff_frame(parser.image1)


if __name__ =='__main__':
    main()
