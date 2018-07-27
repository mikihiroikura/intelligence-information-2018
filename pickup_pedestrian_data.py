# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 09:31:33 2018

@author: Mikihiro Ikura
"""

import argparse
import matplotlib.pyplot as plt

import cv2 as cv
import numpy as np
import os
import pickle
import scipy as sp

def to_matplotlib_format(img):
    return cv.cvtColor(img, cv.COLOR_BGR2RGB)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('video')
    args = parser.parse_args()
    
    fgbg = cv.createBackgroundSubtractorKNN()
    cap = cv.VideoCapture(args.video)
    cnt = 1
    while(1):
        ret,frame = cap.read()
        frame2 = to_matplotlib_format(frame)
         #胴体検知のマスクをかける
        fgmask = fgbg.apply(frame)
        fgmask = cv.GaussianBlur(fgmask,(17,17),0)#GaussianBlurをかけることで，細かいコンタ成分を消すことができる
        ret,thresh = cv.threshold(fgmask,127,255,cv.THRESH_BINARY)#閾値を入れることで，2値化する
        cv.imshow('thresh',thresh)
        #contourを検出する
        image, contours, hierarchy = cv.findContours(thresh,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
        
        for c in contours:
            if cv.contourArea(c) < 200:
                continue
            # rectangle area
            x, y, w, h = cv.boundingRect(c)
            part = frame2[y:(y+h),x:(x+w)]
            path = './dataset/from_vtest_200/vtest_'+str(cnt)+'.jpg'
            sp.misc.imsave(path, part)
            cnt +=1
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cv.destroyAllWindows()
    

if __name__ =='__main__':
    main()