import argparse


import chainer
import cv2 as cv
import numpy as np
from chainer import serializers
import chainer.links as L
from PIL import Image

import VGG_chainer

def to_matplotlib_format(img):
    return cv.cvtColor(img, cv.COLOR_BGR2RGB)

#CNNモデルでの学習結果出力関数
#引数：1．CPU学習モデル　２．Matplotlib用のnumpy.ndarray uint8
#戻り値：argmax(y),max(y) y:CNNでの出力ベクトル
def predict(model, x_data):
    pilImg = Image.fromarray(np.uint8(x_data))#PIL dataに変換
    img_resize = pilImg.resize((32, 32))#画像サイズ変換32*32*3
    imgArray = np.asarray(img_resize)#numpy ndarrayに変換
    imgArray2 = imgArray.astype(np.float32)/256#float32の配列に変換 256で割る
    img_re = np.reshape(imgArray2,(1,3,32,32))#学習データ用に変換する1*3*32*32
    img_cuda = model.xp.asarray(img_re)#
    with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
        Y = model.predictor(img_cuda)
    y = Y.array
    pred_label = y.argmax(axis=1)
    return pred_label, np.max(y.data, axis = 1)

#動画からの移動体検知関数
#引数：1，動画　2，検出結果のラベルリスト
def diff_frame(video,model,flabels):
    fgbg = cv.createBackgroundSubtractorKNN()
    cap = cv.VideoCapture(video)
    
    font = cv.FONT_HERSHEY_PLAIN

    ret, frame = cap.read()#OpenCVでの移動体検知用Frame
    while(ret==True):
        #動画からフレーム取得
        
        frame2 = to_matplotlib_format(frame)#opencv→matplotlib 学習機への入力用
        frame3 = frame#学習結果の出力用
        
        #動体検知のマスクをかける
        fgmask = fgbg.apply(frame)
        fgmask = cv.GaussianBlur(fgmask,(17,17),0)#GaussianBlurをかけることで，細かいコンタ成分を消すことができる
        ret,thresh = cv.threshold(fgmask,127,255,cv.THRESH_BINARY)#閾値を入れることで，2値化する
        cv.imshow('thresh',thresh)
        
        #contourを検出する
        image, contours, hierarchy = cv.findContours(thresh,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
        img_rect = frame3

        #１フレーム上の検出したContourで識別処理
        for c in contours:
            if cv.contourArea(c) < 200:#Contour内の面積が小さすぎるものを排除する
                continue
            # rectangle area　で区切る
            x, y, w, h = cv.boundingRect(c)
            part = frame2[y:(y+h),x:(x+w)]#学習機への入力用に区切ったnumpy.ndarrray配列を保存する
            img_rect = cv.rectangle(img_rect, (x, y), (x + w, y + h), (0, 255, 0), 3)  #rectangle contourを描く

            ans, val = predict(model,part)#学習機での予測結果の出力
            cv.putText(img_rect,flabels[ans[0]],(x-5,y-5),font,1,(255,255,0))#予測結果を画像に描く
        cv.imshow('rectangle',img_rect)#表示
        
        ret, frame = cap.read()#OpenCVでの移動体検知用Frame
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu',type=int, default=-1)
    parser.add_argument('--pretrained-model',default='trained_model_cpu3')
    parser.add_argument('video')
    args = parser.parse_args()
    if args.pretrained_model == 'trained_model_cpu3':
        #model = L.Classifier(Mynet.MyNet(100))
        model = L.Classifier(VGG_chainer.VGG(5))
        serializers.load_npz('trained_model_cpu3',model)
        print('VGG is defined')
    else:
        print('error!! model should be changed.')
        exit()
    if args.gpu >= 0:
        model.to_gpu(args.gpu)
        print('gpu is defined')
    else:
        model.to_cpu()
        print('cpu is defined')
    flabels = ['bicycle','motorcycle','train','automobile','person']
    diff_frame(args.video,model,flabels)


if __name__ =='__main__':
    main()
