# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 21:14:45 2018

@author: Mikihiro Ikura
"""

# In[]
import argparse

import chainer
import chainer.links as L

import os
import numpy
import matplotlib.pyplot as plt
import numpy as np

from chainer import training
from chainer.training import extensions
#from chainer.training import triggers
from chainer.datasets import get_cifar100
from chainer.datasets import get_cifar10
from chainer.datasets import tuple_dataset
from chainer import serializers
from chainer.datasets import split_dataset_random
from chainer.cuda import to_cpu
from chainer import cuda
from PIL import Image

import VGG_chainer
#import Mynet

# In[]
#datasetの作成関数
#bicycle:0 motorcycle:1 train:2 automobile:3 person:4
#cifar10 →　automobile
#cifar100 → bicycle motorcycle train
#person →　/dataset/PedCut2013_SegmentationDataset
def make_datasets():
    Images = []#3*32*32
    Nums = []
    Images_test = []#3*32*32
    Nums_test = []
    cf100_train , cf100_test = get_cifar100()
    cf10_train, cf10_test = get_cifar10()
    #cifar100のリストへの保存
    for i in cf100_train:
        if(i[1]==8 or i[1]==48 or i[1]==90):#bicycle 8,motorcycle 48, train 90
            Images.append(i[0])
            itrans = i[0][:,:,::-1]
            Images.append(itrans)
            if(i[1]==8):
                Nums.append(0)
                Nums.append(0)
            elif(i[1]==48):
                Nums.append(1)
                Nums.append(1)
            else:
                Nums.append(2)
                Nums.append(2)
    for j in cf100_test:
        if(j[1]==8 or j[1]==48 or j[1]==90):
            Images_test.append(j[0])
            jtrans = j[0][:,:,::-1]
            Images_test.append(jtrans)
            if(j[1]==8):
                Nums_test.append(0)
                Nums_test.append(0)
            elif(j[1]==48):
                Nums_test.append(1)
                Nums_test.append(1)
            else:
                Nums_test.append(2)
                Nums_test.append(2)
    for k in cf10_train:
        if(k[1]==1):#automobile
            Images.append(k[0])
            Images.append(k[0][:,:,::-1])
            Nums.append(3)
            Nums.append(3)
        if(len(Images)==2000*2):
            break
    for k in cf10_test:
        if(k[1]==1):#automobile
            Images_test.append(k[0])
            Images_test.append(k[0][:,:,::-1])
            Nums_test.append(3)
            Nums_test.append(3)
            if(len(Images_test)==400*2):
                break
    data_dir_path = u"./dataset/from_vtest/"
    file_list = os.listdir(r'./dataset/from_vtest/')
    for file_name in file_list:
        root, ext = os.path.splitext(file_name)
        if ext == u'.png' or u'.jpeg' or u'.jpg':
            abs_name = data_dir_path + '/' + file_name
            im = Image.open(abs_name)
            im = im.resize((32,32))
            imarray = numpy.asarray(im)
            Images.append(imarray.transpose(2,0,1).astype(numpy.float32)/256)
            s = imarray.transpose(2,0,1).astype(numpy.float32)/256
            strans = s[:,:,::-1]
            Images.append(strans)            
            Nums.append(4)
            Nums.append(4)
        if(len(Images)==2500*2+2000):
            break
    for i in range(3000,3200):
        file_name = file_list[i]
        root, ext = os.path.splitext(file_name)
        if ext == u'.png' or u'.jpeg' or u'.jpg':
            abs_name = data_dir_path + '/' + file_name
            im = Image.open(abs_name)
            im = im.resize((32,32))
            imarray = numpy.asarray(im)
            Images_test.append(imarray.transpose(2,0,1).astype(numpy.float32)/256)
            t = imarray.transpose(2,0,1).astype(numpy.float32)/256
            ttrans = t[:,:,::-1]
            Images_test.append(ttrans)
            Nums_test.append(4)
            Nums_test.append(4)
    trains = tuple_dataset.TupleDataset(Images,Nums)
    tests = tuple_dataset.TupleDataset(Images_test,Nums_test)
    return trains,tests


# In[]
batchsize = 256
learnrate = 0.05
epoch = 300
gpu = 0
out = 'result'
resume = ''

# In[]
class_labels = 5
train_val ,test= make_datasets() 
# model = L.Classifier(VGG16Net.VGG16Net(class_labels))
train_size = int(len(train_val) * 0.9)  
train, valid = split_dataset_random(train_val, train_size, seed=0)
model = L.Classifier(VGG_chainer.VGG(class_labels))
#GPUのセットアップ
if gpu >= 0:
    # Make a specified GPU current
    chainer.backends.cuda.get_device_from_id(gpu).use()
    model.to_gpu(gpu)  # Copy the model to the GPU
    xp = cuda.cupy

optimizer = chainer.optimizers.MomentumSGD(lr=learnrate).setup(model)
optimizer.add_hook(chainer.optimizer.WeightDecay(0.0005))

# In[]
train_iter = chainer.iterators.SerialIterator(train, batchsize)
valid_iter = chainer.iterators.SerialIterator(valid, batchsize,
                                             repeat=False, shuffle=False)
stop_trigger = (epoch,'epoch')
updater = training.updaters.StandardUpdater(train_iter,optimizer,device=gpu)
trainer = training.Trainer(updater,stop_trigger,out=out)

# Evaluate the model with the test dataset for each epoch
trainer.extend(extensions.Evaluator(valid_iter, model, device=gpu))
# Dump a computational graph from 'loss' variable at the first iteration
# The "main" refers to the target link of the "main" optimizer.
trainer.extend(extensions.dump_graph('main/loss'))
# Write a log of evaluation statistics for each epoch
trainer.extend(extensions.LogReport())
trainer.extend(extensions.PrintReport(
    ['epoch', 'main/loss', 'validation/main/loss',
     'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
# Print a progress bar to stdout
trainer.extend(extensions.ProgressBar())
trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'], x_key='epoch', file_name='loss.png'))
trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'], x_key='epoch', file_name='accuracy.png'))

# In[]
trainer.run()

# In[]
model.to_cpu()
serializers.save_npz('trained_model_cpu4',model)

# In[]
model = L.Classifier(VGG_chainer.VGG(5))
serializers.load_npz('trained_model',model)
# In[]
test_iter = chainer.iterators.MultiprocessIterator(test, batchsize, False, False)
test_evaluator = extensions.Evaluator(test_iter, model, device=gpu)
results = test_evaluator()
print('Test accuracy:', results['main/accuracy'])

# In[]
gpu_id = 0 # CPUで計算をしたい場合は、-1を指定してください

if gpu_id >= 0:
    model.to_gpu(gpu_id)

# 1つ目のテストデータを取り出します
x, t = test[40]  #  tは使わない

# どんな画像か表示してみます
plt.imshow(x.transpose(1,2,0))
plt.show()

# ミニバッチの形にする（複数の画像をまとめて推論に使いたい場合は、サイズnのミニバッチにしてまとめればよい）
print('元の形：', x.shape, end=' -> ')

x = x[None, ...]

print('ミニバッチの形にしたあと：', x.shape)

# ネットワークと同じデバイス上にデータを送る
x = model.xp.asarray(x)

# モデルのforward関数に渡す
with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
    y = model.predictor(x)

# Variable形式で出てくるので中身を取り出す
y = y.array

# 結果をCPUに送る
#y = to_cpu(y)

# 予測確率の最大値のインデックスを見る
pred_label = y.argmax(axis=1)

print('ネットワークの予測:', pred_label[0])
# In[]
model = L.Classifier(VGG_chainer.VGG(5))
serializers.load_npz('trained_model_cpu4',model)

# In[]
#gpu_id = 0  # CPUで計算をしたい場合は、-1を指定してください
#
#if gpu_id >= 0:
#    model.to_gpu(gpu_id)
model.to_cpu()
cnt = 0
cnt_num = 1000
# 1つ目のテストデータを取り出します
for i in range(cnt_num):
    x, t = test[i]  #  tは使わない
    
#    # どんな画像か表示してみます
#    plt.imshow(x.transpose(1,2,0))
#    plt.show()
    
    # ミニバッチの形にする（複数の画像をまとめて推論に使いたい場合は、サイズnのミニバッチにしてまとめればよい）
    print('元の形：', x.shape, end=' -> ')
    
    x = x[None, ...]
    
#    print('ミニバッチの形にしたあと：', x.shape)
    
    # ネットワークと同じデバイス上にデータを送る
    x = model.xp.asarray(x)
    
    # モデルのforward関数に渡す
    with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
        y = model.predictor(x)
    
    # Variable形式で出てくるので中身を取り出す
    y = y.array
    
    # 結果をCPUに送る
    #y = to_cpu(y)
    
    # 予測確率の最大値のインデックスを見る
    pred_label = y.argmax(axis=1)
    
    print('ネットワークの予測:', pred_label[0],'  実際の結果:',t)
    if(pred_label[0]==t):
        cnt = cnt+1
print('final accuracy: ',cnt,'/',cnt_num)

# In[]
#gpu_id = 0  # CPUで計算をしたい場合は、-1を指定してください
#
#if gpu_id >= 0:
#    model.to_gpu(gpu_id)
model.to_cpu()
cnt = 0
cnt_num = 500
# 1つ目のテストデータを取り出します
x_data = Image.open('./dataset/from_vtest/example4.jpg').convert('RGB')

pilImg = Image.fromarray(numpy.uint8(x_data))#PIL dataに変換
img_resize = pilImg.resize((32, 32))#画像サイズ変換32*32*3
imgArray = numpy.asarray(img_resize)#numpy ndarrayに変換
imgArray2 = imgArray.astype(np.float32)/256#float32の配列に変換 256で割る
img_re = np.reshape(imgArray2,(1,3,32,32))#学習データ用に変換する1*3*32*32
img_cuda = model.xp.asarray(img_re)#


# モデルのforward関数に渡す
with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
    y = model.predictor(img_cuda)

# Variable形式で出てくるので中身を取り出す
y = y.array

# 結果をCPUに送る
#y = to_cpu(y)

# 予測確率の最大値のインデックスを見る
pred_label = y.argmax(axis=1)

print('ネットワークの予測:', pred_label[0])

