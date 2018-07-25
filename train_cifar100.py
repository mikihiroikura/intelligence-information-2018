# In[]
import argparse

import chainer
import chainer.links as L

import os
import numpy


from chainer import training
from chainer.training import extensions
#from chainer.training import triggers
from chainer.datasets import get_cifar100
from chainer.datasets import get_cifar10
from chainer.datasets import tuple_dataset
from chainer import serializers
from chainer.datasets import split_dataset_random
from PIL import Image

import VGG_chainer
#import Mynet

# In[]
#datasetの作成関数
#bicycle:0 motorcycle:1 automobile:2 train:3 person:4
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
            if(i[1]==8):
                Nums.append(0)
            elif(i[1]==48):
                Nums.append(1)
            else:
                Nums.append(2)
    for j in cf100_test:
        if(j[1]==8 or j[1]==48 or j[1]==90):
            Images_test.append(j[0])
            if(j[1]==8):
                Nums_test.append(0)
            elif(j[1]==48):
                Nums_test.append(1)
            else:
                Nums_test.append(2)
    for k in cf10_train:
        if(k[1]==1):#automobile
            Images.append(k[0])
            Nums.append(3)
        if(len(Images)==2000):
            break
    for k in cf10_test:
        if(k[1]==1):#automobile
            Images_test.append(k[0])
            Nums_test.append(3)
        if(len(Images_test)==400):
            break
    data_dir_path = u"./dataset/PedCut2013_SegmentationDataset/data/completeData/left_images/"
    file_list = os.listdir(r'./dataset/PedCut2013_SegmentationDataset/data/completeData/left_images/')
    for file_name in file_list:
        root, ext = os.path.splitext(file_name)
        if ext == u'.png' or u'.jpeg' or u'.jpg':
            abs_name = data_dir_path + '/' + file_name
            im = Image.open(abs_name)
            im = im.resize((32,32))
            imarray = numpy.asarray(im)
            Images.append(imarray.transpose(2,0,1).astype(numpy.float32)/256)
            Nums.append(4)
        if(len(Images)==2500):
            break
    for i in range(500,600):
        file_name = file_list[i]
        root, ext = os.path.splitext(file_name)
        if ext == u'.png' or u'.jpeg' or u'.jpg':
            abs_name = data_dir_path + '/' + file_name
            im = Image.open(abs_name)
            im = im.resize((32,32))
            imarray = numpy.asarray(im)
            Images_test.append(imarray.transpose(2,0,1).astype(numpy.float32)/256)
            Nums_test.append(4)
    trains = tuple_dataset.TupleDataset(Images,Nums)
    tests = tuple_dataset.TupleDataset(Images_test,Nums_test)
    return trains,tests
# In[]
def main():
    parser = argparse.ArgumentParser(description='Chainer CIFAR example:')
    parser.add_argument('--batchsize', '-b', type=int, default=64,
                        help='Number of images in each mini-batch')
    parser.add_argument('--learnrate', '-l', type=float, default=0.05,
                        help='Learning rate for SGD')
    parser.add_argument('--epoch', '-e', type=int, default=300,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--early-stopping', type=str,
                        help='Metric to watch for early stopping')
    args = parser.parse_args()

    # Set up a neural network to train.
    # Classifier reports softmax cross entropy loss and accuracy at every
    # iteration, which will be used by the PrintReport extension below.

    # In[]
    class_labels = 5
    train_val ,test= make_datasets() 
    # model = L.Classifier(VGG16Net.VGG16Net(class_labels))
    train_size = int(len(train_val) * 0.9)
    train, valid = split_dataset_random(train_val, train_size, seed=0)
    model = L.Classifier(VGG_chainer.VGG(class_labels))
    #GPUのセットアップ
    if args.gpu >= 0:
        # Make a specified GPU current
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()  # Copy the model to the GPU

    optimizer = chainer.optimizers.MomentumSGD(lr=args.learnrate).setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(0.0005))

    # In[]
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    valid_iter = chainer.iterators.SerialIterator(valid, args.batchsize,
                                                 repeat=False, shuffle=False)
    stop_trigger = (args.epoch,'epoch')
    updater = training.updaters.StandardUpdater(train_iter,optimizer,device=args.gpu)
    trainer = training.Trainer(updater,stop_trigger,out=args.out)

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(extensions.Evaluator(valid_iter, model, device=args.gpu))
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
    trainer.extend(extensions.PlotReport(['main/loss', 'val/main/loss'], x_key='epoch', file_name='loss.png'))
    trainer.extend(extensions.PlotReport(['main/accuracy', 'val/main/accuracy'], x_key='epoch', file_name='accuracy.png'))

    # In[]
    trainer.run()

    # In[]
    serializers.save_npz('trained_model',model)

if __name__ == '__main__':
    main()
