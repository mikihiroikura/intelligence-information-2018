# In[]
import argparse
from chainer.datasets import cifar
import matplotlib.pyplot as plt
import pickle
import os
import numpy as np
import chainer.links as L
import Mynet
from chainer import serializers
import chainer

# In[]
os.getcwd()
os.chdir('C:/Users/mikiu/Desktop/det_rec_mov_obj')
os.getcwd()
# In[]
_, test = cifar.get_cifar100(withlabel=True)
test
# In[]
x,t=test[10]
t

# In[]
plt.imshow(x.transpose(1,2,0))

# In[]
def unpickle(file):
    # file.decode('utf-8')
    fo = open(file, 'rb')
    # fo.decode('utf-8')
    dict = pickle.load(fo,encoding="latin1")
    fo.close()
    return dict

# In[]
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

# In[]
data_path = "./cifar-100-python"
test_data, test_clabels, test_flabels, clabels,flabels = get_cifar100(data_path)

# In[]
print(flabels[t])
for i in clabels:
    print(i)
test_flabels[10]
# Mikihiro Ikura

# In[]
def predict(model, x_data):
    x = chainer.Variable(x_data.astype(np.float32))
    y = model.predictor(x)
    return np.argmax(y.data, axis = 1)

# In[]
model = L.Classifier(Mynet.MyNet(100))
serializers.load_npz('trained_model',model)
a,b = test[40]
plt.imshow(a.transpose(1,2,0))
print('predicted_label:', flabels[b])
# In[]
print('answer:', flabels[predict(model,a)])