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
import VGG_chainer
import cv2 as cv

# In[]
os.getcwd()
os.chdir('C:/Users/mikiu/Desktop/det_rec_mov_obj')
os.getcwd()
# In[]
train, test = cifar.get_cifar100(withlabel=True)
test
# In[]
x,t=test[10]
t
# In[]
z,w = train[10]
w
# In[]
plt.imshow(x.transpose(1,2,0))
plt.imshow(z.transpose(1,2,0))
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
    x_data = np.reshape(x_data,(1,3,32,32))
    if(x_data!='float32'):
        x_data = x_data.astype(np.float32)
        x_data = x_data/256
    #x = chainer.Variable(x_data.astype(np.float32))
    y = model.predictor(x_data)
    return np.argmax(y.data, axis = 1), np.max(y.data, axis = 1)

# In[]
#model = L.Classifier(Mynet.MyNet(100))
model = L.Classifier(VGG_chainer.VGG(100))
serializers.load_npz('trained_model',model)
a,b = test[30]
plt.imshow(a.transpose(1,2,0))
print('predicted_label:', flabels[b])
ans,score = predict(model,a)
print("score: ",score)
print("answer: ",ans)
flabels[ans[0]]
# In[]
cnt = 0
for i in range(10000):
    a,b= train[i]
    ans,score = predict(model,a)

    for u in ans:
        if u==b:
            print('photo num: ', i)
            print("correct!!")
            cnt = cnt+1
print("correct cnt: ", cnt)
            
            
# In[]
c,d =test[730]
plt.imshow(c.transpose(1,2,0))

# In[]
img = cv.imread('walker.jpg')
img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
img = cv.resize(img,(32,32))
plt.imshow(img)
cv.waitKey(0)
cv.destroyAllWindows()

# In[]
img_trans = img.transpose(2,0,1)
plt.imshow(img_trans.transpose(1,2,0))
cv.waitKey(0)
cv.destroyAllWindows()

# In[]
ans,score,y = predict(model,img_trans)
print("answer: ",ans)
print("score: ",score)

# In[]
a,b = test[30]