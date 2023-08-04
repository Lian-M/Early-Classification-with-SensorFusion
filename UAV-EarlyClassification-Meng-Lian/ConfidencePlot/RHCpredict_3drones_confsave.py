## This code is used to achieve the confidence of produced by RHC and save it as .txt file.
import torch
from torch import nn
from modules import *
from model import RHC
import numpy as np
#from loss import *
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import hamming_loss, f1_score, roc_auc_score, confusion_matrix
#import pandas as pd
#import seaborn as sns
import os
import h5py
import scipy.io
import matplotlib.pyplot as plt
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
sequence_length = 225 # length of timesteps, due to how u divide the original time series data 
ninp = 10            # ninp is the number of variables
#nhid = 300            # nhid is the hidden state size of the rnn, and is 20 in this paper, use to set hidden state 
nclasses = 2         # nclasses is the number of classes you want RHC to predict
nlayers = 2          # nlayers is the number of hidden layer, use to set hidden state 
batch_size = 16      # empirical value, the smaller it is, the easier loss converge to a ideal value, but slower, Need to Just divide the amount of data
num_epochs = 500     
learning_rate = 0.001

# Hamid simulated UAV dataset, uses an 80% training, 10% validation, and 10% testing split for all datasets.
#data_D = pd.read_csv('E:\\渥太华大学\\RHC UAV仿真数据实验\\Hamid-simulated UAV data\\myData_Detections.csv')     # load dataset from a dir_path
#data_M = pd.read_csv('E:\\渥太华大学\\RHC UAV仿真数据实验\\Hamid-simulated UAV data\\myData_Measurements.csv')
#data_T = pd.read_csv('E:\\渥太华大学\\RHC UAV仿真数据实验\\Hamid-simulated UAV data\\myData_Tracks.csv')
#exit()
#data = np.load('E:/渥太华大学/RHC UAV仿真数据实验/Meng-simulated UAV&Bird data/data.npy')
#labels = np.load('E:/渥太华大学/RHC UAV仿真数据实验/Meng-simulated UAV&Bird data/label.npy') # load labels of these dataset from a dir_path

#data_dir = 'E:/渥太华大学/RHC UAV仿真数据实验/Meng-simulated UAV&Bird data/'
data_dir = 'E:/渥太华大学/Early Classification文章发表/RHC UAV仿真数据实验/Meng-simulated UAV&Bird data/Mixdata/'
data_temp = h5py.File(os.path.join(data_dir, 'data_mix.mat'))
data = data_temp['data_mix'][:, :, :]
data = data.transpose(2, 1, 0)

#label_dir = 'E:/渥太华大学/RHC UAV仿真数据实验/Meng-simulated UAV&Bird data/'
label_dir = 'E:/渥太华大学/Early Classification文章发表/RHC UAV仿真数据实验/Meng-simulated UAV&Bird data/Mixdata/'
label_temp = h5py.File(os.path.join(label_dir, 'label_mix.mat'))
labels = label_temp['label_mix'][:, :]
labels = labels.transpose(1, 0)

print(np.shape(data))   # （B, T, V）T: timesteps, B: number of time series, V: dimension of time series(number of variables) (3,225,10)
print(np.shape(labels)) # （B, L） L: number of classes (3,2)

data = torch.tensor(data, dtype=torch.float)
labels = torch.tensor(labels, dtype=torch.float)

# Combined packing of training data and its labels
test_dataset0 = torch.utils.data.TensorDataset(data, labels)

# Data loader (Dividing the data into batches)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset0,
                                          batch_size=33,                      # batch_size设置为3（即test数据集中instance的数量），将所有测试数据一次性输入模型进行预测，比较推荐此种做法
                                          shuffle=False)

# load the model
RHC = torch.load('E:/渥太华大学/Early Classification文章发表/RHC UAV仿真数据实验/Meng-simulated UAV&Bird data/BestWeight/last.pt')
# Test the model
RHC.eval()
ham_sum = torch.zeros((len(test_loader), 1))
MiF1_sum = torch.zeros((len(test_loader), 1))
MaF1_sum = torch.zeros((len(test_loader), 1))
Ins_sum = torch.zeros((len(test_loader), 1))
MiAUC_sum = torch.zeros((len(test_loader), 1))
MaAUC_sum = torch.zeros((len(test_loader), 1))

Drone1_conf_predict = torch.zeros(225,2)
Drone2_conf_predict = torch.zeros(225,2)
Drone3_conf_predict = torch.zeros(225,2)
#print(np.shape(ham_sum)) # (6,1)
with torch.no_grad():
    for i in range(225): 
        for k, (data, labels) in enumerate(test_loader):
            B = np.shape(data)[0]  #33
            T = np.shape(data)[1]  #225
            class_pre = torch.zeros(B, nclasses) # Record predicted classes 
            class_one = torch.ones(B, nclasses)  # replace probe
            percentage_observed = (i+1)/T            # the percentage of time steps that has observed
            test_dataset1 = data[:, :round(T*percentage_observed), :]
            #print(test_dataset1)
            test_dataset2 = test_dataset1.transpose(1, 0).to(device)
            labels = labels.to(device)
            outputs, mhp = RHC(test_dataset2, 0, test=True)

            Drone1_conf_predict[i,:] = outputs[6,:]
            Drone2_conf_predict[i,:] = outputs[3,:]
            Drone3_conf_predict[i,:] = outputs[24,:]

            print(outputs)           # predictions by RHC, compare with labels to Check classification accuracy
            #print(testlabel_dataset) # labels
            print(mhp)               # mean_halting_points, Check earliness
            # Mean STD
            #difference = testlabel_dataset - outputs
            #STD = torch.std(difference, dim=1).mean()
            #print('Mean STD of RHC on the 49 test time series: {}'.format(STD))

            threshold = 0.5 # empirical value (how to set a more reasonable value?)
            class_pre = torch.where((outputs >= threshold) & (class_pre == 0), class_one, class_pre) # predicted classification result
            #print(class_pre)

            #error = testlabel_dataset - class_pre
            #error_num = torch.count_nonzero(error) 
            #print('Average classification accuracy over the 49 test time series: {} %'.format(100 * (1 - error_num / (B * nclasses))))

            # Micro-AUC, Macro-AUC, Micro-F1, Macro-F1, HammingLoss
            HammingLoss = hamming_loss(labels, class_pre)
            Micro_F1 = f1_score(labels, class_pre, average='micro') # the input 'class_pre' can be replaced by 'np.round(outputs)', but the round operation will decrease the performance, due to the Threshold of rounding function (default value is 0.5)
            Macro_F1 = f1_score(labels, class_pre, average='macro')
            Instance_AUC = roc_auc_score(labels, class_pre, average='samples')
            Micro_AUC = roc_auc_score(labels, class_pre, average='micro')
            Macro_AUC = roc_auc_score(labels, class_pre, average='macro')
            #print('HammingLoss over {} % of the test time series: {}'.format(100*percentage_observed, HammingLoss))
            #print('Micro_F1 over {} % of the test time series: {}'.format(100*percentage_observed, Micro_F1))
            #print('Macro_F1 over {}% of the test time series: {}'.format(100*percentage_observed, Macro_F1))
            #print('Instance_AUC over {} % of the test time series: {}'.format(100*percentage_observed, Instance_AUC))
            #print('Micro_AUC over {} % of the test time series: {}'.format(100*percentage_observed, Micro_AUC))
            #print('Macro_AUC over {} % of the test time series: {}'.format(100*percentage_observed, Macro_AUC))
            ham_sum[k] = HammingLoss
            MiF1_sum[k] = Micro_F1
            MaF1_sum[k] = Macro_F1
            Ins_sum[k] = Instance_AUC
            MiAUC_sum[k] = Micro_AUC
            MaAUC_sum[k] = Macro_AUC
            conf_matrix = confusion_matrix(labels.argmax(axis=1),class_pre.argmax(axis=1))
            print(conf_matrix)
            #print(confusion_matrix(testlabel_dataset.argmax(axis=1), class_pre.argmax(axis=1))) # Since this is a multi-label problem, confusion matrix can not be provided regularly.
np.savetxt('Drone1_conf.txt',Drone1_conf_predict)
np.savetxt('Drone2_conf.txt',Drone2_conf_predict)
np.savetxt('Drone3_conf.txt',Drone3_conf_predict)
print('Average HammingLoss over {} % of the test time series: {}'.format(100*percentage_observed, ham_sum.mean()))
print('Average Micro_F1 over {} % of the test time series: {}'.format(100*percentage_observed, MiF1_sum.mean()))
print('Average Macro_F1 over {} % of the test time series: {}'.format(100*percentage_observed, MaF1_sum.mean()))
print('Average Instance_AUC {} % of the test time series: {}'.format(100*percentage_observed, Ins_sum.mean()))
print('Average Micro_AUC {} % of the test time series: {}'.format(100*percentage_observed, MiAUC_sum.mean()))
print('Average Macro_AUC {} % of the test time series: {}'.format(100*percentage_observed, MaAUC_sum.mean()))
# 绘制混淆矩阵
Emotion=2#这个数值是具体的分类数，大家可以自行修改
lab = ['Bird', 'UAV']#每种类别的标签

# 显示数据
plt.imshow(conf_matrix, cmap=plt.cm.Blues)

# 在图中标注数量/概率信息
thresh = conf_matrix.max() / 2  #数值颜色阈值，如果数值超过这个，就颜色加深。
for x in range(Emotion):
    for y in range(Emotion):
        # 注意这里的matrix[y, x]不是matrix[x, y]
        info = int(conf_matrix[y, x])
        plt.text(x, y, info,
                 verticalalignment='center',
                 horizontalalignment='center',
                 color="white" if info > thresh else "black")
                 
plt.tight_layout()#保证图不重叠
plt.yticks(range(Emotion), lab)
plt.xticks(range(Emotion), lab,rotation=45)#X轴字体倾斜45°
plt.show()
plt.close()
# Save the model checkpoint
#torch.save(RHC.state_dict(), 'RHC.ckpt')