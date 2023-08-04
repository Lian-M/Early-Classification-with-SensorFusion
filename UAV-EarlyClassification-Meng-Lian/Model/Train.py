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
nhid = 5            # nhid is the hidden state size of the rnn, and is 20 in this paper, use to set hidden state 
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
data_dir = 'E:/渥太华大学/Early Classification文章发表/RHC UAV仿真数据实验/Meng-simulated UAV&Bird data/225/'
data_temp = h5py.File(os.path.join(data_dir, 'data.mat'))
data = data_temp['data'][:, :, :]
data = data.transpose(2, 1, 0)

#label_dir = 'E:/渥太华大学/RHC UAV仿真数据实验/Meng-simulated UAV&Bird data/'
label_dir = 'E:/渥太华大学/Early Classification文章发表/RHC UAV仿真数据实验/Meng-simulated UAV&Bird data/225/'
label_temp = h5py.File(os.path.join(label_dir, 'label.mat'))
labels = label_temp['label'][:, :]
labels = labels.transpose(1, 0)

print(np.shape(data))   # （B, T, V）T: timesteps, B: number of time series, V: dimension of time series(number of variables) (9600,15,13)
print(np.shape(labels)) # （B, L） L: number of classes (9600,2)
train_rate = 0.8        # the percentage of train data in dataset, train_rate + validation_rate + test_rate = 1
validation_rate = 0.1   # the percentage of validation data in dataset
test_rate = 0.1         # the percentage of test data in dataset
train_size = int(train_rate * np.shape(data)[0])
validation_size = int(validation_rate * np.shape(data)[0])
test_size = int(test_rate * np.shape(data)[0])
train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(data, [train_size, validation_size, test_size], generator=torch.Generator().manual_seed(42))                  # randomly split the loaded dataset with given rate(eg, 0.8, 0.1, 0.1) 
trainlabel_dataset, validationlabel_dataset, testlabel_dataset = torch.utils.data.random_split(labels, [train_size, validation_size, test_size], generator=torch.Generator().manual_seed(42)) # the manual_seed needs to be as same as above to keep the Corresponding relationship between data and labels

train_dataset = torch.tensor(train_dataset, dtype=torch.float)
validation_dataset = torch.tensor(validation_dataset, dtype=torch.float)
test_dataset = torch.tensor(test_dataset, dtype=torch.float)
trainlabel_dataset = torch.tensor(trainlabel_dataset, dtype=torch.float)
validationlabel_dataset = torch.tensor(validationlabel_dataset, dtype=torch.float)
testlabel_dataset = torch.tensor(testlabel_dataset, dtype=torch.float)

# Shape check 
print(np.shape(train_dataset))
print(np.shape(validation_dataset))
print(np.shape(test_dataset))
print(np.shape(trainlabel_dataset))
print(np.shape(validationlabel_dataset))
print(np.shape(testlabel_dataset))

# Combined packing of training data and its labels
train_dataset0 = torch.utils.data.TensorDataset(train_dataset, trainlabel_dataset)
validation_dataset0 = torch.utils.data.TensorDataset(validation_dataset, validationlabel_dataset)
test_dataset0 = torch.utils.data.TensorDataset(test_dataset, testlabel_dataset)

# Data loader (Dividing the data into batches)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset0,
                                           batch_size=batch_size, 
                                           shuffle=False)
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset0,
                                           batch_size=batch_size, 
                                           shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset0,
                                          batch_size=96,                      # batch_size设置为96（即test数据集中instance的数量），将所有测试数据一次性输入模型进行预测，比较推荐此种做法
                                          shuffle=False)

# Recurrent Halting Chain(RHC)
min_loss = 1
RHC = RHC(ninp, nhid, nclasses, nlayers).to(device)              # Setup network

# Optimizer
optimizer = torch.optim.Adam(RHC.parameters(), lr=learning_rate) # Setup optimizer

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    #confidence_sum = torch.zeros((1))
    RHC.train()
    for i, (train_dataset, trainlabel_dataset) in enumerate(train_loader):
        train_dataset1 = train_dataset.transpose(1, 0).to(device)    # transform the shape of X from (B, T, V) to (T, B, V) which is the input shape of RHC. PS:reshape will destroy the Data continuity in both space and time domain.
        #print('train_dataset ', np.shape(train_dataset))            # torch.Size([15, 49, 77])
        trainlabel_dataset = trainlabel_dataset.to(device)
        #print('trainlabel_dataset ', np.shape(trainlabel_dataset))  # torch.Size([49,6])

        # Forward pass
        outputs,mhp  = RHC(train_dataset1, epoch)           # output: the class probabilities predicted by RHC after one batch., mhp: a scalar indicates On average how much each class was halted over the whole batch      
        loss = RHC.computeLoss(outputs, trainlabel_dataset) # Since computeLoss needs inner variables, we put it here rather than in above Optimizer Section
        
        # outputs_val,mhp_val  = RHC(validation_dataset1, epoch)
        # loss_val = RHC.computeLoss(outputs_val, validationlabel_dataset)
   
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #confidence_sum = confidence_sum + confidence.item() 
        #if (i+1) % 100 == 0:                               # use this if the number of batches is too large.
        print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
    #print('Epoch [{}/{}], confidence: {:.4f}' 
                #.format(epoch+1, num_epochs, confidence_sum.item() / total_step))
    total_loss = 0
    RHC.eval()
    with torch.no_grad():
        for i, (validation_dataset, validationlabel_dataset) in enumerate(validation_loader):
            validation_dataset1 = validation_dataset.transpose(1, 0).to(device)
            validationlabel_dataset = validationlabel_dataset.to(device)
            # Forward pass
            outputs_val,mhp_val  = RHC(validation_dataset1, epoch)           # output: the class probabilities predicted by RHC after one batch., mhp: a scalar indicates On average how much each class was halted over the whole batch      
            loss_val = RHC.computeLoss(outputs_val, validationlabel_dataset) # Since computeLoss needs inner variables, we put it here rather than in above Optimizer Section
            total_loss += loss.item()
    # 计算平均损失
    avg_loss = total_loss / len(validation_loader)
    if avg_loss <= min_loss:
            min_loss = avg_loss
            print('save model')
            torch.save(RHC,'E:/渥太华大学/Early Classification文章发表/RHC UAV仿真数据实验/Meng-simulated UAV&Bird data/BestWeight/best.pt')

# Save the model
torch.save(RHC, 'E:/渥太华大学/Early Classification文章发表/RHC UAV仿真数据实验/Meng-simulated UAV&Bird data/BestWeight/last.pt')
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
#print(np.shape(ham_sum)) # (6,1)
with torch.no_grad(): 
    for k, (test_dataset, testlabel_dataset) in enumerate(test_loader):
        B = np.shape(test_dataset)[0]  #40
        T = np.shape(test_dataset)[1]  #15
        class_pre = torch.zeros(B, nclasses) # Record predicted classes 
        class_one = torch.ones(B, nclasses)  # replace probe
        percentage_observed = 0.2            # the percentage of time steps that has observed
        test_dataset1 = test_dataset[:, :round(T*percentage_observed), :]
        test_dataset2 = test_dataset1.transpose(1, 0).to(device)
        testlabel_dataset = testlabel_dataset.to(device)
        outputs, mhp = RHC(test_dataset2, 0, test=True)
        #print(outputs)           # predictions by RHC, compare with labels to Check classification accuracy
        #print(testlabel_dataset) # labels
        #print(mhp)               # mean_halting_points, Check earliness
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
        HammingLoss = hamming_loss(testlabel_dataset, class_pre)
        Micro_F1 = f1_score(testlabel_dataset, class_pre, average='micro') # the input 'class_pre' can be replaced by 'np.round(outputs)', but the round operation will decrease the performance, due to the Threshold of rounding function (default value is 0.5)
        Macro_F1 = f1_score(testlabel_dataset, class_pre, average='macro')
        Instance_AUC = roc_auc_score(testlabel_dataset, class_pre, average='samples')
        Micro_AUC = roc_auc_score(testlabel_dataset, class_pre, average='micro')
        Macro_AUC = roc_auc_score(testlabel_dataset, class_pre, average='macro')
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
        conf_matrix = confusion_matrix(testlabel_dataset.argmax(axis=1), class_pre.argmax(axis=1))
        print(conf_matrix)
        #print(confusion_matrix(testlabel_dataset.argmax(axis=1), class_pre.argmax(axis=1))) # Since this is a multi-label problem, confusion matrix can not be provided regularly.
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
torch.save(RHC.state_dict(), 'RHC.ckpt')