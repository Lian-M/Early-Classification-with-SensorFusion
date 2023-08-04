%%%%%%%%%%%%%%%%%%%%%%提取.csv文件中目标的雷达检测时间序列%%%%%%%%%%%%%%%%%%%
%% extract the tracks of the three AirSim drones 
clear
clc
close all
%% Use fixed random seed for simulation repeatablity.
rng(0)
%% 读取包含雷达数据的.csv文件
data_track = csvread('E:\渥太华大学\RHC UAV仿真数据实验\Hamid-New\Scenario 1\myData_Tracks.csv', 1, 15, [1,15,675,20]);
track1=zeros(225,6);              % 目标1的位置(m)与速度信息(m/s)，x,y,z,v_x,v_y,v_z
track2=zeros(225,6);
track3=zeros(225,6);
for i=1:225
    track1(i,:)=data_track((i*3-2),:);
end
for i=1:225
    track2(i,:)=data_track((i*3-1),:);
end
for i=1:225
    track3(i,:)=data_track((i*3-0),:);
end
% figure
% plot3(data1(:,1),data1(:,2),data1(:,3))
% figure
% plot3(data2(:,1),data2(:,2),data2(:,3))
% figure
% plot3(data3(:,1),data3(:,2),data3(:,3))
data_detection = csvread('E:\渥太华大学\RHC UAV仿真数据实验\Hamid-New\Scenario 1\myData_Detections.csv', 1, 15, [1,15,675,18]);
detection1=zeros(225,4);              % 目标1的距离(m)，方位(deg)，俯仰(deg)与径向速度(m/s)，range,az,el,vrad
detection2=zeros(225,4);
detection3=zeros(225,4);
for i=1:225
    detection1(i,:)=data_detection((i*3-2),:);
end
for i=1:225
    detection2(i,:)=data_detection((i*3-1),:);
end
for i=1:225
    detection3(i,:)=data_detection((i*3-0),:);
end
%% 拼接track与detection
Target1=zeros(225,10);              % 目标1的位置(m)，速度信息(m/s)，距离(m)，方位(deg)，俯仰(deg)与径向速度(m/s)——x,y,z,v_x,v_y,v_z,range,az,el,vrad
Target2=zeros(225,10);
Target3=zeros(225,10);

Target1(:,1:6)=track1;
Target1(:,7:10)=detection1;
Target2(:,1:6)=track2;
Target2(:,7:10)=detection2;
Target3(:,1:6)=track3;
Target3(:,7:10)=detection3;
%% 变换特征位置为-x,vx,y,vy,z,vz,range,az,el,vrad
Target1(:,[4 2])=Target1(:,[2 4]);
Target1(:,[4 3])=Target1(:,[3 4]);
Target1(:,[5 4])=Target1(:,[4 5]);
Target2(:,[4 2])=Target2(:,[2 4]);
Target2(:,[4 3])=Target2(:,[3 4]);
Target2(:,[5 4])=Target2(:,[4 5]);
Target3(:,[4 2])=Target3(:,[2 4]);
Target3(:,[4 3])=Target3(:,[3 4]);
Target3(:,[5 4])=Target3(:,[4 5]);

data = zeros(3,225,10);   % B,T,V
data(1,:,:)=Target1;
data(2,:,:)=Target2;
data(3,:,:)=Target3;
label=zeros(3,2);
label(:,2)=1;
%% 将每个UAV数据经分割后放入矩阵data，并生成对应标签放入矩阵label
% data_temp=zeros(4*2,15,10);
% label_temp=zeros(4*2,2);
% for k=1:4
%     data_temp(k*2-1,:,:)=Track11(:,(k-1)*15+1:k*15).'; %(15,13)
%     label_temp(k*2-1,:)=[1 0];
%     data_temp(k*2,:,:)=Track22(:,(k-1)*15+1:k*15).';
%     label_temp(k*2,:)=[0 1];
% end
% % data((i-1)*40*2+1:i*40*2,:,:)=data_temp;
% % label((i-1)*40*2+1:i*40*2,:)=label_temp;
% data((i-1)*4*2+1:i*4*2,:,:)=data_temp;
% label((i-1)*4*2+1:i*4*2,:)=label_temp;
% 
% %% 随机采样打乱data的顺序
% Index=randperm(4*Num*2);
% data(:,:,:)=data(Index,:,:);
% data(:,:,3)=[];
% data(:,:,6)=[];
% data(:,:,9)=[];
% size(data)
% label(:,:)=label(Index,:);
%% 数据归一化
data_nor=zeros(3,225,10);  
% for i=1:4*2*Num
%     data_nor(i,:,:)=(data(i,:,:)-min(min(data(i,:,:))))/(max(max(data(i,:,:)))-min(min(data(i,:,:)))); 
%     i
% end
data_nor(:,:,:)=(data(:,:,:)-min(min(min(data(:,:,:)))))/(max(max(max(data(:,:,:))))-min(min(min(data(:,:,:))))); 
%% 保存数据
save('E:\\渥太华大学\\RHC UAV仿真数据实验\\Meng-simulated UAV&Bird data\\3Drones\\data.mat','data','-v7.3')
save('E:\\渥太华大学\\RHC UAV仿真数据实验\\Meng-simulated UAV&Bird data\\3Drones\\data_nor.mat','data_nor','-v7.3')
save('E:\\渥太华大学\\RHC UAV仿真数据实验\\Meng-simulated UAV&Bird data\\3Drones\\label.mat','label','-v7.3')













