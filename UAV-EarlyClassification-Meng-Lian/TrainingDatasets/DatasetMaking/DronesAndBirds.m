%%%%%%%%%%%%%%%%%%%%% This code is used to mix 3 AirSim simulated drones with 30 simulated birds.%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% The Bird simulation method is Based on <Hybrid AI-enabled Method for UAS and Bird Detection and Classification>
clear
clc
close all
%% Use fixed random seed for simulation repeatablity.
% rng(0)
%% Hamid参数读取/验证轨迹
% data = csvread('E:\渥太华大学\RHC UAV仿真数据实验\myData_Tracks.csv', 1, 15, [1,15,675,17]);
% data1=zeros(225,3);
% for i=1:225
%     data1(i,:)=data((i*3-2),:);
% end
% plot3(data(:,1),data(:,2),data(:,3))
%% 初始状态设定
x_s=0;                                                                     % 传感器坐标
y_s=0;
z_s=0;
Num=30;                                                                     % 生成120对Bird和UAV的实例
% data=zeros(40*Num*2,15,13);                                                % 并行循环120次，每次1个Bird&1个UAV,将长度为600的时间序列分成40个长度为15的时间序列，（9600，15，13）=（B,T,V）
% label=zeros(40*Num*2,2);                                                   % 标签采用one-hot编码，（1，0）代表鸟，（0，1）代表UAV
data_bird=zeros(Num,225,10);                                                 % 并行循环120次，每次1个Bird&1个UAV,将长度为60的时间序列分成4个长度为15的时间序列，（4*120*2，15，13）=（B,T,V）
label_bird=zeros(Num,2);                                                    % 标签采用one-hot编码，（1，0）代表鸟，（0，1）代表UAV
tic
for i=1:Num
%% Target1:Bird初始状态
x1=100+(150-100)*rand(1);                                                  % ~U(100,150)
vx1=sqrt(3)*randn(1)+5;                                                    % v~N(5,3)
ax1=sqrt(3)*randn(1)+7;                                                    % a~N(7,3)
y1=100+(150-100)*rand(1);
vy1=sqrt(3)*randn(1)+5;
ay1=sqrt(3)*randn(1)+7;
z1=100+(150-100)*rand(1);
vz1=sqrt(3)*randn(1)+5;
az1=sqrt(3)*randn(1)+7;
w1=0.2+(0.98-0.2)*rand(1);                                                 % w&omega~U(0.2,0.98)
                                              
%% 分阶段创建Target1:Bird的时间序列
t11=79;         % 210*225/600
t12=98;        % 260*225/600
t13=154;        % 410*225/600
t14=191;        % 510*225/600
Track1=zeros(10,226);
Track1(1,1)=x1;
Track1(2,1)=vx1;
Track1(3,1)=ax1;
Track1(4,1)=y1;
Track1(5,1)=vy1;
Track1(6,1)=ay1;
Track1(7,1)=z1;
Track1(8,1)=vz1;
Track1(9,1)=az1;
Track1(10,1)=w1;
T=0.1; %步长100ms
for t=2:t11
    type_decision1=round(rand(1));
    if type_decision1 == 0
        Track1(:,t)=TransFunction1(T)*Track1(:,t-1);
    else
        Track1(:,t)=TransFunction2(T)*Track1(:,t-1);
    end
end

for t=t11+1:t12
    type_decision2=round(rand(1));
    if type_decision2 == 0
        w=0.2+(0.98-0.2)*rand(1);
        Track1(:,t)=TransFunction3(w,T)*Track1(:,t-1);
    else
        omega=0.2+(0.98-0.2)*rand(1);
        Track1(:,t)=TransFunction4(omega,T)*Track1(:,t-1);
    end
end

for t=t12+1:t13
    Track1(:,t)=TransFunction2(T)*Track1(:,t-1);
end

for t=t13+1:t14
    type_decision3=round(rand(1));
    if type_decision3 == 0
        Track1(:,t)=TransFunction2(T)*Track1(:,t-1);
    else
        w=0.2+(0.98-0.2)*rand(1);
        Track1(:,t)=TransFunction3(w,T)*Track1(:,t-1);
    end
end

for t=t14+1:226
    type_decision4=round(rand(1));
    if type_decision4 == 0
        w=0.2+(0.98-0.2)*rand(1);
        Track1(:,t)=TransFunction3(w,T)*Track1(:,t-1);
    else
        omega=0.2+(0.98-0.2)*rand(1);
        Track1(:,t)=TransFunction4(omega,T)*Track1(:,t-1);
    end
end

% figure
% plot3(Track1(1,:),Track1(4,:),Track1(7,:));grid on;

%% 增补Target1：Bird的运动参数，range，az，el，vradial
Track11=zeros(14,226);
Track11(1:10,:)=Track1;
Track11(11,:)=sqrt(Track1(1,:).^2+Track1(4,:).^2+Track1(7,:).^2);          % range
Track11(12,:)=rad2deg(atan(Track1(4,:)./Track1(1,:)));                     % azimuth
Track11(13,:)=rad2deg(acos(sqrt(Track1(4,:).^2+Track1(1,:).^2)./Track11(11,:))); % elevation
Track11(14,:)=(-Track1(1,:).*Track1(2,:)-Track1(4,:).*Track1(5,:)-Track1(7,:).*Track1(8,:))./Track11(11,:); % v_radial
Track11(10,:)=[];
Track11(3,:)=[];
Track11(6,:)=[];
Track11(9,:)=[];
Track11(:,1)=[];
% figure
% plot3(Track11(1,:),Track11(4,:),Track11(7,:));grid on;                   % 舍弃初始列与第10行的w后，Track11大小为(13,600)
data_bird(i,:,:)=Track11.';
label_bird(i,1)=1;
end
%% 数据归一化
data_nor_bird=zeros(Num,225,10);  
% for i=1:4*2*Num
%     data_nor(i,:,:)=(data(i,:,:)-min(min(data(i,:,:))))/(max(max(data(i,:,:)))-min(min(data(i,:,:)))); 
%     i
% end
data_nor_bird(:,:,:)=(data_bird(:,:,:)-min(min(min(data_bird(:,:,:)))))/(max(max(max(data_bird(:,:,:))))-min(min(min(data_bird(:,:,:))))); 
save('E:\\渥太华大学\\RHC UAV仿真数据实验\\Meng-simulated UAV&Bird data\\Mixdata\\data_bird.mat','data_bird','-v7.3')
save('E:\\渥太华大学\\RHC UAV仿真数据实验\\Meng-simulated UAV&Bird data\\Mixdata\\data_nor_bird.mat','data_nor_bird','-v7.3')
save('E:\\渥太华大学\\RHC UAV仿真数据实验\\Meng-simulated UAV&Bird data\\Mixdata\\label_bird.mat','label_bird','-v7.3')
%% 拼接生成的Bird与load的Drone
load('E:\\渥太华大学\\RHC UAV仿真数据实验\\Meng-simulated UAV&Bird data\\3Drones\\data.mat','data')
load('E:\\渥太华大学\\RHC UAV仿真数据实验\\Meng-simulated UAV&Bird data\\3Drones\\data_nor.mat','data_nor')
load('E:\\渥太华大学\\RHC UAV仿真数据实验\\Meng-simulated UAV&Bird data\\3Drones\\label.mat','label')
data_mix=zeros(Num+3,225,10);
data_nor_mix=zeros(Num+3,225,10);
label_mix=zeros(Num+3,2);

data_mix(1:3,:,:)=data;
data_mix(4:Num+3,:,:)=data_bird;

data_nor_mix(1:3,:,:)=data_nor;
data_nor_mix(4:Num+3,:,:)=data_nor_bird;

label_mix(1:3,:)=label;
label_mix(4:Num+3,:)=label_bird;
%% 随机采样打乱data的顺序
Index=randperm(Num+3);
data_mix(:,:,:)=data_mix(Index,:,:);
data_nor_mix(:,:,:)=data_nor_mix(Index,:,:);
size(data_mix)
label_mix(:,:)=label_mix(Index,:);
%% 保存数据
save('E:\\渥太华大学\\RHC UAV仿真数据实验\\Meng-simulated UAV&Bird data\\Mixdata\\data_mix.mat','data_mix','-v7.3')
save('E:\\渥太华大学\\RHC UAV仿真数据实验\\Meng-simulated UAV&Bird data\\Mixdata\\data_nor_mix.mat','data_nor_mix','-v7.3')
save('E:\\渥太华大学\\RHC UAV仿真数据实验\\Meng-simulated UAV&Bird data\\Mixdata\\label_mix.mat','label_mix','-v7.3')







