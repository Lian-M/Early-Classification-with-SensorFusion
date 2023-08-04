%%%%%%%%%%%%%%%%%%%%%UAV与Bird的运动学特征生成仿真%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Randomly Simulation of 480 bird instances and 480 bird instances 
%%% based on <<Hybrid AI-enabled Method for UAS and Bird Detection and Classification>>
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
Num=480;                                                                     % 生成120对Bird和UAV的实例
data=zeros(Num*2,225,13);                                                % 并行循环120次，每次1个Bird&1个UAV,将长度为600的时间序列分成40个长度为15的时间序列，（9600，15，13）=（B,T,V）
label=zeros(Num*2,2);                                                   % 标签采用one-hot编码，（1，0）代表鸟，（0，1）代表UAV
% data=zeros(4*Num*2,225,13);                                                 % 并行循环120次，每次1个Bird&1个UAV,将长度为60的时间序列分成4个长度为15的时间序列，（4*120*2，15，13）=（B,T,V）
% label=zeros(4*Num*2,2);                                                    % 标签采用one-hot编码，（1，0）代表鸟，（0，1）代表UAV
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
%% Target2:UAV初始状态
x2=100+(150-100)*rand(1);                                                  % ~U(100,150) 
vx2=sqrt(3)*randn(1)+7;                                                    % v~N(7,3)
ax2=sqrt(3)*randn(1)+10;                                                   % a~N(10,3)
y2=100+(150-100)*rand(1);
vy2=sqrt(3)*randn(1)+7;
ay2=sqrt(3)*randn(1)+10;
z2=100+(150-100)*rand(1);
vz2=sqrt(3)*randn(1)+7;
az2=sqrt(3)*randn(1)+10;
w2=0+(0.09-0)*rand(1);                                                     % w&omega~U(0.2,0.98)
%% 转移矩阵设定:TransFunction1.m TransFunction2.m TransFunction3.m TransFunction4.m
% T=1;                                                                       % 时间步长
% % Fcv=[1 T 0;0 1 0;0 0 0];
% % Fcv3=diag([1 1 0 1 1 0 1 1 0 1])+diag([T 0 0 T 0 0 T 0 0],1);
% Fcv3=TransFunction1(T);
% % Fca=[1 T 0.5*T^2;0 1 T;0 0 1];
% % Fca3=diag([1 1 1 1 1 1 1 1 1 1])+diag([T T 0 T T 0 T T 0],1)+diag([0.5*T^2 0 0 0.5*T^2 0 0 0.5*T^2 0],2);
% Fca3=TransFunction2(T);
% w=0.2+(0.98-0.2)*rand(1);
% omega=0.2+(0.98-0.2)*rand(1);
% % Fhct=[1 sin(w*T)/w 0 0 (cos(w*T)-1)/w 0;...
% %       0 cos(w*T) 0 0 -sin(w*T) 0;...
% %       0 -w*sin(w*T) 0 0 -w*cos(w*T) 0;...
% %       0 (1-cos(w*T))/w 0 1 cos(w*T)/w 0;...
% %       0 sin(w*T) 0 0 cos(w*T) 0;...
% %       0 w*cos(w*T) 0 0 -w*sin(w*T) 0];
% % Fhct3=[1 sin(w*T)/w 0 0 (cos(w*T)-1)/w 0 0 0 0 0;...
% %       0 cos(w*T) 0 0 -sin(w*T) 0 0 0 0 0;...
% %       0 -w*sin(w*T) 0 0 -w*cos(w*T) 0 0 0 0 0;...
% %       0 (1-cos(w*T))/w 0 1 cos(w*T)/w 0 0 0 0 0;...
% %       0 sin(w*T) 0 0 cos(w*T) 0 0 0 0 0;...
% %       0 w*cos(w*T) 0 0 -w*sin(w*T) 0 0 0 0 0;...
% %       0 0 0 0 0 0 1 T 0 0;...
% %       0 0 0 0 0 0 0 1 0 0;...
% %       0 0 0 0 0 0 0 0 0 0;...
% %       0 0 0 0 0 0 0 0 0 1];
% Fhct3=TransFunction3(w,T);
% % F3dct=[1 sin(omega*T)/omega (1-cos(omega*T))/omega^2;...
% %        0 cos(omega*T) sin(omega*T)/omega;...
% %        0 -omega*sin(omega*T) cos(omega*T)];
% % F3dct3=diag([1 cos(omega*T) cos(omega*T) 1 cos(omega*T) cos(omega*T) 1 cos(omega*T) cos(omega*T) 1])+...
% %     diag([sin(omega*T)/omega sin(omega*T)/omega 0 sin(omega*T)/omega sin(omega*T)/omega 0 sin(omega*T)/omega sin(omega*T)/omega 0],1)+...
% %     diag([(1-cos(omega*T))/omega^2 0 0 (1-cos(omega*T))/omega^2 0 0 (1-cos(omega*T))/omega^2 0],2)+...
% %     diag([0 -omega*sin(omega*T) 0 0 -omega*sin(omega*T) 0 0 -omega*sin(omega*T) 0],-1);
% F3dct3=TransFunction4(omega,T);
%% 分阶段创建Target1:Bird的时间序列
t11=79;
t12=98;
t13=154;
t14=191;
Track1=zeros(10,226);
% t11=21;
% t12=26;
% t13=41;
% t14=51;
% Track1=zeros(10,61);
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
T=0.1;
% T=1;
for t=2:t11
    type_decision1=round(rand(1));
    if type_decision1 == 0
        Track1(:,t)=TransFunction1(T)*Track1(:,t-1);
    else
        Track1(:,t)=TransFunction2(T)*Track1(:,t-1);
    end
end
% type_decision1=round(rand(1));
% if type_decision1 == 0
%    for t=2:t11
%        Track1(:,t)=TransFunction1(T)*Track1(:,t-1);
%    end
% else
%     for t=2:t11
%         Track1(:,t)=TransFunction2(T)*Track1(:,t-1);
%     end
% end

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
% type_decision2=round(rand(1));
% if type_decision2 == 0
% %     w=0.2+(0.98-0.2)*rand(1);
%    for t=t11+1:t12
%        w=0.2+(0.98-0.2)*rand(1);
%        Track1(:,t)=TransFunction3(w,T)*Track1(:,t-1);
%    end
% else
% %     omega=0.2+(0.98-0.2)*rand(1);
%     for t=t11+1:t12
%         omega=0.2+(0.98-0.2)*rand(1);
%         Track1(:,t)=TransFunction4(omega,T)*Track1(:,t-1);
%     end
% end

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
% type_decision3=round(rand(1));
% if type_decision3 == 0
%     for t=t13+1:t14
%         Track1(:,t)=TransFunction2(T)*Track1(:,t-1);
%     end
% else
% %     w=0.2+(0.98-0.2)*rand(1);
%     for t=t13+1:t14
%         w=0.2+(0.98-0.2)*rand(1);
%         Track1(:,t)=TransFunction3(w,T)*Track1(:,t-1);
%     end
% end

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
% type_decision4=round(rand(1));
% if type_decision4 == 0
% %     w=0.2+(0.98-0.2)*rand(1);
%     for t=t14+1:61
%         w=0.2+(0.98-0.2)*rand(1);
%         Track1(:,t)=TransFunction3(w,T)*Track1(:,t-1);
%     end
% else
% %     omega=0.2+(0.98-0.2)*rand(1);
%     for t=t14+1:61
%         omega=0.2+(0.98-0.2)*rand(1);
%         Track1(:,t)=TransFunction4(omega,T)*Track1(:,t-1);
%     end
% end
% 
% figure
% plot3(Track1(1,:),Track1(4,:),Track1(7,:));grid on;

%% 分阶段创建Target2:UAV的时间序列
t21=79;
t22=154;
t23=191;
Track2=zeros(10,226);
% t21=21;
% t22=41;
% t23=51;
% Track2=zeros(10,61);
Track2(1,1)=x2;
Track2(2,1)=vx2;
Track2(3,1)=ax2;
Track2(4,1)=y2;
Track2(5,1)=vy2;
Track2(6,1)=ay2;
Track2(7,1)=z2;
Track2(8,1)=vz2;
Track2(9,1)=az2;
Track2(10,1)=w2;
T=0.1;
% T=1;
% for t=2:t21
%     type_decision=round(rand(1));
%     if type_decision == 0
%         Track2(:,t)=TransFunction1(T)*Track2(:,t-1);
%     else
%         Track2(:,t)=TransFunction2(T)*Track2(:,t-1);
%     end
% end
type_decision5=round(rand(1));
if type_decision5 == 0
    for t=2:t21
        Track2(:,t)=TransFunction1(T)*Track2(:,t-1);
    end
else
    for t=2:t21
        Track2(:,t)=TransFunction2(T)*Track2(:,t-1);
    end
end

% for t=t21+1:t22
%     type_decision=round(rand(1));
%     if type_decision == 0
%         w=0+(0.09-0)*rand(1);
%         Track2(:,t)=TransFunction3(w,T)*Track2(:,t-1);
%     else
%         omega=0+(0.09-0)*rand(1);
%         Track2(:,t)=TransFunction4(omega,T)*Track2(:,t-1);
%     end
% end
type_decision6=round(rand(1));
if type_decision6 == 0
    w=0+(0.09-0)*rand(1);
    for t=t21+1:t22
%         w=0+(0.09-0)*rand(1);
        Track2(:,t)=TransFunction3(w,T)*Track2(:,t-1);
    end
else
    omega=0+(0.09-0)*rand(1);
    for t=t21+1:t22
%         omega=0+(0.09-0)*rand(1);
        Track2(:,t)=TransFunction4(omega,T)*Track2(:,t-1);
    end
end

% for t=t22+1:t23
%     type_decision=round(rand(1));
%     if type_decision == 0
%         Track2(:,t)=TransFunction2(T)*Track2(:,t-1);
%     else
%         w=0+(0.09-0)*rand(1);
%         Track2(:,t)=TransFunction3(w,T)*Track2(:,t-1);
%     end
% end
type_decision7=round(rand(1));
if type_decision7 == 0
    for t=t22+1:t23
        Track2(:,t)=TransFunction2(T)*Track2(:,t-1);
    end
else
    w=0+(0.09-0)*rand(1);
    for t=t22+1:t23
%         w=0+(0.09-0)*rand(1);
        Track2(:,t)=TransFunction3(w,T)*Track2(:,t-1);
    end
end

for t=t23+1:226
    Track2(:,t)=TransFunction2(T)*Track2(:,t-1);
end
% figure
% plot3(Track2(1,:),Track2(4,:),Track2(7,:));grid on;
%% 增补Target1：Bird的运动参数，range，az，el，vradial
Track11=zeros(14,226);
% Track11=zeros(14,61);
Track11(1:10,:)=Track1;
Track11(11,:)=sqrt(Track1(1,:).^2+Track1(4,:).^2+Track1(7,:).^2);          % range
Track11(12,:)=rad2deg(atan(Track1(4,:)./Track1(1,:)));                     % azimuth
Track11(13,:)=rad2deg(acos(sqrt(Track1(4,:).^2+Track1(1,:).^2)./Track11(11,:))); % elevation
Track11(14,:)=(-Track1(1,:).*Track1(2,:)-Track1(4,:).*Track1(5,:)-Track1(7,:).*Track1(8,:))./Track11(11,:); % v_radial
Track11(10,:)=[];
Track11(:,1)=[];
% figure
% plot3(Track11(1,:),Track11(4,:),Track11(7,:));grid on;                   % 舍弃初始列与第10行的w后，Track11大小为(13,600)
%% 增补Target2：UAV的运动参数，range，az，el，vradial
Track22=zeros(14,226);
% Track22=zeros(14,61);
Track22(1:10,:)=Track2;
Track22(11,:)=sqrt(Track2(1,:).^2+Track2(4,:).^2+Track2(7,:).^2);          % range
Track22(12,:)=rad2deg(atan(Track2(4,:)./Track2(1,:)));                     % azimuth
Track22(13,:)=rad2deg(acos(sqrt(Track2(4,:).^2+Track2(1,:).^2)./Track22(11,:))); % elevation
Track22(14,:)=(-Track2(1,:).*Track2(2,:)-Track2(4,:).*Track2(5,:)-Track2(7,:).*Track2(8,:))./Track22(11,:); % v_radial
Track22(10,:)=[];
Track22(:,1)=[];
% figure
% plot3(Track22(1,:),Track22(4,:),Track22(7,:));grid on;                   % 舍弃初始列与第10行的w后，Track22大小为(13,600)
%% 将生成的每个Bird和UAV轨迹数据经分割后放入矩阵data，并生成对应标签放入矩阵label
data_temp=zeros(2,225,13);
label_temp=zeros(2,2);
% data_temp=zeros(4*2,15,13);
% label_temp=zeros(4*2,2);
% for k=1:4
%     data_temp(k*2-1,:,:)=Track11(:,(k-1)*15+1:k*15).'; %(15,13)
%     label_temp(k*2-1,:)=[1 0];
%     data_temp(k*2,:,:)=Track22(:,(k-1)*15+1:k*15).';
%     label_temp(k*2,:)=[0 1];
% end
data_temp(1,:,:)=Track11.'; %(15,13)
label_temp(1,:)=[1 0];
data_temp(2,:,:)=Track22.';
label_temp(2,:)=[0 1];

data((i-1)*2+1:i*2,:,:)=data_temp;
label((i-1)*2+1:i*2,:)=label_temp;
% data((i-1)*4*2+1:i*4*2,:,:)=data_temp;
% label((i-1)*4*2+1:i*4*2,:)=label_temp;
i
end
%% 随机采样打乱data的顺序
Index=randperm(Num*2);
data(:,:,:)=data(Index,:,:);
data(:,:,3)=[];
data(:,:,6)=[];
data(:,:,9)=[];
size(data)
label(:,:)=label(Index,:);
%% 数据归一化
data_nor=zeros(Num*2,225,10);  
% for i=1:4*2*Num
%     data_nor(i,:,:)=(data(i,:,:)-min(min(data(i,:,:))))/(max(max(data(i,:,:)))-min(min(data(i,:,:)))); 
%     i
% end
data_nor(:,:,:)=(data(:,:,:)-min(min(min(data(:,:,:)))))/(max(max(max(data(:,:,:))))-min(min(min(data(:,:,:))))); 
%% 保存数据
save('E:\\渥太华大学\\RHC UAV仿真数据实验\\Meng-simulated UAV&Bird data\\225\\data.mat','data','-v7.3')
save('E:\\渥太华大学\\RHC UAV仿真数据实验\\Meng-simulated UAV&Bird data\\225\\data_nor.mat','data_nor','-v7.3')
save('E:\\渥太华大学\\RHC UAV仿真数据实验\\Meng-simulated UAV&Bird data\\225\\label.mat','label','-v7.3')
toc
disp(['运行时间：',num2str(toc)]);

% [max_val, position_max]=max(data(:));
% [x,y,z]=ind2sub(size(data),position_max);
% [min_val, position_min]=min(data(:));
% [r,s,t]=ind2sub(size(data),position_min);




