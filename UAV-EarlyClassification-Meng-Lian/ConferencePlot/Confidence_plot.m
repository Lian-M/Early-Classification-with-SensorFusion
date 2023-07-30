clc,clear all
close all
warning off
%% Use fixed random seed for simulation repeatablity.
rng(0)
%% Extracting the confidence

foldername = 'validation';
datafolder = cd;
datafolder = [datafolder,'\',foldername,'\'];
%% Extract confidence of YOLOv7
filename = [datafolder,'Drone123_yolov7_conf.txt'];
fileID = fopen(filename);
C = textscan(fileID,'%d %f %f %f');
fclose(fileID);

drone1_yolov7conf = C{1,2}(1:226,1);
drone2_yolov7conf = C{1,3}(1:226,1);
drone3_yolov7conf = C{1,4}(1:226,1);

t_yolo=[0.1:0.1:22.6];
figure;
plot(t_yolo,drone1_yolov7conf,'-b','LineWidth',1);
hold on
plot(t_yolo,drone2_yolov7conf,':r','LineWidth',1);
hold on
plot(t_yolo,drone3_yolov7conf,'-.c','LineWidth',1);
grid minor;
%title('Confidences achieved by YOLOv7')
xlabel('t(s)');ylabel('Confidence')
legend('Drone1','Drone2','Drone3')
%% Extract confidence of RHC
filename = [datafolder,'Drone1_conf.txt'];
fileID = fopen(filename);
C = textscan(fileID,'%f %f');
fclose(fileID);

drone1_RHCconf_bird = C{1,1}(1:225,1);
drone1_RHCconf_drone = C{1,2}(1:225,1);

filename = [datafolder,'Drone2_conf.txt'];
fileID = fopen(filename);
C = textscan(fileID,'%f %f');
fclose(fileID);

drone2_RHCconf_bird = C{1,1}(1:225,1);
drone2_RHCconf_drone = C{1,2}(1:225,1);

filename = [datafolder,'Drone3_conf.txt'];
fileID = fopen(filename);
C = textscan(fileID,'%f %f');
fclose(fileID);

drone3_RHCconf_bird = C{1,1}(1:225,1);
drone3_RHCconf_drone = C{1,2}(1:225,1);

t_rhc=[0.1:0.1:22.5];
figure;
plot(t_rhc,drone1_RHCconf_bird,'-.r','LineWidth',1);
hold on
plot(t_rhc,drone1_RHCconf_drone,'-b','LineWidth',1);
grid minor;
title('Confidence of drone1 by RHC')
xlabel('t(s)');ylabel('Confidence')
legend('bird','drone')

figure;
plot(t_rhc,drone2_RHCconf_bird,'-.r','LineWidth',1);
hold on
plot(t_rhc,drone2_RHCconf_drone,'-b','LineWidth',1);
grid minor;
title('Confidence of drone2 by RHC')
xlabel('t(s)');ylabel('Confidence')
legend('bird','drone')

figure;
plot(t_rhc,drone3_RHCconf_bird,'-.r','LineWidth',1);
hold on
plot(t_rhc,drone3_RHCconf_drone,'-b','LineWidth',1);
grid minor;
title('Confidence of drone3 by RHC')
xlabel('t(s)');ylabel('Confidence')
legend('bird','drone')
%%
drone1_yolo = drone1_yolov7conf(1:225);
drone2_yolo = drone2_yolov7conf(1:225);
drone3_yolo = drone3_yolov7conf(1:225);
% fused_drone1_conf = max(drone1_RHCconf_drone, drone1_yolo);
% fused_drone2_conf = max(drone2_RHCconf_drone, drone2_yolo);
% fused_drone3_conf = max(drone3_RHCconf_drone, drone3_yolo);
% figure;
% plot(t_rhc,fused_drone1_conf,'-b','LineWidth',1);
% hold on
% plot(t_rhc,fused_drone2_conf,':r','LineWidth',1);
% hold on
% plot(t_rhc,fused_drone3_conf,'-.c','LineWidth',1);
% grid minor;
% title('Fused confidence')
% xlabel('t(s)');ylabel('Fused confidence')
% legend('Drone1','Drone2','Drone3')
fused_drone1_conf = zeros(1,length(t_rhc));
fused_drone2_conf = zeros(1,length(t_rhc));
fused_drone3_conf = zeros(1,length(t_rhc));
omega = 0.7; 
for i=1:length(t_rhc)
    if drone1_yolo(i,1) >= omega
        fused_drone1_conf(1,i) = drone1_yolo(i,1);
    else
        fused_drone1_conf(1,i) = max(drone1_yolo(i,1),drone1_RHCconf_drone(i,1));
    end
    if drone2_yolo(i,1) >= omega
        fused_drone2_conf(1,i) = drone2_yolo(i,1);
    else
        fused_drone2_conf(1,i) = max(drone2_yolo(i,1),drone2_RHCconf_drone(i,1));
    end
    if drone3_yolo(i,1) >= omega
        fused_drone3_conf(1,i) = drone3_yolo(i,1);
    else
        fused_drone3_conf(1,i) = max(drone3_yolo(i,1),drone3_RHCconf_drone(i,1));
    end
end

figure;
plot(t_rhc,fused_drone1_conf,'-b','LineWidth',1);
hold on
plot(t_rhc,fused_drone2_conf,':r','LineWidth',1);
hold on
plot(t_rhc,fused_drone3_conf,'-.c','LineWidth',1);
grid minor;
title('Fused confidence')
xlabel('t(s)');ylabel('Fused confidence')
legend('Drone1','Drone2','Drone3')