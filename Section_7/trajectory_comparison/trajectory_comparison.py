# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 13:17:18 2021

@author: CC-i7-11700
"""
# https://blog.csdn.net/jasonzhoujx/article/details/81780774

# 数据读取

def read_data(file_name, index_x, index_y, index_z):
    xyz = []
    tx = []
    ty = []
    tz = []
    
    with open(file_name, 'r', encoding='utf-8') as txtfile:
        rows = txtfile.readlines()
    for row in rows:    
        temp = row[0:-1].split(' ')
        for ii in range(len(temp)):
            if ii == index_x:
                tx.append(float(temp[ii]))
            elif ii == index_y:
                ty.append(float(temp[ii]))
            elif ii == index_z:
                tz.append(float(temp[ii]))
    xyz.append(tx)
    xyz.append(ty)
    xyz.append(tz)
    return xyz

cam_pose_tum = read_data("cam_pose_tum.txt", 1,2,3)
cam_pose = read_data("cam_pose.txt", 5,6,7)
imu_pose = read_data("imu_pose.txt", 5,6,7)
imu_pose_noise = read_data("imu_pose_noise.txt", 5,6,7)
standard_deviation_x0 = read_data("standard_deviation_x0.txt", 1,2,3)
basic_standard_deviation = read_data("basic_standard_deviation.txt", 1,2,3)


# 数据检验
for ii in range(len(imu_pose[0])):
    if imu_pose[0][ii] != imu_pose_noise[0][ii] or imu_pose[1][ii] != imu_pose_noise[1][ii] or imu_pose[2][ii] != imu_pose_noise[2][ii]:
        print(imu_pose[0][ii]-imu_pose_noise[0][ii],
              imu_pose[1][ii]-imu_pose_noise[1][ii],
              imu_pose[2][ii]-imu_pose_noise[2][ii])

for ii in range(len(cam_pose[0])):
    if abs(cam_pose[0][ii] - cam_pose_tum[0][ii]) < 1e-4 and abs(cam_pose[1][ii] - cam_pose_tum[1][ii]) < 1e-4 and abs(cam_pose[2][ii] - cam_pose_tum[2][ii]) < 1e-4:
        pass
    else:
        print(cam_pose[0][ii]-cam_pose_tum[0][ii],
              cam_pose[1][ii]-cam_pose_tum[1][ii],
              cam_pose[2][ii]-cam_pose_tum[2][ii])
 


# 绘图
# from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt


fig = plt.figure()
ax0 = fig.add_subplot(111, projection='3d')
# ax0 = fig.gca(projection='3d')

ax0.plot(cam_pose[0], cam_pose[1], cam_pose[2], color='red', linestyle='-.', label='cam_pose')
ax0.plot(basic_standard_deviation[0], basic_standard_deviation[1], basic_standard_deviation[2], color='green', linestyle='-', label='basic_standard_deviation')
ax0.plot(standard_deviation_x0[0], standard_deviation_x0[1], standard_deviation_x0[2], color='blue', linestyle='-', label='standard_deviation_x0')

# ax0.plot3D(tx, ty, tz, color='r')

ax0.set_title("Trajectory")
ax0.set_xlabel("tx")
ax0.set_ylabel("ty")
ax0.set_zlabel("tz")
ax0.legend()













