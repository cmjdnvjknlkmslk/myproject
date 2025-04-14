import numpy as np
import cv2

# 通过九点标定获取的圆心相机坐标
STC_points_camera = np.array([
    [570, 235],
    [706, 233],
    [846, 234],
    [573, 318],
    [707, 314],
    [847, 312],
    [577, 397],
    [709, 393],
    [851, 395],
])
# 通过九点标定获取的圆心机械臂坐标
STC_points_robot = np.array([
    [194.1506, 10.5948],
    [194.1117, -4.0361],
    [193.7807, -18.5886],
    [178.1745, 11.7058],
    [177.4658, -3.7721],
    [178.1643, -17.8385],
    [160.4256, 10.1676],
    [160.2371, -3.8502],
    [160.7926, -18.2781],
])

# [1090, 680],
#     [1350, 671],
#     [1615, 671],
#     [1095, 899],
#     [1351, 894],
#     [1617, 894],
#     [1103, 1127],
#     [1358, 1125],
#     [1621, 1127],
# 手眼标定方法
class HandInEyeCalibration:

    def get_m(self, points_camera, points_robot):
        """
        取得相机坐标转换到机器坐标的仿射矩阵
        :param points_camera:
        :param points_robot:
        :return:
        """
        # 确保两个点集的数量级不要差距过大，否则会输出None
        m, _ = cv2.estimateAffine2D(points_camera, points_robot)
        return m

    def get_points_robot(self, x_camera, y_camera):
        """
        相机坐标通过仿射矩阵变换取得机器坐标
        :param x_camera:
        :param y_camera:
        :return:
        """
        m = self.get_m(STC_points_camera, STC_points_robot)
        robot_x = (m[0][0] * x_camera) + (m[0][1] * y_camera) + m[0][2]
        robot_y = (m[1][0] * x_camera) + (m[1][1] * y_camera) + m[1][2]
        return robot_x, robot_y