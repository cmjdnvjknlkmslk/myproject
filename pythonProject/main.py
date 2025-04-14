
# -*- coding:utf-8 -*-
import cv2
from HandTrackingModule import HandDetector
import csv
import matplotlib.pyplot as plt
#import DobotDllType as dType
#from Biaoding_zhuanhuan import HandInEyeCalibration
class Main:
    def __init__(self):

        #self.camera = cv2.VideoCapture(0,cv2.CAP_DSHOW)


        #self.camera = cv2.VideoCapture("D:\\videosave\\MV-CS050-10GC-PRO (DA1351773)\\Video_20240806165509244.avi")
        self.camera = cv2.VideoCapture("D:\\videosave\\MV-CS050-10GC-PRO (DA1351773)\\Video_20240703165023648.avi")
        self.detector = HandDetector()
        # self.camera.set(3, 1280)
        # self.camera.set(4, 720)


    def Gesture_recognition(self):
        #self.detector = HandDetector()
        xList = []
        yList = []
        frame_counter = 0  # 用于记录帧数
        with open('index_finger_coordinates.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Frame', 'X', 'Y'])
            while True:
             frame,img = self.camera.read()
             if not frame:
                  break
             img = cv2.resize(img, (1280, 720))
             img = self.detector.findHands(img)
             x,y,lmList, bbox = self.detector.findPosition(img)
             xList.append(x)
             yList.append(y)
             cv2.putText(img, f"Frame: {frame_counter}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
             if frame_counter < len(xList):
                 x = int(xList[frame_counter])
                 y = int(yList[frame_counter])
                 cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
             frame_counter += 1

             cv2.imshow("camera", img)
             if cv2.waitKey(10) & 0xFF == 27:
                break
        # 写入所有帧的坐标数据
        with open('index_finger_coordinates.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Frame', 'X', 'Y'])
            for frame_num, (x, y) in enumerate(zip(xList, yList)):
                    writer.writerow([frame_num, x, y])



        plt.plot(range(len(xList)), xList, label='x')
        plt.plot(range(len(yList)), yList, label='y')
        plt.xlabel('Frame Index')
        plt.ylabel('Coordinate Position')
        plt.title('Coordinate Position of x and y')
        plt.legend()
        plt.show()
        # 测试处理坐标的函数
    # file_path = 'index_finger_coordinates.csv'
    # self.process_coordinates(file_path)


    def process_coordinates(file_path):
        with open(file_path, mode='r') as file:
            reader = csv.reader(file)
            next(reader)  # 跳过标题行
            prev_x, prev_y = None, None
            valid_frames = 0  # 记录有效帧数
            sum_x, sum_y = 0, 0 # 用于累加坐标值
            for row in reader:
                frame_num, x, y = map(float, row)
                # 如果坐标不为 0，则继续处理
                if x != 0 and y != 0:
                    if prev_x is not None and prev_y is not None:
                        diff_x = abs(x - prev_x)
                        diff_y = abs(y - prev_y)
                        if diff_x < 100 and diff_y < 100:
                            valid_frames += 1
                            sum_x += x
                            sum_y += y
                            print(f'Frame {frame_num}: X = {x}, Y = {y}')
                    prev_x, prev_y = x, y
            #api = dType.CDLL("D:\\anaconda\\shoushirecognition\\DobotDll.dll", winmode=0)
            #state = dType.ConnectDobot(api, "COM3", 115200)[0]


            if valid_frames > 0:
                avg_x = round(sum_x / valid_frames, 1)
                avg_y = round(sum_y / valid_frames, 1)
                print(f'Number of valid frames: {valid_frames}')
                print(f'Average X coordinate: {avg_x}')
                print(f'Average Y coordinate: {avg_y}')
                #S=HandInEyeCalibration()
                #x,y=S.get_points_robot(avg_x, avg_y)
                #print(x)
                #print("----")
                #print(y)
                #dType.SetPTPCmd(api, 0, 1, x, y, -46, 0, isQueued=1)
                #print("----")
            else:
                print("No valid frames found.")
            return valid_frames

    # 测试处理坐标的函数
    file_path = 'index_finger_coordinates.csv'
    valid_frame_count = process_coordinates(file_path)
    print(f'Number of valid frames: {valid_frame_count}')
    # def process_coordinates(file_path):
    #     with open(file_path, mode='r') as file:
    #         reader = csv.reader(file)
    #         next(reader)  # 跳过标题行
    #         prev_x, prev_y = None, None
    #         valid_frames = []  # 保存有效帧的列表
    #         current_batch = []  # 当前批次的有效帧
    #         batch_count = 1  # 批次计数器
    #         for row in reader:
    #             frame_num, x, y = map(float, row)
    #             # 如果坐标不为 0，则继续处理
    #             if x != 0 and y != 0:
    #                 if prev_x is not None and prev_y is not None:
    #                     diff_x = abs(x - prev_x)
    #                     diff_y = abs(y - prev_y)
    #                     if diff_x < 100 and diff_y < 100:
    #                         current_batch.append((x, y))
    #                     else:
    #                         # 计算当前批次的平均坐标并保存
    #                         if current_batch:
    #                             avg_x = sum(coord[0] for coord in current_batch) / len(current_batch)
    #                             avg_y = sum(coord[1] for coord in current_batch) / len(current_batch)
    #                             print(f'Average coordinates of batch {batch_count}:')
    #                             print(f'Average X coordinate: {avg_x:.2f}')
    #                             print(f'Average Y coordinate: {avg_y:.2f}')
    #                             valid_frames.append((avg_x, avg_y))
    #                             current_batch = []  # 重置当前批次
    #                             batch_count += 1  # 增加批次计数器
    #                 prev_x, prev_y = x, y
    #
    #         # 处理最后一批坐标
    #         if current_batch:
    #             avg_x = sum(coord[0] for coord in current_batch) / len(current_batch)
    #             avg_y = sum(coord[1] for coord in current_batch) / len(current_batch)
    #             print(f'Average coordinates of batch {batch_count}:')
    #             print(f'Average X coordinate: {avg_x:.2f}')
    #             print(f'Average Y coordinate: {avg_y:.2f}')
    #             valid_frames.append((avg_x, avg_y))

            # if valid_frames:
            #     # 计算所有有效帧的平均坐标
            #     avg_x = sum(coord[0] for coord in valid_frames) / len(valid_frames)
            #     avg_y = sum(coord[1] for coord in valid_frames) / len(valid_frames)
            #     print("Average coordinates of all valid frames:")
            #     print(f'Average X coordinate: {avg_x}')
            #     print(f'Average Y coordinate: {avg_y}')
            # else:
            #     print("No valid frames found.")
            #
            # return valid_frames

    #测试处理坐标的函数
    # file_path = 'index_finger_coordinates.csv'
    # valid_frames = process_coordinates(file_path)


if __name__ == '__main__':
    Solution = Main()
    Solution.Gesture_recognition()
