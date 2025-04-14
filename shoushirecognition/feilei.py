import os, cv2

def file_name(file_dir):

    # dirs = os.listdir(file_dir)
    # for dir in dirs:
    #     file_dir_d = file_dir + "\\" + str(dir)
    #     print(file_dir_d)
    #     files = os.listdir(file_dir_d)
        # 找到每一个视频文件
        # for file in files:
        #     file_dir_e = file_dir_d + "\\" + str(file)
    file_n = os.path.splitext(file_dir)[0]
    cap = cv2.VideoCapture(file_dir)
    success, _ = cap.read()
    # 重新合成的视频在原文件夹，如果需要分开，可以修改file_n
    videowriter = cv2.VideoWriter(file_n+".avi", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 15, (1280,720))
    while success:
        success, img1 = cap.read()
        try:
            img = cv2.resize(img1, (1280, 720), interpolation=cv2.INTER_LINEAR)
            videowriter.write(img)
        except:
            break
file_dir="D:\\videosave\\test2.avi"
file_name(file_dir)


