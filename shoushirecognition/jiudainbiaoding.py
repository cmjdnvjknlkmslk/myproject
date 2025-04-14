import cv2
import numpy as np

img = cv2.imread('D:\\anaconda\\shoushirecognition\\resized_image.jpg')#读取图片
font_face,font_scale,thickness=cv2.FONT_HERSHEY_SIMPLEX,0.5,1
#鼠标交互
def mouseHandler(event,x,y,flags,param):
    points = (x,y)
    global imgCopy
    #鼠标左键双击事件
    if event == cv2.EVENT_LBUTTONDBLCLK:
#输出坐标
        print(x,y)
        #拷贝一张与原图像格式相同的新图像
        imgCopy = img.copy()
        #拼接文字
        text = '['+str(x)+','+str(y)+']'+str(img[x,y])
        #读取文字（宽，高），下基线
        (t_w,t_h),baseLine = cv2.getTextSize(text,font_face,font_scale,thickness)
        #在鼠标当前位置的左上角显示文字
        cv2.putText(imgCopy,text,(x-t_w,y),font_face,font_scale,(125,125,125))
        cv2.imshow('win',imgCopy)
    #鼠标移动事件
    elif event == cv2.EVENT_MOUSEMOVE:
#显示原图片能使文本框消失
        cv2.imshow('win',img)

cv2.namedWindow('win',0)
#窗口与回调函数绑定
cv2.setMouseCallback('win',mouseHandler)
cv2.imshow('win',img)
cv2.waitKey()