# 导入相关库
import cv2
import numpy as np

# 读取图片
image = cv2.imread('test222.jpg')
cv2.imshow('Original Image', image)

# 让我们使用新的宽度和高度缩小图像
down_width = 1280
down_height = 720
down_points = (down_width, down_height)
resized_down = cv2.resize(image, down_points, interpolation= cv2.INTER_LINEAR)

# 让我们使用新的宽度和高度来增加图像尺寸
# up_width = 600
# up_height = 400
# up_points = (up_width, up_height)
# resized_up = cv2.resize(image, up_points, interpolation= cv2.INTER_LINEAR)

# 显示图像
cv2.imshow('Resized Down by defining height and width', resized_down)
cv2.waitKey()
# cv2.imshow('Resized Up image by defining height and width', resized_up)
# cv2.waitKey()
# 将调整后的图像保存到文件
cv2.imwrite('D:\\anaconda\\shoushirecognition\\resized_image.jpg', resized_down)

#按下任意键退出
cv2.destroyAllWindows()
