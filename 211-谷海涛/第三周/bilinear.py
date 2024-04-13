import numpy as np
import cv2

def bilinear_interpolation(img, new_size):
    ori_height, ori_width, channels = img.shape
    new_height, new_width = new_size

    if ori_width == new_width and ori_height == new_height:
        return img.copy()
    else:
        scale_height, scale_width = float(ori_height/new_height), float(ori_width/new_width)  #计算高度和宽度方向的缩放比例
        empty_image = np.zeros((new_width, new_height, channels), img.dtype)     #生成全0的空白数组
        for i in range(channels):
            for new_y in range(new_height):
                for new_x in range(new_width):
                    ori_x = (new_x + 0.5) * scale_width -0.5
                    ori_y = (new_y + 0.5) * scale_height -0.5

                    ori_x0 = int(np.floor(ori_x))
                    ori_x1 = min(ori_x0 + 1, ori_width - 1)
                    ori_y0 = int(np.floor(ori_y))
                    ori_y1 = min(ori_y0 + 1, ori_width - 1)

                    tmp0 = (ori_x1 - ori_x) * img[ori_y0, ori_x0, i] + (ori_x - ori_x0) * img[ori_y0, ori_x1, i]
                    tmp1 = (ori_x1 - ori_x) * img[ori_y1, ori_x0, i] + (ori_x - ori_x0) * img[ori_y1, ori_x1, i]
                    empty_image[new_y, new_x] = int((ori_y1 - ori_y) * tmp0 + (ori_y - ori_y0) * tmp1)

    return empty_image



img = cv2.imread('1.png')
new_size = [1000, 1000]
new_img = bilinear_interpolation(img, new_size)
cv2.imwrite('bilinear_interpolation.png', new_img)












