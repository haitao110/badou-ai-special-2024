import cv2
import numpy as np


def average_hash(img):
    img = cv2.resize(img, (8, 8), interpolation=cv2.INTER_CUBIC)  #原图转换为8x8
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  #转灰度
    s = 0
    hash_str = ''
    for i in range(8):
        for j in range(8):
            s += gray[i][j]  #计算像素和

    average_value = s / 64  #计算像素均值
    for i in range(8):
        for j in range(8):
            if gray[i][j] > average_value:  #像素值大于平均值时候hash字符串为1，否则为0
                hash_str += '1'
            else:
                hash_str += '0'

    return hash_str


def difference_hash(img):
    img = cv2.resize(img, (9, 8), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hash_str = ''
    for i in range(8):
        for j in range(8):
            if gray[i][j] > gray[i][j + 1]:
                hash_str += '1'
            else:
                hash_str += '0'

    return hash_str


def compare_hash(hash_1, hash_2):
    n = 0
    if len(hash_1) != len(hash_2):  #长度不同直接返回
        return -1
    else:
        for i in range(len(hash_1)):  #比较两个字符串的每一位
            if hash_1[i] != hash_2[i]:
                n += 1

    return n


img1 = cv2.imread('1.png')
img2 = cv2.imread('2.png')
hash_1 = average_hash(img1)
hash_2 = average_hash(img2)
print(hash_1)
print(hash_2)
n = compare_hash(hash_1, hash_2)
print("均值哈希相似度为：{}".format(n))

img1 = cv2.imread('1.png')
img2 = cv2.imread('2.png')
hash_1 = difference_hash(img1)
hash_2 = difference_hash(img2)
print(hash_1)
print(hash_2)
n = compare_hash(hash_1, hash_2)
print("差值哈希相似度为：{}".format(n))
