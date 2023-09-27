# import cv2
# import numpy as np
# from scipy.fftpack import dct, idct
#
# image = cv2.imread('./Lena_256.bmp', 0)  # 以灰度模式读取图像
#
# dct_image = dct(image)  # 对图像进行DCT变换
# # dft_image = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)  # 对图像进行DFT变换
#
# K = 32  # 选择K个系数
# sorted_coeffs = np.argsort(dct_image, axis=None)[-K:]  # 获取最大的K个系数的索引
#
# mean = 0  # 伪随机数的均值
# std_dev = 10  # 伪随机数的标准差
# watermark_sequence = np.random.normal(mean, std_dev, K)  # 生成服从正态分布的伪随机数序列
#
# for index, coeff in np.ndenumerate(dct_image):
#     if np.any(index == sorted_coeffs):
#         dct_image[index] += watermark_sequence[np.where(sorted_coeffs == index)]
#
# # for i in range(K):
# #     x, y = np.unravel_index(sorted_coeffs[i], dct_image.shape)
# #     dct_image[x, y] += watermark_sequence[i]
#
# watermarked_image = idct(dct_image)  # 对嵌入水印后的DCT系数进行逆DCT变换
#
# extracted_coeffs = dct(watermarked_image)  # 对嵌入水印后的图像进行DCT变换
# extracted_watermark_sequence = extracted_coeffs[sorted_coeffs] - dct_image[sorted_coeffs]  # 提取嵌入的水印序列
# correlation = np.correlate(extracted_watermark_sequence, watermark_sequence)  # 计算提取的水印序列与原始水印序列的相关性
#
#
