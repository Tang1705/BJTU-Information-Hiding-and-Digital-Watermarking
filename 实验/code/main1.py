import numpy as np
import cv2


def embed_watermark(image_path, k, a):
    # 加载图像
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 执行DCT或DFT变换
    transformed = cv2.dct(np.float32(image))  # 使用DCT变换，或者可以使用cv2.dft()函数进行DFT变换

    # 将变换系数展平
    flat_transformed = transformed.ravel()

    # 获取最大系数的索引
    max_indices = np.unravel_index(np.argsort(np.abs(flat_transformed)), transformed.shape)

    # 选择前k个最大的系数位置
    selected_indices = np.column_stack(max_indices)[-k:]

    # 产生伪随机数序列
    watermark_info = np.random.normal(size=k)

    # 嵌入水印信息
    embedded_transformed = transformed.copy()
    for index, value in zip(selected_indices, watermark_info):
        embedded_transformed[index[0]][index[1]] *= (1 + a * value)

    # 执行逆DCT或逆DFT变换
    embedded_image = cv2.idct(embedded_transformed)  # 使用逆DCT变换，或者可以使用cv2.idft()函数进行逆DFT变换

    return embedded_image, watermark_info, selected_indices


def detect_watermark(embedded_image, original_image, selected_indices, a):
    # 执行DCT或DFT变换
    embedded_transformed = cv2.dct(np.float32(embedded_image))  # 使用DCT变换，或者可以使用cv2.dft()函数进行DFT变换
    original_transformed = cv2.dct(np.float32(original_image))

    # 提取嵌入的水印信息
    extracted_watermark = []
    for _, index in enumerate(selected_indices):
        extracted_watermark.append(
            (embedded_transformed[index[0]][index[1]] - original_transformed[index[0]][index[1]]) / (
                        a * original_transformed[index[0]][index[1]]))

    return extracted_watermark


def calculate_psnr(image1, image2):
    assert image1.shape == image2.shape, ('错误：两个输⼊图像的⼤⼩不⼀致: {image1.shape}, {image2.shape}.')
    mse = np.mean((image1 - image2) ** 2)
    if mse == 0:
        print('两幅图像完全⼀样')
        return 200
    psnr = 10 * np.log10((255 ** 2) / mse)
    return psnr


# 示例用法
image_path = 'Lena_256.bmp'
k = 10  # 选择的DCT或DFT系数个数
a = 0.1  # 嵌入系数

# 嵌入水印
embedded_image, watermark_info, selected_indices = embed_watermark(image_path, k, a)

# 计算嵌入水印后图像的PSNR值
original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
psnr = calculate_psnr(original_image, embedded_image)

# 提取水印
extracted_watermark = detect_watermark(embedded_image, original_image, selected_indices, a)

# 计算提取的水印与原始水印的相关值
correlation = np.corrcoef(watermark_info, extracted_watermark)[0, 1]

print("PSNR:", psnr)
print("Correlation:", correlation)

import pywt
import numpy as np
import cv2


def embed_watermark(image, watermark, k, a):
    # 进行小波变换
    coeffs = pywt.dwt2(image, 'haar')
    LL, (LH, HL, HH) = coeffs

    # 将水印信息嵌入到小波系数中
    watermark_indices = np.unravel_index(np.argsort(np.abs(HH.ravel()), axis=None), HH.shape)
    selected_indices = np.column_stack(watermark_indices)[-k:]
    HH_watermarked = HH.copy()
    for indices, value in zip(selected_indices, watermark):
        HH_watermarked[indices[0]][indices[1]] *= (1 + a * value)

    # 逆小波变换恢复图像
    watermarked_coeffs = (LL, (LH, HL, HH_watermarked))
    watermarked_image = pywt.idwt2(watermarked_coeffs, 'haar')
    watermarked_image = np.clip(watermarked_image, 0, 255).astype(np.uint8)

    return watermarked_image, selected_indices


def extract_watermark(image, original_image, selected_indices, a):
    # 进行小波变换
    coeffs = pywt.dwt2(image, 'haar')
    LL, (LH, HL, HH) = coeffs

    original_coeffs = pywt.dwt2(original_image, 'haar')
    original_LL, (original_LH, original_HL, original_HH) = original_coeffs

    # 提取嵌入的水印信息
    extracted_watermark = []
    for indices in selected_indices:
        extracted_watermark.append((HH[indices[0]][indices[1]] - original_HH[indices[0]][indices[1]]) / (
                a * original_HH[indices[0]][indices[1]]))

    return extracted_watermark


# 加载图像
image = cv2.imread('./Lena_256.bmp', 0)

# 生成水印
k = 10
watermark = np.random.normal(size=k)
a = 0.1

# 嵌入水印
watermarked_image, selected_indices = embed_watermark(image, watermark, k, a)

# 提取水印
extracted_watermark = extract_watermark(watermarked_image, image, selected_indices, a)


def calculate_psnr(image1, image2):
    assert image1.shape == image2.shape, ('错误：两个输⼊图像的⼤⼩不⼀致: {image1.shape}, {image2.shape}.')
    mse = np.mean((image1 - image2) ** 2)
    if mse == 0:
        print('两幅图像完全⼀样')
        return 200
    psnr = 10 * np.log10((255 ** 2) / mse)
    return psnr


psnr = calculate_psnr(image, watermarked_image)
print("PSNR:", psnr)

# 判断水印是否存在
correlation = np.corrcoef(watermark, extracted_watermark)[0, 1]
print("Correlation:", correlation)

threshold = 0.1
if correlation > threshold:
    print("水印存在")
else:
    print("水印不存在")
