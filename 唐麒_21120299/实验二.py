import cv2
import numpy as np
import matplotlib.pyplot as plt


class DCT_Embed(object):
    def __init__(self, background, watermark, block_size=8, alpha=30):
        b_h, b_w = background.shape[:2]
        w_h, w_w = watermark.shape[:2]
        assert w_h <= b_h / block_size and w_w <= b_w / block_size, \
            "\r\n请确保您的的水印图像尺寸 不大于 背景图像尺寸的1/{:}\r\nbackground尺寸{:}\r\nwatermark尺寸{:}".format(
                block_size, background.shape, watermark.shape
            )

        # 保存参数
        self.block_size = block_size
        # 水印强度控制
        self.alpha = alpha

    def dct_blkproc(self, background):
        """
        对background进行分块，然后进行dct变换，得到dct变换后的矩阵

        :param image: 输入图像
        :param split_w: 分割的每个patch的w
        :param split_h: 分割的每个patch的h
        :return: 经dct变换的分块矩阵、原始的分块矩阵
        """
        background_dct_blocks_h = background.shape[0] // self.block_size  # 高度
        background_dct_blocks_w = background.shape[1] // self.block_size  # 宽度
        background_dct_blocks = np.zeros(shape=(
            (background_dct_blocks_h, background_dct_blocks_w, self.block_size, self.block_size)
        ))  # 前2个维度用来遍历所有block，后2个维度用来存储每个block的DCT变换的值

        h_data = np.vsplit(background, background_dct_blocks_h)  # 垂直方向分成background_dct_blocks_h个块
        for h in range(background_dct_blocks_h):
            block_data = np.hsplit(h_data[h], background_dct_blocks_w)  # 水平方向分成background_dct_blocks_w个块
            for w in range(background_dct_blocks_w):
                a_block = block_data[w]
                background_dct_blocks[h, w, ...] = cv2.dct(a_block.astype(np.float64))  # dct变换
        return background_dct_blocks

    def dct_embed(self, dct_data, watermark):
        """
        将水印嵌入到载体的dct系数中
        :param dct_data: 背景图像（载体）的DCT系数
        :param watermark: 归一化二值图像0-1 (uint8类型)
        :return: 空域图像
        """
        temp = watermark.flatten()
        assert temp.max() == 1 and temp.min() == 0, "为方便处理，请保证输入的watermark是被二值归一化的"

        for h in range(watermark.shape[0]):
            for w in range(watermark.shape[1]):
                if watermark[h, w] == 0:
                    if dct_data[h, w, 4, 1] < dct_data[h, w, 3, 2]:
                        tmp = dct_data[h, w, 4, 1]
                        dct_data[h, w, 4, 1] = dct_data[h, w, 3, 2]
                        dct_data[h, w, 3, 2] = tmp
                else:
                    if dct_data[h, w, 4, 1] > dct_data[h, w, 3, 2]:
                        tmp = dct_data[h, w, 4, 1]
                        dct_data[h, w, 4, 1] = dct_data[h, w, 3, 2]
                        dct_data[h, w, 3, 2] = tmp
                if dct_data[h, w, 4, 1] < dct_data[h, w, 3, 2]:
                    dct_data[h, w, 4, 1] -= self.alpha
                else:
                    dct_data[h, w, 3, 2] -= self.alpha
        return dct_data

    def idct_embed(self, dct_data):
        """
        进行对dct矩阵进行idct变换，完成从频域到空域的变换
        :param dct_data: 频域数据
        :return: 空域数据
        """
        row = None
        result = None
        h, w = dct_data.shape[0], dct_data.shape[1]
        for i in range(h):
            for j in range(w):
                block = cv2.idct(dct_data[i, j, ...])
                row = block if j == 0 else np.hstack((row, block))
            result = row if i == 0 else np.vstack((result, row))
        return result.astype(np.uint8)

    def dct_extract(self, synthesis, watermark_size):
        """
        从嵌入水印的图像中提取水印
        :param synthesis: 嵌入水印的空域图像
        :param watermark_size: 水印大小
        :return: 提取的空域水印
        """
        w_h, w_w = watermark_size
        recover_watermark = np.zeros(shape=watermark_size)
        synthesis_dct_blocks = self.dct_blkproc(background=synthesis)
        for h in range(w_h):
            for w in range(w_w):

                if synthesis_dct_blocks[h, w, 4, 1] < synthesis_dct_blocks[h, w, 3, 2]:
                    recover_watermark[h, w] = 1
                else:
                    recover_watermark[h, w] = 0
        return recover_watermark


def calculate_psnr(image1, image2):
    assert image1.shape == image2.shape, ('错误：两个输⼊图像的⼤⼩不⼀致: {image1.shape}, {image2.shape}.')
    mse = np.mean((image1 - image2) ** 2)
    if mse == 0:
        print('两幅图像完全⼀样')
        return 200
    psnr = 10 * np.log10((255 ** 2) / mse)
    return psnr


if __name__ == '__main__':
    root = ".."

    # 0. 超参数设置
    alpha = 100  # 尺度控制因子，控制水印添加强度，决定频域系数被修改的幅度
    blocksize = 8  # 分块大小

    # 1. 数据读取

    # watermak
    watermark = cv2.imread(r"./watermark_resize.png".format(root), cv2.IMREAD_GRAYSCALE)
    watermark = np.where(watermark < np.mean(watermark), 0, 1)  # watermark进行(归一化的)二值化
    background = cv2.imread(r"./Lena_256.bmp".format(root), cv2.IMREAD_GRAYSCALE)

    # 2. 初始化DCT算法
    dct_emb = DCT_Embed(background=background, watermark=watermark, block_size=blocksize, alpha=alpha)

    # 3. 进行分块与DCT变换
    background_dct_blocks = dct_emb.dct_blkproc(background=background)  # 得到分块的DCTblocks

    # 4. 嵌入水印图像
    embed_watermark_blocks = dct_emb.dct_embed(dct_data=background_dct_blocks, watermark=watermark)  # 在dct块中嵌入水印图像

    # 5. 将图像转换为空域形式
    synthesis = dct_emb.idct_embed(dct_data=embed_watermark_blocks)  # idct变换得到空域图像
    print(calculate_psnr(background, synthesis))

    # 6. 提取水印
    extract_watermark = dct_emb.dct_extract(synthesis=synthesis, watermark_size=watermark.shape) * 255
    extract_watermark.astype(np.uint8)
    # 7. 可视化处理
    images = [background, watermark, synthesis, extract_watermark]
    titles = ["background", "watermark", "systhesis", "extract"]
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        if i % 2:
            plt.imshow(images[i], cmap=plt.cm.gray)
        else:
            plt.imshow(images[i])
        plt.title(titles[i])
        plt.axis("off")
    plt.show()
