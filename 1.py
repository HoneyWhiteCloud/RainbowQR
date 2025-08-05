from PIL import Image, ImageDraw
import random
import numpy as np

def create_color_qr(size=480, info_size=4):  # 添加info_size参数，默认为4
    """
    创建基于16级色阶的彩色编码图像
    
    参数：
    size: 图像的总像素大小
    info_size: 中心信息正方形的边长大小(2-10之间的整数)
    """
    # 验证info_size参数
    if not 2 <= info_size <= 10:
        raise ValueError("信息正方形大小必须在2到10之间")
    
    # 创建白色背景画布
    img = Image.new('RGB', (size, size), color='white')
    draw = ImageDraw.Draw(img)
    
    # 基本单位大小
    unit = size // 12  # 12x12的网格
    
    # 16级色阶 (0-255范围内平均分布)
    color_levels = np.linspace(0, 255, 16, dtype=int)
    
    # ==== 中心信息区域 ====
    # 计算信息区域的起始位置，确保居中
    info_start_x = (12 - info_size) // 2 * unit
    info_start_y = (12 - info_size) // 2 * unit
    
    # 存储编码数据
    encoded_data = []
    
    # 绘制信息区域 (随机颜色)
    for row in range(info_size):
        for col in range(info_size):
            x = info_start_x + col * unit
            y = info_start_y + row * unit
            
            # 随机选择16级色阶的RGB值
            r = int(random.choice(color_levels))
            g = int(random.choice(color_levels))
            b = int(random.choice(color_levels))
            
            encoded_data.append((r, g, b))
            draw.rectangle([x, y, x + unit, y + unit], fill=(r, g, b))
    
    # ==== 校准标记 ====
    # 第一象限 (右上) - 红色校准正方形
    x1, y1 = 10 * unit, 0
    
    # 红色校准正方形 - 按逆时针顺序排列饱和度递减
    draw.rectangle([x1, y1 + unit, x1 + unit, y1 + 2*unit], fill=(255, 255, 255))  # 白色（靠近中心）
    draw.rectangle([x1, y1, x1 + unit, y1 + unit], fill=(255, 0, 0))  # 高饱和度
    draw.rectangle([x1 + unit, y1, x1 + 2*unit, y1 + unit], fill=(128, 0, 0))  # 中饱和度
    draw.rectangle([x1 + unit, y1 + unit, x1 + 2*unit, y1 + 2*unit], fill=(64, 0, 0))  # 低饱和度
    
    # 第二象限 (左上) - 绿色校准正方形
    x2, y2 = 0, 0
    
    # 绿色校准正方形 - 按逆时针顺序排列饱和度递减
    draw.rectangle([x2 + unit, y2 + unit, x2 + 2*unit, y2 + 2*unit], fill=(255, 255, 255))  # 白色（靠近中心）
    draw.rectangle([x2, y2 + unit, x2 + unit, y2 + 2*unit], fill=(0, 64, 0))  # 低饱和度
    draw.rectangle([x2, y2, x2 + unit, y2 + unit], fill=(0, 128, 0))  # 中饱和度
    draw.rectangle([x2 + unit, y2, x2 + 2*unit, y2 + unit], fill=(0, 255, 0))  # 高饱和度
    
    # 第三象限 (左下) - 蓝色校准正方形
    x3, y3 = 0, 10 * unit
    
    # 蓝色校准正方形 - 按逆时针顺序排列饱和度递减
    draw.rectangle([x3 + unit, y3, x3 + 2*unit, y3 + unit], fill=(255, 255, 255))  # 白色（靠近中心）
    draw.rectangle([x3 + unit, y3 + unit, x3 + 2*unit, y3 + 2*unit], fill=(0, 0, 64))  # 低饱和度
    draw.rectangle([x3, y3 + unit, x3 + unit, y3 + 2*unit], fill=(0, 0, 128))  # 中饱和度
    draw.rectangle([x3, y3, x3 + unit, y3 + unit], fill=(0, 0, 255))  # 高饱和度
    
    # 第四象限 (右下) - 黑色校准正方形
    x4, y4 = 10 * unit, 10 * unit
    
    # 黑色校准正方形 - 仅用于方向标识，全黑
    draw.rectangle([x4, y4, x4 + 2*unit, y4 + 2*unit], fill=(0, 0, 0))  # 全黑
    draw.rectangle([x4, y4, x4 + unit, y4 + unit], fill=(255, 255, 255))  # 白色（靠近中心）
    
    return img, encoded_data

# 创建示例 - 可以指定不同的info_size
#img, data = create_color_qr(info_size=4)  # 默认4x4信息区域
#img.show()  # 直接显示图像，不保存
#print("彩色编码图像已创建完成并显示")

# 您可以尝试不同大小的信息区域
img2, data2 = create_color_qr(info_size=6)  # 6x6信息区域
img2.show()
# 
# img3, data3 = create_color_qr(info_size=2)  # 2x2信息区域
# img3.show()