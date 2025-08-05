# 自动检测并安装缺失的库
import subprocess
import sys
import importlib.util

def check_and_install(package, import_name=None):
    if import_name is None:
        import_name = package
    
    if importlib.util.find_spec(import_name) is None:
        print(f"检测到缺少 {package} 库，正在安装...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"{package} 安装成功!")
        except Exception as e:
            print(f"安装 {package} 失败: {e}")
            print(f"请手动运行命令安装: pip install {package}")
            if package == "opencv-python":
                print("或尝试: pip install opencv-python-headless")
            sys.exit(1)

# 检查并安装必要的库
check_and_install("opencv-python", "cv2")
check_and_install("numpy")
check_and_install("pillow", "PIL")

# 导入所需的库
from PIL import Image, ImageDraw, ImageTk
import random
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import zlib  # 用于CRC32校验
import cv2   # 用于图像处理
import math  # 用于数学计算

# 修改：减少色阶数量，从16级减少到8级，增加间距
def get_color_levels(levels=8):
    """生成色阶值，以便更好地区分"""
    # 使用更宽的步进 - 8级色阶，确保值不接近0和255
    # 这样做可以避免与黑色和白色混淆
    return np.linspace(32, 223, levels, dtype=int)

def text_to_colors(text, size, color_levels):
    """将文本转换为RGB颜色值数组 - 针对8级色阶优化"""
    # 将文本转换为UTF-8字节数组
    text_bytes = text.encode('utf-8')
    
    # 计算每个像素可存储的字节数
    # 每个通道3位(8级)，3个通道共9位，可以存储约1.125字节
    bytes_per_pixel = 1.125
    
    # 计算可存储的最大字节数
    max_bytes = int(size * size * bytes_per_pixel)
    
    # 如果文本太长，截断它
    if len(text_bytes) > max_bytes:
        text_bytes = text_bytes[:max_bytes]
    
    # 将字节数组填充到3的倍数，便于处理
    while len(text_bytes) % 3 != 0:
        text_bytes += b'\x00'
    
    # 创建颜色列表
    colors = []
    
    # 每3个字节编码为8个RGB像素
    for i in range(0, len(text_bytes), 3):
        if i + 2 < len(text_bytes):
            # 提取三个字节
            b1 = text_bytes[i]
            b2 = text_bytes[i+1]
            b3 = text_bytes[i+2]
            
            # 8级色阶每个通道用3位表示 (值0-7)
            # 因此三个字节共24位可以编码成8个值 (每个值3位)
            
            # 第一个像素
            r1 = (b1 >> 5) & 0x07  # b1的高3位
            g1 = (b1 >> 2) & 0x07  # b1的中3位
            b1_val = ((b1 & 0x03) << 1) | ((b2 >> 7) & 0x01)  # b1的低2位 + b2的高1位
            
            # 第二个像素
            r2 = (b2 >> 4) & 0x07  # b2的高3位(除了最高位)
            g2 = (b2 >> 1) & 0x07  # b2的中3位
            b2_val = ((b2 & 0x01) << 2) | ((b3 >> 6) & 0x03)  # b2的低1位 + b3的高2位
            
            # 第三个像素 (只用6位)
            r3 = (b3 >> 3) & 0x07  # b3的中高3位
            g3 = b3 & 0x07        # b3的低3位
            b3_val = 0            # 填充位
            
            # 映射到色阶值
            colors.append((
                color_levels[r1],
                color_levels[g1],
                color_levels[b1_val]
            ))
            
            colors.append((
                color_levels[r2],
                color_levels[g2],
                color_levels[b2_val]
            ))
            
            colors.append((
                color_levels[r3],
                color_levels[g3],
                color_levels[b3_val]
            ))
    
    # 填充剩余像素
    while len(colors) < size * size:
        colors.append((255, 255, 255))  # 白色填充
    
    return colors[:size*size]  # 确保不超过信息区域大小

def colors_to_text(colors, color_levels, debug=False):
    """从RGB颜色值数组恢复文本 - 针对8级色阶优化"""
    if debug:
        print(f"开始解码，颜色数量: {len(colors)}")
        print(f"色阶级别: {color_levels}")
    
    # 提取量化后的颜色索引
    indices = []
    for i, (r, g, b) in enumerate(colors):
        # 查找最接近的色阶值
        r_idx = min(range(len(color_levels)), key=lambda x: abs(color_levels[x] - r))
        g_idx = min(range(len(color_levels)), key=lambda x: abs(color_levels[x] - g))
        b_idx = min(range(len(color_levels)), key=lambda x: abs(color_levels[x] - b))
        
        indices.append((r_idx, g_idx, b_idx))
        
        if debug and i < 10:  # 仅打印前10个颜色的详细信息
            print(f"颜色 {i}: RGB({r},{g},{b}) -> 索引({r_idx},{g_idx},{b_idx})")
    
    # 转换回字节
    result_bytes = bytearray()
    
    # 每三个像素解码为3个字节
    for i in range(0, len(indices), 3):
        if i + 2 < len(indices):
            # 提取三个像素的索引
            r1, g1, b1 = indices[i]
            r2, g2, b2 = indices[i+1]
            r3, g3, b3 = indices[i+2]  # b3不使用
            
            # 重构三个字节 (每个通道3位，共9位/像素)
            byte1 = (r1 << 5) | (g1 << 2) | ((b1 & 0x06) >> 1)
            byte2 = ((b1 & 0x01) << 7) | (r2 << 4) | (g2 << 1) | ((b2 & 0x04) >> 2)
            byte3 = ((b2 & 0x03) << 6) | (r3 << 3) | g3
            
            result_bytes.append(byte1)
            result_bytes.append(byte2)
            result_bytes.append(byte3)
            
            if debug and i < 9:  # 仅打印前几组的详细信息
                print(f"像素组 {i//3}: ({r1},{g1},{b1})+({r2},{g2},{b2})+({r3},{g3},{b3}) -> 字节: {byte1:02x} {byte2:02x} {byte3:02x}")
    
    # 移除尾部的零字节
    while result_bytes and result_bytes[-1] == 0:
        result_bytes.pop()
    
    if debug:
        print(f"解码后字节数: {len(result_bytes)}")
        print(f"前20个字节: {result_bytes[:20].hex()}")
    
    # 尝试解码为UTF-8文本
    try:
        decoded = result_bytes.decode('utf-8')
        if debug:
            print(f"成功解码为UTF-8: {decoded[:30]}...")
        return decoded
    except UnicodeDecodeError as e:
        if debug:
            print(f"UTF-8解码失败: {e}")
        
        # 如果解码失败，尝试找到有效的UTF-8子序列
        valid_text = ""
        for i in range(len(result_bytes), 0, -1):
            try:
                test_text = result_bytes[:i].decode('utf-8')
                valid_text = test_text
                break
            except UnicodeDecodeError:
                continue
        
        if valid_text:
            if debug:
                print(f"找到部分有效UTF-8: {valid_text[:30]}...")
            return valid_text
        
        # 尝试其他编码方式
        for encoding in ['latin1', 'gbk', 'gb2312', 'gb18030', 'big5']:
            try:
                decoded = result_bytes.decode(encoding)
                if debug:
                    print(f"成功使用{encoding}解码: {decoded[:30]}...")
                return decoded
            except (UnicodeDecodeError, LookupError):
                pass
        
        # 如果所有尝试都失败，以十六进制显示
        hex_data = result_bytes.hex()
        if debug:
            print(f"所有解码尝试均失败，返回十六进制: {hex_data[:60]}...")
        return f"[无法解码为文本，十六进制数据] {hex_data}"

def create_color_qr(size=480, info_size=4, text=""):
    """创建基于8级色阶的彩色编码图像，编码指定文本"""
    # 验证info_size参数
    if not 2 <= info_size <= 8 or info_size % 2 != 0:
        raise ValueError("信息正方形大小必须是2到8之间的偶数")
    
    # 创建白色背景画布
    img = Image.new('RGB', (size, size), color='white')
    draw = ImageDraw.Draw(img)
    
    # 基本单位大小
    unit = size // 12  # 12x12的网格
    
    # 8级色阶 (使用更宽的步进)
    color_levels = get_color_levels(8)
    
    # ==== 中心信息区域 ====
    # 计算信息区域的起始位置，确保居中
    info_start_x = (12 - info_size) // 2 * unit
    info_start_y = (12 - info_size) // 2 * unit
    
    # 将文本转换为颜色数据
    if text:
        encoded_data = text_to_colors(text, info_size, color_levels)
    else:
        # 如果没有文本，生成随机颜色
        encoded_data = []
        for _ in range(info_size * info_size):
            r = int(random.choice(color_levels))
            g = int(random.choice(color_levels))
            b = int(random.choice(color_levels))
            encoded_data.append((r, g, b))
    
    # 绘制信息区域
    pixel_index = 0
    for row in range(info_size):
        for col in range(info_size):
            x = info_start_x + col * unit
            y = info_start_y + row * unit
            
            if pixel_index < len(encoded_data):
                color = encoded_data[pixel_index]
                draw.rectangle([x, y, x + unit, y + unit], fill=color)
                pixel_index += 1
    
    # ==== 校准标记 ====
    # 第一象限 (右上) - 红色校准正方形
    x1, y1 = 10 * unit, 0
    
    # 红色校准正方形 - 使用更鲜明的色彩
    draw.rectangle([x1 + unit, y1 + unit, x1 + 2*unit, y1 + 2*unit], fill=(255, 0, 0))  # 高饱和度（右下）
    draw.rectangle([x1 + unit, y1, x1 + 2*unit, y1 + unit], fill=(128, 0, 0))  # 中饱和度（右上）
    draw.rectangle([x1, y1, x1 + unit, y1 + unit], fill=(64, 0, 0))  # 低饱和度（左上）
    
    # 第二象限 (左上) - 绿色校准正方形
    x2, y2 = 0, 0
    
    # 绿色校准正方形 - 使用更鲜明的色彩
    draw.rectangle([x2, y2 + unit, x2 + unit, y2 + 2*unit], fill=(0, 64, 0))  # 低饱和度（左下）
    draw.rectangle([x2, y2, x2 + unit, y2 + unit], fill=(0, 128, 0))  # 中饱和度（左上）
    draw.rectangle([x2 + unit, y2, x2 + 2*unit, y2 + unit], fill=(0, 255, 0))  # 高饱和度（右上）
    
    # 第三象限 (左下) - 蓝色校准正方形
    x3, y3 = 0, 10 * unit
    
    # 蓝色校准正方形 - 使用更鲜明的色彩
    draw.rectangle([x3 + unit, y3 + unit, x3 + 2*unit, y3 + 2*unit], fill=(0, 0, 64))  # 低饱和度（右下）
    draw.rectangle([x3, y3 + unit, x3 + unit, y3 + 2*unit], fill=(0, 0, 128))  # 中饱和度（左下）
    draw.rectangle([x3, y3, x3 + unit, y3 + unit], fill=(0, 0, 255))  # 高饱和度（左上）
    
    # 第四象限 (右下) - 黑色校准正方形
    x4, y4 = 10 * unit, 10 * unit
    
    # 黑色校准正方形 - 仅用于方向标识
    draw.rectangle([x4 + unit, y4, x4 + 2*unit, y4 + unit], fill=(0, 0, 0))  # 黑色（右上）
    draw.rectangle([x4 + unit, y4 + unit, x4 + 2*unit, y4 + 2*unit], fill=(0, 0, 0))  # 黑色（右下）
    draw.rectangle([x4, y4 + unit, x4 + unit, y4 + 2*unit], fill=(0, 0, 0))  # 黑色（左下）
    
    # ==== 添加纠错码 ====
    # 生成纠错数据
    error_correction_data = generate_error_correction(encoded_data)
    ec_index = 0
    
    # 在外层两圈的安全位置放置纠错码，避开与校准正方形相邻的位置
    
    # 外层第一圈 (最外层)
    # 顶部边缘 (从左至右，避开校准正方形)
    for col in range(3, 9):  # 避开左上角和右上角校准正方形附近
        x = col * unit
        y = 0
        if ec_index < len(error_correction_data):
            draw.rectangle([x, y, x + unit, y + unit], fill=error_correction_data[ec_index])
            ec_index += 1
    
    # 右侧边缘 (从上至下，避开校准正方形)
    for row in range(3, 9):  # 避开右上角和右下角校准正方形附近
        x = 11 * unit
        y = row * unit
        if ec_index < len(error_correction_data):
            draw.rectangle([x, y, x + unit, y + unit], fill=error_correction_data[ec_index])
            ec_index += 1
    
    # 底部边缘 (从左至右，避开校准正方形)
    for col in range(3, 9):  # 避开左下角和右下角校准正方形附近
        x = col * unit
        y = 11 * unit
        if ec_index < len(error_correction_data):
            draw.rectangle([x, y, x + unit, y + unit], fill=error_correction_data[ec_index])
            ec_index += 1
    
    # 左侧边缘 (从上至下，避开校准正方形)
    for row in range(3, 9):  # 避开左上角和左下角校准正方形附近
        x = 0
        y = row * unit
        if ec_index < len(error_correction_data):
            draw.rectangle([x, y, x + unit, y + unit], fill=error_correction_data[ec_index])
            ec_index += 1
    
    # 外层第二圈
    # 顶部第二行 (从左至右，避开校准正方形)
    for col in range(3, 9):
        x = col * unit
        y = unit
        if ec_index < len(error_correction_data):
            draw.rectangle([x, y, x + unit, y + unit], fill=error_correction_data[ec_index])
            ec_index += 1
    
    # 右侧第二列 (从上至下，避开校准正方形)
    for row in range(3, 9):
        x = 10 * unit
        y = row * unit
        if ec_index < len(error_correction_data):
            draw.rectangle([x, y, x + unit, y + unit], fill=error_correction_data[ec_index])
            ec_index += 1
    
    # 底部第二行 (从左至右，避开校准正方形)
    for col in range(3, 9):
        x = col * unit
        y = 10 * unit
        if ec_index < len(error_correction_data):
            draw.rectangle([x, y, x + unit, y + unit], fill=error_correction_data[ec_index])
            ec_index += 1
    
    # 左侧第二列 (从上至下，避开校准正方形)
    for row in range(3, 9):
        x = unit
        y = row * unit
        if ec_index < len(error_correction_data):
            draw.rectangle([x, y, x + unit, y + unit], fill=error_correction_data[ec_index])
            ec_index += 1
    
    return img, encoded_data

def generate_error_correction(data):
    """
    生成简单的纠错码，包括：
    1. 全局CRC32校验和
    2. 每行和每列的奇偶校验
    3. Reed-Solomon样式的冗余数据
    """
    error_correction = []
    
    # 1. 计算全局CRC32校验和
    data_bytes = b''
    for r, g, b in data:
        data_bytes += bytes([r, g, b])
    
    checksum = zlib.crc32(data_bytes)
    
    # 将32位校验和拆分为4个字节
    for i in range(4):
        val = (checksum >> (i * 8)) & 0xFF
        # 将每个字节映射到RGB颜色
        r = val & 0xF0  # 高4位
        g = (val & 0x0F) << 4  # 低4位
        b = 128  # 固定值，便于识别
        error_correction.append((r, g, b))
    
    # 2. 计算行校验和
    size = int(len(data)**0.5)  # 信息正方形的边长
    
    # 为每行计算校验和
    for i in range(size):
        row = data[i*size:(i+1)*size]
        # 计算RGB通道的校验和
        r_sum = sum(pixel[0] for pixel in row) % 256
        g_sum = sum(pixel[1] for pixel in row) % 256
        b_sum = sum(pixel[2] for pixel in row) % 256
        
        # 校验和转为RGB颜色
        error_correction.append((r_sum, g_sum, b_sum))
    
    # 3. 计算列校验和
    for i in range(size):
        col = [data[i + j*size] for j in range(size)]
        # 计算RGB通道的校验和
        r_sum = sum(pixel[0] for pixel in col) % 256
        g_sum = sum(pixel[1] for pixel in col) % 256
        b_sum = sum(pixel[2] for pixel in col) % 256
        
        # 校验和转为RGB颜色
        error_correction.append((r_sum, g_sum, b_sum))
    
    # 4. 简化版Reed-Solomon样式冗余 - 为每个数据块添加XOR校验
    for i in range(0, len(data), 2):
        if i + 1 < len(data):
            r = data[i][0] ^ data[i+1][0]
            g = data[i][1] ^ data[i+1][1]
            b = data[i][2] ^ data[i+1][2]
            error_correction.append((r, g, b))
    
    # 添加视觉标识，区分不同类型的纠错码
    for i in range(len(error_correction)):
        r, g, b = error_correction[i]
        # 根据纠错码类型调整颜色特征
        if i < 4:  # CRC32校验
            error_correction[i] = (r, g, b | 32)  # 在蓝色通道添加标记
        elif i < 4 + size:  # 行校验
            error_correction[i] = (r, g | 32, b)  # 在绿色通道添加标记
        elif i < 4 + size*2:  # 列校验
            error_correction[i] = (r | 32, g, b)  # 在红色通道添加标记
    
    return error_correction

# 图像分析与识别 - 仅提取颜色信息而不返回处理后的图像
def analyze_image(image, info_size=4, debug=False):
    """分析外部导入的图像，仅提取色块信息"""
    # 转换为OpenCV格式进行处理
    if isinstance(image, Image.Image):
        img_cv = np.array(image)
        # 如果是RGBA图像，只保留RGB通道
        if len(img_cv.shape) > 2 and img_cv.shape[2] == 4:
            img_cv = img_cv[:, :, :3]
        # OpenCV使用BGR，而PIL使用RGB，需要转换
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    else:
        img_cv = image
    
    # 调整为标准大小
    height, width = img_cv.shape[:2]
    target_size = 480
    if height != target_size or width != target_size:
        img_cv = cv2.resize(img_cv, (target_size, target_size))
    
    # 保存原始图像供调试
    original_img = img_cv.copy()
    
    # 首先尝试基于颜色的标记检测
    print("尝试基于颜色的标记检测...")
    markers = locate_calibration_markers_by_color(img_cv, debug)
    
    # 如果颜色检测失败，尝试基于区域的标记检测
    if not markers or len(markers) < 4:
        print(f"颜色检测只找到 {len(markers) if markers else 0} 个标记，尝试基于区域的检测...")
        markers = locate_calibration_markers_by_region(img_cv, debug)
    
    # 如果仍然失败，尝试简化版的信息提取
    if not markers or len(markers) < 4:
        print(f"区域检测只找到 {len(markers) if markers else 0} 个标记，尝试简化信息提取...")
        # 使用直接中心提取方法
        colors = extract_center_colors(img_cv, info_size)
        return colors
    
    print("找到全部4个校准标记，执行透视校正...")
    # 根据校准标记进行透视校正
    corrected_img = perspective_correction(img_cv, markers)
    
    if debug:
        # 显示校正后的图像
        cv2.imshow("校正后的图像", corrected_img)
        cv2.waitKey(1000)
    
    # 定位信息区域
    info_area = determine_info_area(corrected_img, info_size)
    
    # 提取信息区域的颜色数据
    colors = extract_colors(corrected_img, info_area, info_size)
    
    return colors

def locate_calibration_markers_by_color(img, debug=False):
    """基于颜色特征定位四个角上的校准标记"""
    height, width = img.shape[:2]
    
    # 转换为HSV颜色空间，更容易分离颜色
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 使用更宽松的颜色范围
    # 绿色
    lower_green = np.array([40, 20, 20])
    upper_green = np.array([90, 255, 255])
    
    # 红色 (需要两个范围，因为红色在HSV中横跨两端)
    lower_red1 = np.array([0, 20, 20])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 20, 20])
    upper_red2 = np.array([180, 255, 255])
    
    # 蓝色
    lower_blue = np.array([90, 20, 20])
    upper_blue = np.array([140, 255, 255])
    
    # 黑色
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 50])
    
    # 创建颜色掩码
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_black = cv2.inRange(hsv, lower_black, upper_black)
    
    if debug:
        # 显示掩码图像
        cv2.imshow("绿色掩码", mask_green)
        cv2.imshow("红色掩码", mask_red)
        cv2.imshow("蓝色掩码", mask_blue)
        cv2.imshow("黑色掩码", mask_black)
        cv2.waitKey(1000)
    
    # 定义角落区域 (使用更大的搜索区域)
    regions = [
        ("green", mask_green, (0, 0, width//3, height//3)),                     # 左上 - 绿色
        ("red", mask_red, (width*2//3, 0, width, height//3)),                   # 右上 - 红色
        ("blue", mask_blue, (0, height*2//3, width, height)),                   # 左下 - 蓝色
        ("black", mask_black, (width*2//3, height*2//3, width, height))         # 右下 - 黑色
    ]
    
    markers = []
    debug_img = img.copy() if debug else None
    
    # 在每个区域内查找对应颜色的轮廓
    for color, mask, (x1, y1, x2, y2) in regions:
        region_mask = np.zeros_like(mask)
        region_mask[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
        
        # 腐蚀和膨胀操作，去除噪声
        kernel = np.ones((3,3), np.uint8)
        region_mask = cv2.morphologyEx(region_mask, cv2.MORPH_OPEN, kernel)
        region_mask = cv2.morphologyEx(region_mask, cv2.MORPH_CLOSE, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # 筛选面积合适的轮廓
            valid_contours = [c for c in contours if cv2.contourArea(c) > 50 and cv2.contourArea(c) < 5000]
            
            if valid_contours:
                # 找到最大的轮廓
                c = max(valid_contours, key=cv2.contourArea)
                
                # 计算轮廓中心
                M = cv2.moments(c)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    if debug:
                        # 在调试图像上绘制标记
                        cv2.drawContours(debug_img, [c], 0, (0, 255, 255), 2)
                        cv2.circle(debug_img, (cx, cy), 5, (255, 0, 255), -1)
                        cv2.putText(debug_img, color, (cx-20, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    markers.append((color, (cx, cy)))
    
    if debug and debug_img is not None:
        # 显示带有标记的调试图像
        cv2.imshow("找到的校准标记", debug_img)
        cv2.waitKey(1000)
    
    return markers

def locate_calibration_markers_by_region(img, debug=False):
    """基于图像区域特征定位四个角上的校准标记"""
    height, width = img.shape[:2]
    unit = width // 12  # 基本单位大小
    
    # 我们知道校准标记的大致位置
    corners = [
        ("green", (0, 0, 2*unit, 2*unit)),               # 左上 - 绿色
        ("red", (width-2*unit, 0, width, 2*unit)),       # 右上 - 红色
        ("blue", (0, height-2*unit, 2*unit, height)),    # 左下 - 蓝色
        ("black", (width-2*unit, height-2*unit, width, height))  # 右下 - 黑色
    ]
    
    markers = []
    debug_img = img.copy() if debug else None
    
    for color, (x1, y1, x2, y2) in corners:
        # 提取角落区域
        region = img[y1:y2, x1:x2]
        
        # 转换为灰度图
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        
        # 二值化，创建掩码
        _, mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        # 使用形态学操作清理图像
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # 找到最大的轮廓
            c = max(contours, key=cv2.contourArea)
            
            # 计算轮廓中心
            M = cv2.moments(c)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"]) + x1
                cy = int(M["m01"] / M["m00"]) + y1
                
                if debug and debug_img is not None:
                    # 在调试图像上绘制标记
                    cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.circle(debug_img, (cx, cy), 5, (255, 0, 255), -1)
                    cv2.putText(debug_img, color, (cx-20, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                markers.append((color, (cx, cy)))
    
    if debug and debug_img is not None:
        # 显示带有标记的调试图像
        cv2.imshow("基于区域的校准标记", debug_img)
        cv2.waitKey(1000)
    
    return markers

def extract_center_colors(img, info_size):
    """当无法找到校准标记时，直接提取中心区域的信息"""
    print("使用简化方法：直接提取中心区域")
    height, width = img.shape[:2]
    
    # 计算中心区域
    unit = width // 12
    start_x = (12 - info_size) // 2 * unit
    start_y = (12 - info_size) // 2 * unit
    end_x = start_x + info_size * unit
    end_y = start_y + info_size * unit
    
    # 截取中心区域
    center_img = img[start_y:end_y, start_x:end_x]
    
    # 调整为正确大小
    center_img = cv2.resize(center_img, (info_size * unit, info_size * unit))
    
    # 提取颜色
    colors = []
    cell_size = unit
    
    for row in range(info_size):
        for col in range(info_size):
            y1 = row * cell_size
            x1 = col * cell_size
            y2 = (row + 1) * cell_size
            x2 = (col + 1) * cell_size
            
            cell = center_img[y1:y2, x1:x2]
            avg_color = np.mean(cell, axis=(0, 1))
            
            # BGR to RGB
            colors.append((int(avg_color[2]), int(avg_color[1]), int(avg_color[0])))
    
    return colors

def perspective_correction(img, markers):
    """根据校准标记进行透视校正"""
    height, width = img.shape[:2]
    
    # 将标记按颜色分类
    marker_dict = dict(markers)
    
    # 检查是否有所有需要的标记
    required_colors = ["green", "red", "blue", "black"]
    if not all(color in marker_dict for color in required_colors):
        print("缺少必要的标记点，使用原始图像")
        return img
    
    # 获取四个角点的坐标
    top_left = marker_dict["green"]      # 左上 - 绿色
    top_right = marker_dict["red"]       # 右上 - 红色
    bottom_left = marker_dict["blue"]    # 左下 - 蓝色
    bottom_right = marker_dict["black"]  # 右下 - 黑色
    
    # 源坐标点
    pts1 = np.float32([top_left, top_right, bottom_left, bottom_right])
    
    # 目标坐标点 (标准正方形)
    pts2 = np.float32([
        [0, 0],
        [width-1, 0],
        [0, height-1],
        [width-1, height-1]
    ])
    
    # 计算透视变换矩阵
    M = cv2.getPerspectiveTransform(pts1, pts2)
    
    # 进行透视变换
    dst = cv2.warpPerspective(img, M, (width, height))
    
    return dst

def determine_info_area(img, info_size):
    """确定信息区域的位置"""
    height, width = img.shape[:2]
    unit = width // 12  # 基本单位大小
    
    # 计算信息区域的起始和结束位置
    start_x = (12 - info_size) // 2 * unit
    start_y = (12 - info_size) // 2 * unit
    end_x = start_x + info_size * unit
    end_y = start_y + info_size * unit
    
    return (start_x, start_y, end_x, end_y)

def extract_colors(img, info_area, info_size):
    """从信息区域提取颜色数据"""
    start_x, start_y, end_x, end_y = info_area
    
    # 计算单元格大小
    cell_width = (end_x - start_x) / info_size
    cell_height = (end_y - start_y) / info_size
    
    # 提取每个单元格的颜色
    colors = []
    for row in range(info_size):
        for col in range(info_size):
            # 计算单元格位置
            x1 = int(start_x + col * cell_width)
            y1 = int(start_y + row * cell_height)
            x2 = int(start_x + (col + 1) * cell_width)
            y2 = int(start_y + (row + 1) * cell_height)
            
            # 确保边界不越界
            x1 = max(0, min(x1, img.shape[1]-1))
            y1 = max(0, min(y1, img.shape[0]-1))
            x2 = max(0, min(x2, img.shape[1]-1))
            y2 = max(0, min(y2, img.shape[0]-1))
            
            # 获取单元格区域
            cell = img[y1:y2, x1:x2]
            
            if cell.size > 0:
                # 计算平均颜色
                avg_color = np.mean(cell, axis=(0, 1))
                # BGR to RGB
                colors.append((int(avg_color[2]), int(avg_color[1]), int(avg_color[0])))
            else:
                # 如果单元格为空，使用白色
                colors.append((255, 255, 255))
    
    return colors

# 新增函数：使用提取的颜色数据重新生成标准图像
def recreate_image_from_colors(colors, size=480, info_size=4):
    """使用提取的颜色数据重新生成标准图像"""
    # 创建白色背景画布
    img = Image.new('RGB', (size, size), color='white')
    draw = ImageDraw.Draw(img)
    
    # 基本单位大小
    unit = size // 12  # 12x12的网格
    
    # 计算信息区域的起始位置
    info_start_x = (12 - info_size) // 2 * unit
    info_start_y = (12 - info_size) // 2 * unit
    
    # 绘制信息区域
    pixel_index = 0
    for row in range(info_size):
        for col in range(info_size):
            x = info_start_x + col * unit
            y = info_start_y + row * unit
            
            if pixel_index < len(colors):
                color = colors[pixel_index]
                draw.rectangle([x, y, x + unit, y + unit], fill=color)
                pixel_index += 1
    
    # ==== 校准标记 ====
    # 第一象限 (右上) - 红色校准正方形
    x1, y1 = 10 * unit, 0
    
    # 红色校准正方形 - 使用更鲜明的色彩
    draw.rectangle([x1 + unit, y1 + unit, x1 + 2*unit, y1 + 2*unit], fill=(255, 0, 0))  # 高饱和度（右下）
    draw.rectangle([x1 + unit, y1, x1 + 2*unit, y1 + unit], fill=(128, 0, 0))  # 中饱和度（右上）
    draw.rectangle([x1, y1, x1 + unit, y1 + unit], fill=(64, 0, 0))  # 低饱和度（左上）
    
    # 第二象限 (左上) - 绿色校准正方形
    x2, y2 = 0, 0
    
    # 绿色校准正方形 - 使用更鲜明的色彩
    draw.rectangle([x2, y2 + unit, x2 + unit, y2 + 2*unit], fill=(0, 64, 0))  # 低饱和度（左下）
    draw.rectangle([x2, y2, x2 + unit, y2 + unit], fill=(0, 128, 0))  # 中饱和度（左上）
    draw.rectangle([x2 + unit, y2, x2 + 2*unit, y2 + unit], fill=(0, 255, 0))  # 高饱和度（右上）
    
    # 第三象限 (左下) - 蓝色校准正方形
    x3, y3 = 0, 10 * unit
    
    # 蓝色校准正方形 - 使用更鲜明的色彩
    draw.rectangle([x3 + unit, y3 + unit, x3 + 2*unit, y3 + 2*unit], fill=(0, 0, 64))  # 低饱和度（右下）
    draw.rectangle([x3, y3 + unit, x3 + unit, y3 + 2*unit], fill=(0, 0, 128))  # 中饱和度（左下）
    draw.rectangle([x3, y3, x3 + unit, y3 + unit], fill=(0, 0, 255))  # 高饱和度（左上）
    
    # 第四象限 (右下) - 黑色校准正方形
    x4, y4 = 10 * unit, 10 * unit
    
    # 黑色校准正方形 - 仅用于方向标识
    draw.rectangle([x4 + unit, y4, x4 + 2*unit, y4 + unit], fill=(0, 0, 0))  # 黑色（右上）
    draw.rectangle([x4 + unit, y4 + unit, x4 + 2*unit, y4 + 2*unit], fill=(0, 0, 0))  # 黑色（右下）
    draw.rectangle([x4, y4 + unit, x4 + unit, y4 + 2*unit], fill=(0, 0, 0))  # 黑色（左下）
    
    # 其余区域可以选择是否绘制纠错信息，这里简化处理，不绘制纠错码
    
    return img

class ColorQRApp:
    def __init__(self, master):
        self.master = master
        self.master.title("彩色编码图像生成器")
        self.master.geometry("600x850")  # 增大窗口高度以容纳文本输入区
        
        # 设置图像尺寸
        self.img_size = 480
        self.current_info_size = 4
        
        # 创建界面元素
        self.create_widgets()
        
        # 初始生成图像
        self.update_image()

    def create_widgets(self):
        # 顶部控制区域
        control_frame = ttk.Frame(self.master, padding="10 10 10 10")
        control_frame.pack(fill=tk.X)
        
        # 信息正方形大小选择
        ttk.Label(control_frame, text="信息正方形大小:").grid(column=0, row=0, padx=5, pady=5)
        
        # 使用Combobox替代滑动条，确保只能选择偶数大小
        self.size_var = tk.StringVar(value="4")
        self.size_combo = ttk.Combobox(
            control_frame,
            textvariable=self.size_var,
            values=["2", "4", "6", "8"],
            width=5,
            state="readonly"
        )
        self.size_combo.grid(column=1, row=0, padx=5, pady=5)
        self.size_combo.bind("<<ComboboxSelected>>", self.on_size_change)
        
        ttk.Label(control_frame, text="x").grid(column=2, row=0)
        ttk.Label(control_frame, text=self.size_var.get()).grid(column=3, row=0, padx=5, pady=5)
        
        # 添加调试模式开关
        self.debug_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            control_frame, 
            text="调试模式", 
            variable=self.debug_var
        ).grid(column=4, row=0, padx=5, pady=5)
        
        # 文本输入区域
        text_frame = ttk.LabelFrame(self.master, text="输入要编码的文本", padding="10 5 10 10")
        text_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.text_input = scrolledtext.ScrolledText(text_frame, height=4, width=50, wrap=tk.WORD)
        self.text_input.pack(fill=tk.X, expand=True)
        
        # 字符计数标签
        self.char_count_label = ttk.Label(text_frame, text="0/0 字符")
        self.char_count_label.pack(anchor=tk.E, pady=2)
        
        # 绑定文本变化事件
        self.text_input.bind("<KeyRelease>", self.update_char_count)
        
        # 按钮区域
        button_frame = ttk.Frame(self.master, padding="10 5 10 5")
        button_frame.pack(fill=tk.X)
        
        # 编码按钮
        ttk.Button(
            button_frame, 
            text="编码文本到图像", 
            command=self.encode_text
        ).pack(side=tk.LEFT, padx=5, pady=5)
        
        # 解码按钮
        ttk.Button(
            button_frame, 
            text="从图像解码文本", 
            command=self.decode_image
        ).pack(side=tk.LEFT, padx=5, pady=5)
        
        # 导入图片按钮 (新增)
        ttk.Button(
            button_frame, 
            text="导入图片", 
            command=self.import_image
        ).pack(side=tk.LEFT, padx=5, pady=5)
        
        # 随机生成新图像按钮
        ttk.Button(
            button_frame, 
            text="生成随机图像", 
            command=self.update_image
        ).pack(side=tk.LEFT, padx=5, pady=5)
        
        # 保存图像按钮
        ttk.Button(
            button_frame, 
            text="保存图像", 
            command=self.save_image
        ).pack(side=tk.LEFT, padx=5, pady=5)
        
        # 图像显示区域
        self.image_frame = ttk.Frame(self.master, padding="10 10 10 10")
        self.image_frame.pack(fill=tk.BOTH, expand=True)
        self.image_label = ttk.Label(self.image_frame)
        self.image_label.pack(fill=tk.BOTH, expand=True)
        
        # 添加信息标签，显示编码数据容量
        info_frame = ttk.Frame(self.master, padding="10 5 10 10")
        info_frame.pack(fill=tk.X)
        
        # 信息容量标签
        self.info_label = ttk.Label(info_frame, text="")
        self.info_label.pack(pady=2)
        
        # 编码方案标签
        self.encoding_label = ttk.Label(
            info_frame, 
            text="编码方案：8级色阶 (每通道3位) 提高色彩间距，增强识别准确性",
            wraplength=550
        )
        self.encoding_label.pack(pady=2)
        
        # 纠错信息标签
        self.error_correction_label = ttk.Label(
            info_frame, 
            text="纠错机制：CRC32全局校验 + 行列校验和 + 数据块XOR校验",
            wraplength=550
        )
        self.error_correction_label.pack(pady=2)
        
        # 更新信息标签
        self.update_info_label()
        
        # 在文本框中添加默认文本示例
        self.text_input.insert("1.0", "你好世界我是白云")

    def on_size_change(self, event):
        # 获取选择的值并更新图像
        self.current_info_size = int(self.size_var.get())
        self.update_image()
        self.update_info_label()
        self.update_char_count(None)  # 更新字符计数

    def update_image(self):
        # 生成新的彩色编码图像 (随机数据)
        self.img, self.data = create_color_qr(self.img_size, self.current_info_size)
        # 将PIL图像转换为Tkinter可显示的格式
        self.tk_img = ImageTk.PhotoImage(self.img)
        # 更新显示
        self.image_label.config(image=self.tk_img)
    
    def encode_text(self):
        # 获取输入文本
        text = self.text_input.get("1.0", tk.END).strip()
        
        if not text:
            return
        
        # 生成编码文本的彩色图像
        self.img, self.data = create_color_qr(self.img_size, self.current_info_size, text)
        # 将PIL图像转换为Tkinter可显示的格式
        self.tk_img = ImageTk.PhotoImage(self.img)
        # 更新显示
        self.image_label.config(image=self.tk_img)
    
    def decode_image(self):
        # 从当前图像解码文本
        if hasattr(self, 'data'):
            # 8级色阶
            color_levels = get_color_levels(8)
            
            # 使用改进的解码函数
            debug_mode = self.debug_var.get()
            decoded_text = colors_to_text(self.data, color_levels, debug_mode)
            
            # 显示解码结果
            if decoded_text and not decoded_text.startswith("[无法解码"):
                self.text_input.delete("1.0", tk.END)
                self.text_input.insert("1.0", decoded_text)
                if debug_mode:
                    messagebox.showinfo("解码成功", "成功解码文本！请查看控制台了解详细解码过程。")
            else:
                self.text_input.delete("1.0", tk.END)
                self.text_input.insert("1.0", decoded_text if decoded_text else "解码失败，未能提取有效文本")
                if debug_mode:
                    messagebox.showinfo("解码结果", "解码可能不完整，请查看控制台了解详细解码过程。")
            
            # 更新字符计数
            self.update_char_count(None)
    
    def import_image(self):
        """导入外部图片并分析解码"""
        # 打开文件对话框选择图片
        file_path = filedialog.askopenfilename(
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.gif"),
                ("All files", "*.*")
            ]
        )
        
        if not file_path:
            return
        
        try:
            # 加载图片
            imported_img = Image.open(file_path)
            
            # 显示状态信息
            self.master.config(cursor="wait")  # 更改鼠标光标为等待状态
            self.master.update()
            
            # 获取调试模式设置
            debug_mode = self.debug_var.get()
            
            # 分析图像，只提取颜色信息
            colors = analyze_image(imported_img, self.current_info_size, debug_mode)
            
            if not colors:
                messagebox.showerror("分析失败", "无法识别图像中的彩色编码，请确保图像清晰且包含正确的校准标记。")
                self.master.config(cursor="")  # 恢复鼠标光标
                return
            
            # 保存颜色数据
            self.data = colors
            
            # 使用提取的颜色数据重新生成标准图像
            self.img = recreate_image_from_colors(colors, self.img_size, self.current_info_size)
            
            # 更新显示
            self.tk_img = ImageTk.PhotoImage(self.img)
            self.image_label.config(image=self.tk_img)
            
            # 解码文本
            color_levels = get_color_levels(8)
            decoded_text = colors_to_text(colors, color_levels, debug_mode)
            
            # 显示解码结果
            if decoded_text and not decoded_text.startswith("[无法解码"):
                self.text_input.delete("1.0", tk.END)
                self.text_input.insert("1.0", decoded_text)
                messagebox.showinfo("解码成功", f"成功从图像中解码出文本内容。")
            else:
                self.text_input.delete("1.0", tk.END)
                self.text_input.insert("1.0", decoded_text if decoded_text else "解码失败，未能提取有效文本")
                messagebox.showinfo("解码结果", "图像分析成功，但未能解码出完整有效文本。")
            
            # 更新字符计数
            self.update_char_count(None)
            
        except Exception as e:
            messagebox.showerror("导入错误", f"处理图像时出错: {str(e)}")
        finally:
            self.master.config(cursor="")  # 恢复鼠标光标
    
    def update_info_label(self):
        # 更新信息容量标签
        info_size = self.current_info_size
        total_pixels = info_size * info_size
        total_bits = total_pixels * 9  # 每个像素9位信息 (每通道3位)
        total_bytes = total_bits // 8
        
        # 可容纳的ASCII字符数量
        ascii_chars = total_bytes
        
        # 可容纳的UTF-8中文字符数量 (假设平均每个中文字符占3字节)
        chinese_chars = total_bytes // 3
        
        self.info_label.config(
            text=f"信息容量：{total_pixels}像素 = {total_bits}位 = {total_bytes}字节 ≈ {ascii_chars}个ASCII字符 ≈ {chinese_chars}个中文字符"
        )
    
    def update_char_count(self, event):
        # 更新字符计数
        text = self.text_input.get("1.0", tk.END).strip()
        current_chars = len(text)
        
        # 计算最大字符数
        info_size = self.current_info_size
        total_pixels = info_size * info_size
        total_bytes = (total_pixels * 9) // 8  # 每个像素9位 = 1.125字节
        
        # 最大ASCII字符数
        max_ascii = total_bytes
        
        # 估计最大中文字符数 (UTF-8中大约占3字节)
        max_chinese = total_bytes // 3
        
        # 检测文本类型 (ASCII还是包含中文)
        has_chinese = any(ord(c) > 127 for c in text)
        
        if has_chinese:
            # 显示中文字符计数
            self.char_count_label.config(text=f"{current_chars}/{max_chinese} 中文字符")
            # 如果超出，使文本变红
            if current_chars > max_chinese:
                self.char_count_label.config(foreground="red")
            else:
                self.char_count_label.config(foreground="black")
        else:
            # 显示ASCII字符计数
            self.char_count_label.config(text=f"{current_chars}/{max_ascii} ASCII字符")
            # 如果超出，使文本变红
            if current_chars > max_ascii:
                self.char_count_label.config(foreground="red")
            else:
                self.char_count_label.config(foreground="black")
    
    def save_image(self):
        # 保存图像到文件
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )
        if file_path:
            self.img.save(file_path)
            print(f"图像已保存到 {file_path}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ColorQRApp(root)
    root.mainloop()