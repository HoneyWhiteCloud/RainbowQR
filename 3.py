from PIL import Image, ImageDraw, ImageTk
import random
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog
import zlib  # 用于CRC32校验

def create_color_qr(size=480, info_size=4):
    """创建基于16级色阶的彩色编码图像，带有纠错机制"""
    # 验证info_size参数
    if not 2 <= info_size <= 8 or info_size % 2 != 0:
        raise ValueError("信息正方形大小必须是2到8之间的偶数")
    
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
    
    # 红色校准正方形 - 正确排列饱和度
    draw.rectangle([x1 + unit, y1 + unit, x1 + 2*unit, y1 + 2*unit], fill=(255, 0, 0))  # 高饱和度（右下）
    draw.rectangle([x1 + unit, y1, x1 + 2*unit, y1 + unit], fill=(128, 0, 0))  # 中饱和度（右上）
    draw.rectangle([x1, y1, x1 + unit, y1 + unit], fill=(64, 0, 0))  # 低饱和度（左上）
    
    # 第二象限 (左上) - 绿色校准正方形
    x2, y2 = 0, 0
    
    # 绿色校准正方形 - 正确排列饱和度
    draw.rectangle([x2, y2 + unit, x2 + unit, y2 + 2*unit], fill=(0, 64, 0))  # 低饱和度（左下）
    draw.rectangle([x2, y2, x2 + unit, y2 + unit], fill=(0, 128, 0))  # 中饱和度（左上）
    draw.rectangle([x2 + unit, y2, x2 + 2*unit, y2 + unit], fill=(0, 255, 0))  # 高饱和度（右上）
    
    # 第三象限 (左下) - 蓝色校准正方形
    x3, y3 = 0, 10 * unit
    
    # 蓝色校准正方形 - 正确排列饱和度
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

class ColorQRApp:
    def __init__(self, master):
        self.master = master
        self.master.title("彩色编码图像生成器")
        self.master.geometry("600x700")  # 增大窗口高度以容纳更多信息
        
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
        
        # 随机生成新图像按钮
        ttk.Button(
            control_frame, 
            text="生成新图像", 
            command=self.update_image
        ).grid(column=0, row=1, padx=5, pady=5)
        
        # 保存图像按钮
        ttk.Button(
            control_frame, 
            text="保存图像", 
            command=self.save_image
        ).grid(column=1, row=1, padx=5, pady=5)
        
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
        
        # 纠错信息标签
        self.error_correction_label = ttk.Label(
            info_frame, 
            text="纠错机制：CRC32全局校验 + 行列校验和 + 数据块XOR校验",
            wraplength=550
        )
        self.error_correction_label.pack(pady=2)
        
        # 使用说明
        self.instruction_label = ttk.Label(
            info_frame,
            text="彩色编码每个像素使用16级色阶，提供高信息密度。校准正方形位于四角，纠错码位于外圈且与校准正方形保持距离，提高读取可靠性。",
            wraplength=550,
            justify="center"
        )
        self.instruction_label.pack(pady=5)
        
        self.update_info_label()

    def on_size_change(self, event):
        # 获取选择的值并更新图像
        self.current_info_size = int(self.size_var.get())
        self.update_image()
        self.update_info_label()

    def update_image(self):
        # 生成新的彩色编码图像
        self.img, self.data = create_color_qr(self.img_size, self.current_info_size)
        # 将PIL图像转换为Tkinter可显示的格式
        self.tk_img = ImageTk.PhotoImage(self.img)
        # 更新显示
        self.image_label.config(image=self.tk_img)
    
    def update_info_label(self):
        # 更新信息容量标签
        info_size = self.current_info_size
        total_pixels = info_size * info_size
        total_bits = total_pixels * 12  # 每个像素12位信息
        total_bytes = total_bits // 8
        
        # 可容纳的ASCII字符数量
        ascii_chars = total_bytes
        
        # 可容纳的UTF-8中文字符数量 (假设平均每个中文字符占3字节)
        chinese_chars = total_bytes // 3
        
        self.info_label.config(
            text=f"信息容量：{total_pixels}像素 = {total_bits}位 = {total_bytes}字节 ≈ {ascii_chars}个ASCII字符 ≈ {chinese_chars}个中文字符"
        )
        
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