from PIL import Image, ImageDraw, ImageTk
import random
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
import zlib  # 用于CRC32校验

def text_to_colors(text, size, color_levels):
    """将文本转换为RGB颜色值数组 - 修复版"""
    # 将文本转换为UTF-8字节数组
    text_bytes = text.encode('utf-8')
    
    # 计算每个像素可存储的字节数
    # 每个通道4位，3个通道共12位，可以存储1.5字节
    bytes_per_pixel = 1.5
    
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
    
    # 每3个字节编码为2个RGB像素
    for i in range(0, len(text_bytes), 3):
        if i + 2 < len(text_bytes):
            # 提取三个字节
            b1 = text_bytes[i]
            b2 = text_bytes[i+1]
            b3 = text_bytes[i+2]
            
            # 第一个像素
            r1 = (b1 >> 4) & 0x0F  # b1的高4位
            g1 = b1 & 0x0F         # b1的低4位
            b1_val = (b2 >> 4) & 0x0F  # b2的高4位
            
            # 第二个像素
            r2 = b2 & 0x0F         # b2的低4位
            g2 = (b3 >> 4) & 0x0F  # b3的高4位
            b2_val = b3 & 0x0F     # b3的低4位
            
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
    
    # 填充剩余像素
    while len(colors) < size * size:
        colors.append((255, 255, 255))  # 白色填充
    
    return colors[:size*size]  # 确保不超过信息区域大小

def colors_to_text(colors, color_levels):
    """从RGB颜色值数组恢复文本 - 修复版"""
    # 创建色阶值到索引的映射
    reverse_map = {}
    for i, level in enumerate(color_levels):
        reverse_map[level] = i
    
    # 提取颜色索引
    indices = []
    for r, g, b in colors:
        # 查找最接近的色阶值
        closest_r = min(color_levels, key=lambda x: abs(x - r))
        closest_g = min(color_levels, key=lambda x: abs(x - g))
        closest_b = min(color_levels, key=lambda x: abs(x - b))
        
        # 获取索引值
        r_idx = reverse_map[closest_r]
        g_idx = reverse_map[closest_g]
        b_idx = reverse_map[closest_b]
        
        indices.append((r_idx, g_idx, b_idx))
    
    # 转换回字节
    result_bytes = bytearray()
    
    # 每两个像素解码为3个字节
    for i in range(0, len(indices), 2):
        if i + 1 < len(indices):
            # 提取两个像素的索引
            r1, g1, b1 = indices[i]
            r2, g2, b2 = indices[i+1]
            
            # 重构三个字节
            byte1 = (r1 << 4) | g1
            byte2 = (b1 << 4) | r2
            byte3 = (g2 << 4) | b2
            
            result_bytes.append(byte1)
            result_bytes.append(byte2)
            result_bytes.append(byte3)
    
    # 移除尾部的零字节
    while result_bytes and result_bytes[-1] == 0:
        result_bytes.pop()
    
    # 尝试解码为UTF-8文本
    try:
        return result_bytes.decode('utf-8')
    except UnicodeDecodeError:
        # 如果解码失败，尝试找到有效的UTF-8子序列
        for i in range(len(result_bytes), 0, -1):
            try:
                return result_bytes[:i].decode('utf-8')
            except UnicodeDecodeError:
                continue
        # 如果所有尝试都失败，返回十六进制表示
        return result_bytes.hex()

def create_color_qr(size=480, info_size=4, text=""):
    """创建基于16级色阶的彩色编码图像，编码指定文本"""
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
        self.text_input.insert("1.0", "Hello World!")

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
            # 16级色阶
            color_levels = np.linspace(0, 255, 16, dtype=int)
            decoded_text = colors_to_text(self.data, color_levels)
            
            # 显示解码结果
            if decoded_text:
                self.text_input.delete("1.0", tk.END)
                self.text_input.insert("1.0", decoded_text)
            else:
                self.text_input.delete("1.0", tk.END)
                self.text_input.insert("1.0", "解码失败，未能提取有效文本")
            
            # 更新字符计数
            self.update_char_count(None)
    
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
    
    def update_char_count(self, event):
        # 更新字符计数
        text = self.text_input.get("1.0", tk.END).strip()
        current_chars = len(text)
        
        # 计算最大字符数
        info_size = self.current_info_size
        total_pixels = info_size * info_size
        total_bytes = (total_pixels * 12) // 8  # 每个像素12位 = 1.5字节
        
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