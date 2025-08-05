from PIL import Image, ImageDraw, ImageTk
import random
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog

def create_color_qr(size=480, info_size=4):
    """创建基于16级色阶的彩色编码图像"""
    # 验证info_size参数
    if not 2 <= info_size <= 10 or info_size % 2 != 0:
        raise ValueError("信息正方形大小必须是2到10之间的偶数")
    
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
    
    # 红色校准正方形 - 正确排列饱和度，不绘制白色色块
    draw.rectangle([x1 + unit, y1 + unit, x1 + 2*unit, y1 + 2*unit], fill=(255, 0, 0))  # 高饱和度（右下）
    draw.rectangle([x1 + unit, y1, x1 + 2*unit, y1 + unit], fill=(128, 0, 0))  # 中饱和度（右上）
    draw.rectangle([x1, y1, x1 + unit, y1 + unit], fill=(64, 0, 0))  # 低饱和度（左上）
    # 左下角色块保持不绘制（靠近中心）
    
    # 第二象限 (左上) - 绿色校准正方形
    x2, y2 = 0, 0
    
    # 绿色校准正方形 - 正确排列饱和度，不绘制白色色块
    # 右下角色块保持不绘制（靠近中心）
    draw.rectangle([x2, y2 + unit, x2 + unit, y2 + 2*unit], fill=(0, 64, 0))  # 低饱和度（左下）
    draw.rectangle([x2, y2, x2 + unit, y2 + unit], fill=(0, 128, 0))  # 中饱和度（左上）
    draw.rectangle([x2 + unit, y2, x2 + 2*unit, y2 + unit], fill=(0, 255, 0))  # 高饱和度（右上）
    
    # 第三象限 (左下) - 蓝色校准正方形
    x3, y3 = 0, 10 * unit
    
    # 蓝色校准正方形 - 正确排列饱和度，不绘制白色色块
    # 右上角色块保持不绘制（靠近中心）
    draw.rectangle([x3 + unit, y3 + unit, x3 + 2*unit, y3 + 2*unit], fill=(0, 0, 64))  # 低饱和度（右下）
    draw.rectangle([x3, y3 + unit, x3 + unit, y3 + 2*unit], fill=(0, 0, 128))  # 中饱和度（左下）
    draw.rectangle([x3, y3, x3 + unit, y3 + unit], fill=(0, 0, 255))  # 高饱和度（左上）
    
    # 第四象限 (右下) - 黑色校准正方形
    x4, y4 = 10 * unit, 10 * unit
    
    # 黑色校准正方形 - 仅用于方向标识，不绘制白色色块
    draw.rectangle([x4 + unit, y4, x4 + 2*unit, y4 + unit], fill=(0, 0, 0))  # 黑色（右上）
    draw.rectangle([x4 + unit, y4 + unit, x4 + 2*unit, y4 + 2*unit], fill=(0, 0, 0))  # 黑色（右下）
    draw.rectangle([x4, y4 + unit, x4 + unit, y4 + 2*unit], fill=(0, 0, 0))  # 黑色（左下）
    # 左上角色块保持不绘制（靠近中心）
    
    return img, encoded_data

class ColorQRApp:
    def __init__(self, master):
        self.master = master
        self.master.title("彩色编码图像生成器")
        self.master.geometry("600x650")
        
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
            values=["2", "4", "6", "8", "10"],
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

    def on_size_change(self, event):
        # 获取选择的值并更新图像
        self.current_info_size = int(self.size_var.get())
        self.update_image()

    def update_image(self):
        # 生成新的彩色编码图像
        self.img, self.data = create_color_qr(self.img_size, self.current_info_size)
        # 将PIL图像转换为Tkinter可显示的格式
        self.tk_img = ImageTk.PhotoImage(self.img)
        # 更新显示
        self.image_label.config(image=self.tk_img)
        
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