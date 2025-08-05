English Version
RainbowQR - Advanced Color-Calibrated QR Code Generator
RainbowQR is an innovative color-based encoding system that revolutionizes traditional QR codes by utilizing RGB color channels for information storage. Unlike conventional black-and-white QR codes, this system leverages multi-level color gradation schemes, dramatically increasing data density while maintaining visual appeal.

Development Milestones
The project evolved through five major iterations, each introducing significant enhancements:

v1.0 (Foundation) - Established the core concept with 16-level color gradation and RGB calibration squares at four corners. Implemented basic random color generation with a simple GUI framework for proof-of-concept.

v2.0 (Interface Enhancement) - Refined the calibration square layouts by removing redundant white blocks, improving visual clarity. Enhanced the user interface with adjustable information area sizes and cleaner visual design.

v3.0 (Error Correction) - Introduced comprehensive error correction mechanisms including CRC32 global checksums, row/column parity checks, and Reed-Solomon style redundancy. Added detailed capacity information display.

v4.0 (Text Encoding) - Implemented complete text-to-color encoding/decoding pipeline supporting UTF-8 characters. Added real-time character counting, encoding capacity monitoring, and full encode-decode cycle functionality.

v5.0 (Complete System) - Migrated to 8-level color gradation for improved recognition accuracy. Integrated OpenCV for advanced image processing, perspective correction, and calibration marker detection. Added image import functionality with debug visualization tools.

Key Innovations
Automatic Color Calibration: RGB calibration squares at four corners contain known color values at multiple saturation levels, enabling automatic color correction for varying display devices, lighting conditions, and camera sensors.

Advanced Image Processing: Built-in perspective correction automatically detects and rectifies geometric distortions, while sophisticated color space analysis compensates for device-specific color variations.

Enhanced Data Encoding: Each pixel encodes 9 bits of information (3 bits per RGB channel), supporting efficient storage of UTF-8 text including Chinese characters, ASCII text, and binary data.

User-Friendly Interface: Complete GUI application with real-time encoding/decoding, debug visualization tools, and comprehensive image analysis capabilities.

Technical Stack
Python 3.x with OpenCV for image processing
PIL/Pillow for image generation and manipulation
NumPy for numerical computations and color matrix operations
Tkinter for cross-platform GUI interface
Perfect for applications requiring high-density data storage, artistic QR codes, or scenarios where color-based encoding provides advantages over traditional monochrome systems.

中文版本
彩虹码 RainbowQR - 先进的颜色校准二维码生成器
**彩虹码（RainbowQR）**是一个创新的基于颜色编码系统，通过利用RGB颜色通道进行信息存储，革命性地改进了传统二维码技术。与传统的黑白二维码不同，该系统采用多级色阶方案，在保持视觉美观的同时，大幅提升了数据密度。

开发里程碑
项目经历了五个主要迭代版本，每个版本都引入了重要的功能增强：

v1.0 (基础版) - 建立了16级色阶和四角RGB校准正方形的核心概念。实现了基本的随机颜色生成和简单的GUI框架，作为概念验证。

v2.0 (界面增强版) - 通过移除冗余的白色色块优化了校准正方形布局，提高了视觉清晰度。增强了用户界面，支持可调节的信息区域大小和更清洁的视觉设计。

v3.0 (纠错机制版) - 引入了全面的纠错机制，包括CRC32全局校验和、行列奇偶校验和Reed-Solomon样式冗余。添加了详细的容量信息显示。

v4.0 (文本编码版) - 实现了完整的文本到颜色编解码管道，支持UTF-8字符。添加了实时字符计数、编码容量监控和完整的编解码循环功能。

v5.0 (完整系统版) - 迁移到8级色阶以提高识别准确性。集成OpenCV进行高级图像处理、透视校正和校准标记检测。添加了带调试可视化工具的图像导入功能。

核心创新
自动颜色校准：四个角落的RGB校准正方形包含多个饱和度级别的已知颜色值，实现针对不同显示设备、光照条件和相机传感器的自动颜色校正。

先进图像处理：内置透视校正功能自动检测并修正几何畸变，而精密的颜色空间分析可补偿设备特定的颜色变化。

增强数据编码：每个像素编码9位信息（每个RGB通道3位），支持UTF-8文本（包括中文字符）、ASCII文本和二进制数据的高效存储。

用户友好界面：提供完整的GUI应用程序，具备实时编解码、调试可视化工具和全面的图像分析功能。

技术架构
Python 3.x 配合OpenCV进行图像处理
PIL/Pillow 用于图像生成和操作
NumPy 进行数值计算和颜色矩阵运算
Tkinter 提供跨平台GUI界面
适用于需要高密度数据存储、艺术性二维码或颜色编码相比传统单色系统具有优势的应用场景。
