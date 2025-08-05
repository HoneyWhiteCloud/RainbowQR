彩虹码 RainbowQR - 先进的颜色校准二维码生成器
**彩虹码（RainbowQR）**是一个创新的基于颜色编码系统，通过利用RGB颜色通道进行信息存储，革命性地改进了传统二维码技术。与传统的黑白二维码不同，该系统采用红、绿、蓝三个通道的8级色阶方案，在保持视觉美观的同时，大幅提升了数据密度。

核心创新
自动颜色校准：核心特色功能采用位于四个角落的RGB校准正方形。这些正方形包含已知的高、中、低饱和度颜色值，实现针对不同显示设备、光照条件和相机传感器的自动颜色校正。确保即使在不同环境条件下拍摄的图像也能准确解码。

先进图像处理：内置透视校正功能自动检测并修正几何畸变，而精密的颜色空间分析可补偿设备特定的颜色变化。系统智能地在基于HSV的颜色检测和基于区域的标记定位之间切换，以获得最大的可靠性。

增强数据编码：每个像素编码9位信息（每个RGB通道3位），支持UTF-8文本（包括中文字符）、ASCII文本和二进制数据的高效存储。系统包含多层纠错机制：CRC32全局校验和、行列奇偶校验以及基于XOR的冗余校验。

用户友好界面：提供完整的GUI应用程序，具备实时编解码、调试可视化工具和全面的图像分析功能。用户可以从文本生成代码、导入外部图像进行解码，并通过详细的调试窗口监控整个处理流程。

技术架构
Python 3.x 配合OpenCV进行图像处理
PIL/Pillow 用于图像生成和操作
NumPy 进行数值计算和颜色矩阵运算
Tkinter 提供跨平台GUI界面
适用于需要高密度数据存储、艺术性二维码或颜色编码相比传统单色系统具有优势的应用场景。



RainbowQR - Advanced Color-Calibrated QR Code Generator
RainbowQR is an innovative color-based encoding system that revolutionizes traditional QR codes by utilizing RGB color channels for information storage. Unlike conventional black-and-white QR codes, this system leverages an 8-level color gradation scheme across red, green, and blue channels, dramatically increasing data density while maintaining visual appeal.

Key Innovations
Automatic Color Calibration: The cornerstone feature employs RGB calibration squares positioned at four corners of each code. These squares contain known color values at high, medium, and low saturation levels, enabling automatic color correction for varying display devices, lighting conditions, and camera sensors. This ensures accurate decoding even when images are captured under different environmental conditions.

Advanced Image Processing: Built-in perspective correction automatically detects and rectifies geometric distortions, while sophisticated color space analysis compensates for device-specific color variations. The system intelligently switches between HSV-based color detection and region-based marker location for maximum reliability.

Enhanced Data Encoding: Each pixel encodes 9 bits of information (3 bits per RGB channel), supporting efficient storage of UTF-8 text including Chinese characters, ASCII text, and binary data. The system includes multi-layer error correction mechanisms: CRC32 global checksums, row/column parity checks, and XOR-based redundancy.

User-Friendly Interface: Features a complete GUI application with real-time encoding/decoding, debug visualization tools, and comprehensive image analysis capabilities. Users can generate codes from text, import external images for decoding, and monitor the entire processing pipeline through detailed debugging windows.

Technical Stack
Python 3.x with OpenCV for image processing
PIL/Pillow for image generation and manipulation
NumPy for numerical computations and color matrix operations
Tkinter for cross-platform GUI interface
Perfect for applications requiring high-density data storage, artistic QR codes, or scenarios where color-based encoding provides advantages over traditional monochrome systems.

