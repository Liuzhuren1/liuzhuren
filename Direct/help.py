# 导入必要的库
import torch  # PyTorch深度学习框架
import numpy as np  # 数值计算库
import torch.nn as nn  # PyTorch神经网络模块
import torch.nn.functional as F

# 设置计算设备（优先使用GPU）
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# ------------------光学系统参数设置---------------------
N = M = 200  # 采样点数（图像分辨率）
lmbda = 0.515e-6  # 光波波长（单位：米）
L = 4e-3  # 光场尺寸（单位：米）
w = 0.51e-3  # 光束半径（单位：米）
z = 0.08  # 传播距离（单位：米）
j = torch.tensor([0 + 1j], dtype=torch.complex128, device=device)  # 虚数单位

# 上采样尺寸
UPSAMPLED_SIZE = 211

# 像素尺寸参数
INPUT_PIXEL_SIZE = 8e-6  # 输入像素尺寸 (8微米)
UPSAMPLED_PIXEL_SIZE = 7.56e-6  # 上采样后像素尺寸 (7.56微米)

# ---------------------输出区域划分---------------------
square_size = round(M / 20)  # 每个数字区域的边长
canvas_size = M  # 画布大小

# 定义数字位置（10个数字在画布上的坐标）
positions = np.array([
    [0, 0.8], [0, 3.2], [0, 5.6],  # 第一行数字
    [2.4, 0.52], [2.4, 2.12 + 0.4], [2.4, 3.72 + 0.8], [2.4, 5.32 + 1.2],  # 第二行数字
    [4.8, 0.8], [4.8, 3.2], [4.8, 5.6]  # 第三行数字
])

# 计算偏移量使数字区域居中
offset = (canvas_size - 8 * square_size) / 2

# 初始化起始行列索引
start_col = np.zeros(positions.shape[0], dtype=int)
start_row = np.zeros(positions.shape[0], dtype=int)

# 计算每个数字区域的起始像素坐标
for i in range(10):
    row, col = positions[i]
    start_row[i] = round(offset + row * square_size)
    start_col[i] = round(offset + col * square_size)


def count_pixel(img, start_x, start_y, width, height):
    """
    计算图像指定区域内像素值的总和
    参数:
        img: 输入图像张量 (batch_size, height, width)
        start_x: 区域起始x坐标
        start_y: 区域起始y坐标
        width: 区域宽度
        height: 区域高度
    返回:
        每个图像区域像素值的总和
    """
    res = torch.sum(img[:, start_x:start_x + width, start_y:start_y + height], dim=(1, 2))
    return res


def norm(x, eps=1e-9):
    """
    计算张量的L2范数（带小常数防止除零错误）
    参数:
        x: 输入张量
        eps: 极小常数
    返回:
        张量的L2范数
    """
    return torch.sqrt(torch.dot(x, x) + eps)


# ---------------------定义光学神经网络层---------------------
class propagation_layer(nn.Module):
    """
    光传播层：模拟光波在自由空间中的传播
    基于角谱衍射理论实现
    """

    def __init__(self, L, lmbda, z):
        """
        初始化传播层
        参数:
            L: 光场尺寸
            lmbda: 光波波长
            z: 传播距离
        """
        super(propagation_layer, self).__init__()
        self.L = L
        self.lmbda = lmbda
        self.z = z

    def forward(self, u1):
        """
        前向传播计算
        参数:
            u1: 输入光场 (..., M, N)
        返回:
            传播后的光场
        """
        M, N = u1.shape[-2:]  # 获取输入尺寸
        dx = self.L / M  # 采样间隔
        k = 2 * np.pi / self.lmbda  # 波数

        # 创建频率坐标
        fx = torch.linspace(-1 / (2 * dx), 1 / (2 * dx) - 1 / self.L, M,
                            device=u1.device, dtype=u1.dtype)
        FX, FY = torch.meshgrid(fx, fx, indexing='ij')

        # 计算传递函数（角谱衍射）
        H = torch.exp(-j * np.pi * self.lmbda * self.z * (FX ** 2 + FY ** 2)) * \
            torch.exp(j * k * self.z)
        H = torch.fft.fftshift(H, dim=(-2, -1))  # 频域中心化

        # 傅里叶变换计算衍射
        U1 = torch.fft.fft2(torch.fft.fftshift(u1, dim=(-2, -1)))
        U2 = H * U1  # 频域相乘
        u2 = torch.fft.ifftshift(torch.fft.ifft2(U2), dim=(-2, -1))  # 反变换回空域
        return u2


class modulation_layer(nn.Module):
    """
    相位调制层：对光波进行相位调制
    可训练参数实现自适应相位调制
    """

    def __init__(self, M, N):
        """
        初始化调制层
        参数:
            M, N: 调制面的尺寸
        """
        super(modulation_layer, self).__init__()
        # 定义可训练相位参数
        self.phase_values = nn.Parameter(torch.zeros((M, N)))
        nn.init.uniform_(self.phase_values, a=0, b=2)  # 均匀初始化相位值

    def forward(self, input_tensor):
        """
        前向传播计算
        参数:
            input_tensor: 输入光场
        返回:
            调制后的光场
        """
        modulation_matrix = torch.exp(j * 2 * np.pi * self.phase_values)  # 相位调制矩阵
        return input_tensor * modulation_matrix  # 复数乘法实现相位调制


class NonLinearLayer(nn.Module):
    """
    非线性层：模拟光学非线性效应和空间光调制器(SLM)
    实现可训练的参数化非线性处理，使用傅里叶变换方法进行上下采样
    考虑不同像素尺寸：输入8微米，上采样后7.56微米
    """

    def __init__(self, input_size=200, upsampled_size=211, 
                input_pixel_size=INPUT_PIXEL_SIZE, upsampled_pixel_size=UPSAMPLED_PIXEL_SIZE):
        """
        初始化非线性层
        参数:
            input_size: 输入图像尺寸
            upsampled_size: 上采样后的图像尺寸
            input_pixel_size: 输入像素尺寸（8微米）
            upsampled_pixel_size: 上采样后像素尺寸（7.56微米）
        """
        super(NonLinearLayer, self).__init__()
        self.input_size = input_size
        self.upsampled_size = upsampled_size
        self.input_pixel_size = input_pixel_size
        self.upsampled_pixel_size = upsampled_pixel_size
        
        # 计算物理尺寸关系（确保物理场景大小匹配）
        self.input_physical_size = input_size * input_pixel_size
        self.upsampled_physical_size = upsampled_size * upsampled_pixel_size
        
        # 定义可训练参数，用于空间光调制模拟
        self.pixel_weights = nn.Parameter(torch.zeros((upsampled_size, upsampled_size)))
        nn.init.uniform_(self.pixel_weights, a=-1, b=1)  # 初始化参数
        
        # 创建SLM的像素响应模式 (考虑填充因子和衍射效应)
        self.register_buffer('fill_factor', torch.tensor(0.93))  # SLM的填充因子
        
        # 计算相对尺度因子（用于像素响应函数）
        scale_factor = upsampled_pixel_size / input_pixel_size
        
        # 初始化SLM的物理特性参数
        grid_x, grid_y = torch.meshgrid(
            torch.linspace(-upsampled_size//2, upsampled_size//2, upsampled_size),
            torch.linspace(-upsampled_size//2, upsampled_size//2, upsampled_size),
            indexing='ij'
        )
        # 生成像素响应函数（高斯模型）- 调整sigma以匹配物理尺寸
        sigma = 0.5 / scale_factor  # 高斯宽度随像素大小变化
        self.register_buffer('pixel_response', torch.exp(-(grid_x**2 + grid_y**2) / (2 * sigma**2)))

    def fft_resample(self, x, target_size, mode='up', src_pixel_size=None, target_pixel_size=None):
        """
        基于傅里叶变换的重采样方法，保持光场的物理特性
        参数:
            x: 输入光场
            target_size: 目标尺寸
            mode: 'up' 表示上采样，'down' 表示下采样
            src_pixel_size: 源像素尺寸
            target_pixel_size: 目标像素尺寸
        返回:
            重采样后的光场
        """
        # 设置默认像素尺寸
        if src_pixel_size is None:
            src_pixel_size = self.input_pixel_size if mode=='up' else self.upsampled_pixel_size
        if target_pixel_size is None:
            target_pixel_size = self.upsampled_pixel_size if mode=='up' else self.input_pixel_size
            
        # 获取输入尺寸
        input_size = x.shape[-1]
        
        # 计算物理场景尺寸（确保上下采样前后物理尺寸一致）
        src_physical_size = input_size * src_pixel_size
        target_physical_size = target_size * target_pixel_size
        
        # 计算相对尺度变化
        scale_ratio = src_physical_size / target_physical_size
        
        # 执行傅里叶变换 (使用fftshift确保零频率位于中心)
        X = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(x, dim=(-2, -1))), dim=(-2, -1))
        
        if mode == 'up':
            # 上采样：频域零填充，从8微米降到7.56微米
            pad_size = (target_size - input_size) // 2
            X_padded = F.pad(X, (pad_size, pad_size, pad_size, pad_size), "constant", 0)
            
            # 能量校正：由于像素密度增加，需要调整振幅
            # 物理能量守恒：总能量=强度积分，积分区域变小但总能量不变
            energy_scale = (src_pixel_size / target_pixel_size)**2
        else:
            # 下采样：截取频谱中心部分，从7.56微米恢复到8微米
            start = (input_size - target_size) // 2
            end = start + target_size
            X_padded = X[:, :, start:end, start:end] if len(X.shape) == 4 else X[start:end, start:end]
            
            # 能量校正：由于像素密度降低，需要调整振幅
            energy_scale = (src_pixel_size / target_pixel_size)**2
            
        # 执行反傅里叶变换，恢复到空域
        x_resampled = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(X_padded, dim=(-2, -1))), dim=(-2, -1))
        
        # 应用能量守恒校正（复振幅幅值调整）
        x_resampled = x_resampled * torch.sqrt(energy_scale)
        
        return x_resampled

    def forward(self, x):
        """
        前向传播计算
        参数:
            x: 输入光场 (batch_size, height, width) 或 (height, width)
        返回:
            非线性处理后的光场
        """
        # 保存原始形状
        original_shape = x.shape
        
        # 处理复数类型
        is_complex = torch.is_complex(x)
        
        # 确保输入是4D张量 (batch_size, channels, height, width)
        if len(x.shape) == 2:
            x = x.unsqueeze(0).unsqueeze(0)  # 添加batch和channel维度
        elif len(x.shape) == 3:
            x = x.unsqueeze(1)  # 添加channel维度
            
        # 使用傅里叶方法进行上采样 (从8微米到7.56微米)
        x_upsampled = self.fft_resample(
            x, 
            self.upsampled_size, 
            mode='up', 
            src_pixel_size=self.input_pixel_size,
            target_pixel_size=self.upsampled_pixel_size
        )
        
        # 将参数映射到[0,1]区间，模拟SLM的二值响应
        binary_weights = (torch.sigmoid(self.pixel_weights) > 0.5).float()
        
        # 应用像素填充因子和像素响应函数，模拟真实SLM的物理特性
        # 1. 考虑填充因子：非有效区域不影响光场
        effective_weights = binary_weights * self.fill_factor
        
        # 2. 应用像素响应函数（卷积操作模拟像素间的衍射效应）
        # 将响应函数转换为卷积核
        kernel = self.pixel_response.unsqueeze(0).unsqueeze(0)
        
        # 对二值掩码应用卷积，模拟实际的衍射效应
        # 使用mode='same'确保输出尺寸不变
        binary_weights_filtered = F.conv2d(
            effective_weights.unsqueeze(0), 
            kernel, 
            padding=kernel.shape[-1]//2
        ).squeeze(0)
        
        # 应用非线性处理
        x_processed = x_upsampled * binary_weights_filtered
        
        # 使用傅里叶方法进行下采样回原始尺寸 (从7.56微米回到8微米)
        x_downsampled = self.fft_resample(
            x_processed, 
            self.input_size, 
            mode='down',
            src_pixel_size=self.upsampled_pixel_size,
            target_pixel_size=self.input_pixel_size
        )
        
        # 恢复原始形状
        if len(original_shape) == 2:
            return x_downsampled.squeeze(0).squeeze(0)
        elif len(original_shape) == 3:
            return x_downsampled.squeeze(1)
        else:
            return x_downsampled


class imaging_layer(nn.Module):
    """
    成像层：将复振幅转换为强度图像
    """

    def __init__(self):
        super(imaging_layer, self).__init__()

    def forward(self, u):
        """
        前向传播计算
        参数:
            u: 输入复振幅场
        返回:
            光强图像
        """
        return torch.abs(u) ** 2  # 计算光强


# ---------------------光学神经网络模型---------------------
class OpticalNetwork(nn.Module):
    """
    完整的光学神经网络模型
    包含多个调制层和传播层的级联
    """

    def __init__(self, M, L, lmbda, z):
        """
        初始化光学网络
        参数:
            M: 采样点数
            L: 光场尺寸
            lmbda: 光波波长
            z: 传播距离
        """
        super(OpticalNetwork, self).__init__()
        # 定义网络各层
        self.mod1 = modulation_layer(M, M)  # 第一相位调制层
        self.propagate1 = propagation_layer(L, lmbda, z)  # 第一传播层
        self.nonlinear1 = NonLinearLayer(
            input_size=M, 
            upsampled_size=UPSAMPLED_SIZE,
            input_pixel_size=INPUT_PIXEL_SIZE,
            upsampled_pixel_size=UPSAMPLED_PIXEL_SIZE
        )  # 第一非线性层
        self.propagate2 = propagation_layer(L, lmbda, z)  # 第二传播层
        self.mod2 = modulation_layer(M, M)  # 第二相位调制层
        self.propagate3 = propagation_layer(L, lmbda, z)  # 第三传播层
        self.nonlinear2 = NonLinearLayer(
            input_size=M, 
            upsampled_size=UPSAMPLED_SIZE,
            input_pixel_size=INPUT_PIXEL_SIZE,
            upsampled_pixel_size=UPSAMPLED_PIXEL_SIZE
        )  # 第二非线性层
        self.propagate4 = propagation_layer(L, lmbda, z)  # 第四传播层
        self.mod3 = modulation_layer(M, M)  # 第三相位调制层
        self.propagate5 = propagation_layer(L, lmbda, z)  # 第五传播层
        self.nonlinear3 = NonLinearLayer(
            input_size=M, 
            upsampled_size=UPSAMPLED_SIZE,
            input_pixel_size=INPUT_PIXEL_SIZE,
            upsampled_pixel_size=UPSAMPLED_PIXEL_SIZE
        )  # 第二非线性层
        self.propagate6 = propagation_layer(L, lmbda, z)  # 第四传播层
        self.mod4 = modulation_layer(M, M)  # 第三相位调制层
        self.propagate7 = propagation_layer(L, lmbda, z)  # 第五传播层
        self.imaging = imaging_layer()  # 成像层

    def forward(self, x):
        """
        前向传播过程
        参数:
            x: 输入光场
        返回:
            网络输出光强图像
        """
        x = self.mod1(x)  # 第一次相位调制
        x = self.propagate1(x)  # 第一次传播
        x = self.nonlinear1(x)  # 第一非线性处理
        x = self.propagate2(x)  # 第二次传播
        x = self.mod2(x)  # 第二次相位调制
        x = self.propagate3(x)  # 第三次传播
        x = self.nonlinear2(x)  # 第二非线性处理
        x = self.propagate4(x)  # 第四次传播
        x = self.mod3(x)  # 第三次相位调制
        x = self.propagate5(x)  # 第五次传播
        x = self.nonlinear3(x)  # 第三非线性处理
        x = self.propagate6(x)  # 第六次传播
        x = self.mod4(x)  # 第四次相位调制
        x = self.propagate7(x)  # 第七次传播
        x = self.imaging(x)  # 转换为光强图像
        return x