# 导入必要的库
import torch  # PyTorch深度学习框架
import numpy as np  # 数值计算库
import torch.nn as nn  # PyTorch神经网络模块
import torch.nn.functional as F

# 设置计算设备（优先使用GPU）
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# ------------------光学系统参数设置---------------------
# 从prepro.py中匹配相同的参数
N = M = 200  # 采样点数（图像分辨率）
lmbda = 0.525e-6  # 光波波长（单位：米）
L = 4e-3  # 光场尺寸（单位：米）
w = 0.51e-3  # 光束半径（单位：米）
z = 0.08  # 传播距离（单位：米）
j = torch.tensor([0 + 1j], dtype=torch.complex128, device=device)  # 虚数单位

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
    非线性层：模拟光学非线性效应
    实现阈值处理功能
    """

    def __init__(self, threshold_init=0.5):
        """
        初始化非线性层
        参数:
            threshold_init: 强度阈值初始值
        """
        super(NonLinearLayer, self).__init__()
        # 将threshold转换为可训练参数
        self.threshold = nn.Parameter(torch.tensor([threshold_init]))

    def forward(self, x):
        """
        前向传播计算
        参数:
            x: 输入光场
        返回:
            阈值处理后的光场
        """
        intensity = torch.abs(x) ** 2  # 计算光强
        # 使用sigmoid函数实现平滑的阈值处理
        mask = torch.sigmoid((intensity - self.threshold) * 10)  # 10是平滑因子
        return x * mask


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
        self.nonlinear1 = NonLinearLayer(threshold_init=0.5)  # 第一非线性层，初始阈值0.5
        self.propagate2 = propagation_layer(L, lmbda, z)  # 第二传播层
        self.mod2 = modulation_layer(M, M)  # 第二相位调制层
        self.propagate3 = propagation_layer(L, lmbda, z)  # 第三传播层
        self.nonlinear2 = NonLinearLayer(threshold_init=0.6)  # 第二非线性层，初始阈值0.6
        self.propagate4 = propagation_layer(L, lmbda, z)  # 第四传播层
        self.mod3 = modulation_layer(M, M)  # 第三相位调制层
        self.propagate5 = propagation_layer(L, lmbda, z)  # 第五传播层
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
        x = self.imaging(x)  # 转换为光强图像
        return x