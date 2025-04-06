# 导入必要的库
import numpy as np  # 数值计算库
import matplotlib.pyplot as plt  # 绘图库
import torch  # PyTorch深度学习框架
from tqdm import tqdm  # 进度条显示
# 从torchvision导入MNIST数据集和相关转换
from torchvision import datasets, transforms

# 设置计算设备，优先使用GPU(cuda:0)，否则使用CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def pixel_value_center_ele(x, y, imgArray):
    """
    计算图像中心像素值的函数
    参数:
        x, y: 归一化坐标(-1到1)
        imgArray: 图像数组
    返回:
        调整后的像素值数组
    """
    # 计算图像中心点
    rows, cols = imgArray.shape
    center_x = rows // 2
    center_y = cols // 2

    # 将归一化坐标转换为实际像素坐标
    x = x * rows
    y = y * cols

    # 调整坐标到以图像中心为原点
    x_adjusted = center_x + torch.round(x).type(torch.int)
    y_adjusted = center_y - torch.round(y).type(torch.int)

    # 初始化输出数组
    values = torch.zeros(x.shape,dtype=imgArray.dtype, device=device)

    # 确定哪些坐标在图像范围内
    valid_coords = (0 <= x_adjusted) & (x_adjusted < rows) & (0 <= y_adjusted) & (y_adjusted < cols)
    valid_coords = valid_coords.type(torch.bool)
    # 计算修正后的y坐标（考虑图像数组索引）
    y_corrected = cols - y_adjusted[valid_coords] - 1

    # 获取有效像素值并赋值到输出数组
    values[valid_coords] = imgArray[x_adjusted[valid_coords], y_corrected]

    return values

# 数据加载部分
# 加载MNIST训练数据
mnist_data = datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())
mnist_label = mnist_data.targets
# 加载MNIST测试数据
mnist_test_data = datasets.MNIST('./data', train=False, download=True, transform=transforms.ToTensor())
test_data = mnist_test_data.data.to(device)
test_label = mnist_test_data.targets.to(device)
# 取前10000个样本作为验证集
validation_data = mnist_data.data[:10000].to(device)
validation_label = mnist_label[:10000].to(device)
# 其余作为训练集
training_data = mnist_data.data[10000:].to(device)
training_label = mnist_label[10000:].to(device)

# 打印各数据集形状
print(training_data.shape)
print(test_data.shape)
print(validation_data.shape)

# 光学参数设置
N = 500  # 采样点数
lmbda = 0.525e-6  # 波长(m)
L = 4e-3  # 光场尺寸(m)
w = 0.51e-3  # 光束半径(m)
z = 0.08  # 传播距离(m)

# 创建坐标轴
x_axis = torch.linspace(-L/2, L/2, N)
y_axis = torch.linspace(-L/2, L/2, N)

# 数据采样比例(0.1表示使用10%的数据)
ratio = 1
training_data = training_data[:int(ratio * training_data.shape[0])]
test_data = test_data[:int(ratio * test_data.shape[0])]
validation_data = validation_data[:int(ratio * validation_data.shape[0])]
training_label = training_label[:int(ratio * training_label.shape[0])]
test_label = test_label[:int(ratio * test_label.shape[0])]
validation_label = validation_label[:int(ratio * validation_label.shape[0])]

# 将标签转换为one-hot编码形式
train_label_mod = torch.zeros([training_label.shape[0],10])
validation_label_mod = torch.zeros([validation_label.shape[0],10])
test_label_mod = torch.zeros([test_label.shape[0],10])
for i in range(training_label.shape[0]):
    train_label_mod[i,training_label[i]] = 1

for i in range((test_label.shape[0])):
    test_label_mod[i,test_label[i]] = 1

for i in range(validation_label.shape[0]):
    validation_label_mod[i,validation_label[i]] = 1

print("Finish loading label")
train_data_num = training_data.shape[0]
test_data_num = test_data.shape[0]
validation_data_num = validation_data.shape[0]

# 初始化光场数组
u0_train = torch.zeros((train_data_num, N, N),dtype=torch.float32, device=device)
u0_test = torch.zeros((test_data_num, N, N),dtype=torch.float32, device=device)
u0_validation = torch.zeros((validation_data_num, N, N),dtype=torch.float32, device=device)

# 创建坐标网格
X, Y = torch.meshgrid(x_axis / (2 * w), y_axis / (2 * w), indexing='ij')
print("Begin processing data")

# 处理训练数据
for t in tqdm(range(train_data_num)):
    ex = training_data[t]
    u0_train[t, :, :] = pixel_value_center_ele(X, Y, ex)

# 处理测试数据
for t in tqdm(range(test_data_num)):
    ex = test_data[t]
    u0_test[t, :, :] = pixel_value_center_ele(X, Y, ex)

# 处理验证数据
for t in tqdm(range(validation_data_num)):
    ex = validation_data[t]
    u0_validation[t, :, :] = pixel_value_center_ele(X, Y, ex)

# 随机可视化一个样本
index = np.random.randint(0, train_data_num)
plt.imshow(u0_train[index].cpu().detach().numpy())
plt.savefig("u0_train.png")
plt.show()
print(train_label_mod[index])

# 保存预处理后的数据
print('Finish preprocessing, now saving to files')
np.save('u0_train_all.npy', u0_train.cpu().detach().numpy())
np.save('u0_test_all.npy', u0_test.cpu().detach().numpy())
np.save('u0_validation_all.npy', u0_validation.cpu().detach().numpy())
np.save('train_label_all.npy', train_label_mod.cpu().detach().numpy())
np.save('test_label_all.npy', test_label_mod.cpu().detach().numpy())
np.save('validation_label_all.npy', validation_label_mod.cpu().detach().numpy())