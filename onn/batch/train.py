# 导入必要的库
from torch.utils.data import Dataset  # 自定义数据集基类
from help import *  # 自定义辅助函数
from tqdm import tqdm  # 进度条显示
import numpy as np  # 数值计算库
import os  # 用于文件操作

# 清空GPU缓存
torch.cuda.empty_cache()

# 启用梯度异常检测（调试用）
torch.autograd.set_detect_anomaly(True)

# 打印当前使用的设备
print(f"Using {device} device")

# -------------------数据加载部分----------------------------
class BatchedNpyDataset(Dataset):
    """
    批处理模式的自定义numpy数据加载类，继承自PyTorch的Dataset类
    用于加载经过预处理后分批存储的.npy格式的数据文件和标签文件
    """
    def __init__(self, data_dir, data_prefix, label_file, batch_info_file):
        """
        初始化数据集
        参数:
            data_dir: 数据目录路径
            data_prefix: 批处理数据文件前缀
            label_file: 标签文件路径
            batch_info_file: 批处理信息文件路径
        """
        self.data_dir = data_dir
        self.data_prefix = data_prefix
        
        # 加载批处理信息
        self.batch_info = np.load(os.path.join(data_dir, batch_info_file), allow_pickle=True).item()
        self.num_batches = self.batch_info[f'{data_prefix}_batches']
        self.batch_size = self.batch_info['batch_size']
        self.total_samples = self.batch_info[f'{data_prefix}_samples']
        
        # 加载标签文件
        self.labels = np.load(os.path.join(data_dir, label_file))
        
        # 存储批处理文件路径
        self.batch_files = [f"{data_dir}/{data_prefix}_batch_{i}.npy" for i in range(self.num_batches)]
        
        # 内存缓存（用于存储当前加载的批次数据）
        self.current_batch_idx = -1
        self.current_batch_data = None

    def __len__(self):
        """返回数据集样本总数"""
        return self.total_samples

    def __getitem__(self, idx):
        """
        获取单个样本
        参数:
            idx: 样本索引
        返回:
            data: 样本数据张量
            labels: 样本标签张量
        """
        # 计算样本在哪个批次中
        batch_idx = idx // self.batch_size
        local_idx = idx % self.batch_size
        
        # 如果是最后一个批次，需要检查索引是否越界
        if batch_idx == self.num_batches - 1:
            last_batch_size = self.total_samples - (self.num_batches - 1) * self.batch_size
            if local_idx >= last_batch_size:
                # 如果索引越界，使用最后一个有效索引
                local_idx = last_batch_size - 1
        
        # 如果需要加载新的批次
        if batch_idx != self.current_batch_idx:
            self.current_batch_idx = batch_idx
            self.current_batch_data = np.load(self.batch_files[batch_idx])
        
        # 获取样本数据和标签
        data = self.current_batch_data[local_idx]
        label = self.labels[idx]
        
        # 转换为PyTorch张量并指定数据类型和设备
        return torch.tensor(data).float().to(device), torch.tensor(label).float().to(device)

# 设置批量大小
batch_size = 128
data_dir = 'preprocessed_data'  # 预处理数据目录

# 创建训练数据集和数据加载器
train_dataset = BatchedNpyDataset(
    data_dir=data_dir,
    data_prefix='u0_train',
    label_file='train_label_all.npy',
    batch_info_file='batch_info.npy'
)
train_loader = torch.utils.data.DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    shuffle=True  # 打乱数据顺序
)

# 创建测试数据集和数据加载器
test_dataset = BatchedNpyDataset(
    data_dir=data_dir,
    data_prefix='u0_test',
    label_file='test_label_all.npy',
    batch_info_file='batch_info.npy'
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, 
    batch_size=batch_size, 
    shuffle=True
)

# 创建验证数据集和数据加载器
validation_dataset = BatchedNpyDataset(
    data_dir=data_dir,
    data_prefix='u0_validation',
    label_file='validation_label_all.npy',
    batch_info_file='batch_info.npy'
)
validation_loader = torch.utils.data.DataLoader(
    validation_dataset, 
    batch_size=batch_size, 
    shuffle=True
)

# 打印各数据集大小
print("successfully loaded,\n size of training data: ", len(train_dataset), 
      "size of test data: ", len(test_dataset), 
      "size of validation data: ", len(validation_dataset))

# ----------------网络定义与训练部分------------------
# 自定义带分类头的光学神经网络
class OpticalNetworkWithClassifier(nn.Module):
    def __init__(self, N, L, lmbda, z):
        super(OpticalNetworkWithClassifier, self).__init__()
        # 初始化基础光学神经网络
        self.optical_net = OpticalNetwork(N, L, lmbda, z)
        # 定义区域边长
        self.square_size = round(N / 20)
        # 使用外部定义的起始位置数组
        self.start_row = start_row  
        self.start_col = start_col
        # 添加归一化层
        self.norm = nn.BatchNorm1d(10)
        
    def forward(self, x):
        # 获取光学神经网络的输出图像
        optical_output = self.optical_net(x)
        
        # 创建存储10个区域值的张量，使用float32类型
        batch_size = x.shape[0]
        class_outputs = torch.zeros(batch_size, 10, device=x.device, dtype=torch.float32)
        
        # 对10个区域进行积分，获取每个区域的光强总和
        for i in range(10):
            # 将积分结果转换为float32
            pixel_sum = count_pixel(
                optical_output, 
                self.start_row[i], 
                self.start_col[i], 
                self.square_size, 
                self.square_size
            ).to(torch.float32)
            class_outputs[:, i] = pixel_sum
        
        # 对输出进行归一化
        class_outputs = self.norm(class_outputs)
        
        # 使用softmax确保输出和为1
        class_outputs = torch.softmax(class_outputs, dim=1)
        
        return class_outputs

# 超参数设置
learning_rate = 0.001  # 学习率
epochs = 10  # 训练轮数
batch_size = 128  # 批量大小
patience = 5  # 早停耐心值

# 初始化带分类头的光学神经网络模型并转移到指定设备
model = OpticalNetworkWithClassifier(N, L, lmbda, z).to(device)
# 使用交叉熵损失函数
loss_function = nn.CrossEntropyLoss()
# 定义优化器（Adam）并添加权重衰减
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
# 添加学习率调度器
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

print("Model created, start training")

# 早停相关变量
best_val_loss = float('inf')
best_epoch = 0
patience_counter = 0

# 训练循环
for epoch in range(epochs):
    # 将模型设置为训练模式
    model.train()
    running_loss = 0.0  # 初始化运行损失
    
    # 使用进度条遍历训练数据加载器
    for batch_data, batch_labels in tqdm(train_loader):
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(batch_data)
        # 计算损失（注意：标签需要是类别索引而不是one-hot编码）
        _, labels = torch.max(batch_labels, 1)  # 将one-hot转换为类别索引
        loss = loss_function(outputs, labels)
        
        # 反向传播
        loss.backward()
        # 参数更新
        optimizer.step()
        
        # 累加损失
        running_loss += loss.item()
    
    # 计算并打印平均训练损失
    avg_train_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {avg_train_loss:.4f}")

    # -----------训练集评估-----------
    model.eval()  # 评估模式
    correct = 0
    total = 0
    
    with torch.no_grad():  # 禁用梯度计算
        for data, labels in train_loader:
            outputs = model(data)
            # 获取预测类别（取最大概率的类别）
            _, predicted = torch.max(outputs.data, 1)
            # 获取真实类别（处理one-hot编码）
            _, true_labels = torch.max(labels.data, 1)
            
            total += labels.size(0)
            correct += (predicted == true_labels).sum().item()
    
    # 计算训练准确率
    accuracy = 100 * correct / total

    # -----------验证集评估-----------
    val_running_loss = 0.0
    correct_val = 0
    total_val = 0
    
    with torch.no_grad():
        for data, labels in validation_loader:
            outputs = model(data)
            # 计算验证损失
            _, labels_idx = torch.max(labels, 1)
            loss = loss_function(outputs, labels_idx)
            val_running_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            _, true_labels = torch.max(labels.data, 1)
            
            total_val += labels.size(0)
            correct_val += (predicted == true_labels).sum().item()
    
    # 计算验证集平均损失和准确率
    avg_val_loss = val_running_loss / len(validation_loader)
    val_accuracy = 100 * correct_val / total_val
    
    # 更新学习率
    scheduler.step(avg_val_loss)
    
    # 打印训练和验证结果
    print(f"Epoch [{epoch+1}/{epochs}], Training Accuracy: {accuracy:.2f}%, \n"
          f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")
    
    # 早停检查
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_epoch = epoch
        patience_counter = 0
        # 保存最佳模型
        torch.save(model.state_dict(), "best_model.pt")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch+1}")
            break

# 加载最佳模型
model.load_state_dict(torch.load("best_model.pt"))

# -----------测试集评估-----------
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    
    for data, labels in test_loader:
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        _, true_labels = torch.max(labels.data, 1)
        
        total += true_labels.size(0)
        correct += (predicted == true_labels).sum().item()

# 打印测试准确率
print(f"Test Accuracy: {100 * correct / total:.2f}%")

# 保存最终模型权重
torch.save(model.state_dict(), "weights_large_uniform.pt")