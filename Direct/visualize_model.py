import torch
import matplotlib.pyplot as plt
import numpy as np
from help import OpticalNetwork, N, L, lmbda, z, device, start_row, start_col, square_size
import random

def visualize_weights(model_path):
    # 加载模型
    model = OpticalNetwork(N, L, lmbda, z).to(device)
    
    # 加载状态字典并处理键值
    state_dict = torch.load(model_path)
    new_state_dict = {}
    
    # 处理状态字典中的键值
    for key, value in state_dict.items():
        # 移除 "optical_net." 前缀
        if key.startswith('optical_net.'):
            new_key = key.replace('optical_net.', '')
            new_state_dict[new_key] = value
        # 跳过 norm 相关的参数
        elif key.startswith('norm.'):
            continue
        else:
            new_state_dict[key] = value
    
    # 加载处理后的状态字典
    model.load_state_dict(new_state_dict, strict=False)
    
    # 创建一个图形，包含3*3子图
    fig = plt.figure(figsize=(18, 18))
    
    # 从训练集文件中随机抽取一个样本
    data_file = 'u0_train_all.npy'
    label_file = 'train_label_all.npy'
    
    # 加载数据
    data = np.load(data_file, mmap_mode='r')
    labels = np.load(label_file, mmap_mode='r')
    
    # 随机选择一个样本
    sample_idx = random.randint(0, len(data) - 1)
    # 创建数据的副本，使其可写
    input_field = torch.from_numpy(data[sample_idx].copy()).float().to(device)
    label = labels[sample_idx].copy()
    
   
    # 可视化第一层调制层
    plt.subplot(3, 3, 1)
    phase_values1 = model.mod1.phase_values.detach().cpu().numpy()
    plt.imshow(phase_values1)
    plt.colorbar()
    plt.title('Phase Modulation Layer 1')
    
    # 可视化第二层调制层
    plt.subplot(3, 3, 2)
    phase_values2 = model.mod2.phase_values.detach().cpu().numpy()
    plt.imshow(phase_values2)
    plt.colorbar()
    plt.title('Phase Modulation Layer 2')
    
    # 可视化第三层调制层
    plt.subplot(3, 3, 3)
    phase_values3 = model.mod3.phase_values.detach().cpu().numpy()
    plt.imshow(phase_values3)
    plt.colorbar()
    plt.title('Phase Modulation Layer 3')

     # 显示输入图像
    plt.subplot(3, 3, 4)
    plt.imshow(input_field.cpu().numpy())
    plt.colorbar()
    plt.title(f'Input Image (Label: {label})')
    
    # 可视化第一层非线性层
    plt.subplot(3, 3, 5)
    # 获取非线性层的二值化权重
    binary_weights1 = (torch.sigmoid(model.nonlinear1.pixel_weights) > 0.5).float().detach().cpu().numpy()
    plt.imshow(binary_weights1)
    plt.colorbar()
    plt.title('Nonlinear Layer 1 Weights')
    
    # 可视化第二层非线性层
    plt.subplot(3, 3, 6)
    # 获取非线性层的二值化权重
    binary_weights2 = (torch.sigmoid(model.nonlinear2.pixel_weights) > 0.5).float().detach().cpu().numpy()
    plt.imshow(binary_weights2)
    plt.colorbar()
    plt.title('Nonlinear Layer 2 Weights')
    
    # 记录传播过程中的场
    with torch.no_grad():
        # 第一层传播
        x1 = model.mod1(input_field.unsqueeze(0))
        x2 = model.propagate1(x1)
        x3 = model.nonlinear1(x2)
        
        # 第二层传播
        x4 = model.propagate2(x3)
        x5 = model.mod2(x4)
        x6 = model.propagate3(x5)
        x7 = model.nonlinear2(x6)
        
        # 第四次传播
        x8 = model.propagate4(x7)
        x9 = model.mod3(x8)
        
        # 第五次传播
        x10 = model.propagate5(x9)
        
        # 最终输出
        output = model.imaging(x10)
        final_intensity = output[0].cpu().numpy()
    
    # 显示最终输出
    plt.subplot(3, 3, 7)
    plt.imshow(final_intensity)
    plt.colorbar()
    plt.title('Final Output Intensity')
    
    # 在最终输出上添加10个规定区域的标注框
    for i in range(10):
        # 获取当前区域的起始位置
        start_x = start_row[i]
        start_y = start_col[i]
        # 绘制矩形框
        rect = plt.Rectangle((start_y, start_x), square_size, square_size, 
                           fill=False, edgecolor='red', linewidth=1)
        plt.gca().add_patch(rect)
    
    # 显示第一层非线性层输入
    plt.subplot(3, 3, 8)
    intensity_before_nonlinear1 = torch.abs(x2[0]).cpu().numpy()
    plt.imshow(intensity_before_nonlinear1)
    plt.colorbar()
    plt.title('Input Intensity to Nonlinear Layer 1')
    
    # 显示第二层非线性层输入
    plt.subplot(3, 3, 9)
    intensity_before_nonlinear2 = torch.abs(x6[0]).cpu().numpy()
    plt.imshow(intensity_before_nonlinear2)
    plt.colorbar()
    plt.title('Input Intensity to Nonlinear Layer 2')
    
    plt.tight_layout()
    plt.savefig('model_visualization.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    visualize_weights("best_model.pt") 