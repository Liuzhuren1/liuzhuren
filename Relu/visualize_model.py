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
    
    # 创建一个图形，包含2*3子图
    fig = plt.figure(figsize=(18, 12))
    
    # 3. 可视化一个示例输入通过网络的传播过程
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
    
    # 显示输入图像
    plt.subplot(2, 3, 1)
    plt.imshow(input_field.cpu().numpy())
    plt.colorbar()
    plt.title(f'Input Image (Label: {label})')
    
    # 可视化非线性函数 - 手动创建正确的非线性函数
    plt.subplot(2, 3, 2)
    x = np.linspace(0, 2, 1000)  # 创建输入范围
    
    # 第一层非线性函数：0.5以下为0，以上不变
    y1 = np.zeros_like(x)
    mask1 = x > 0.5
    y1[mask1] = x[mask1]
    
    # 第二层非线性函数：0.6以下为0，以上不变
    y2 = np.zeros_like(x)
    mask2 = x > 0.6
    y2[mask2] = x[mask2]
    
    plt.plot(x, y1, label=f'Layer 1 (threshold=0.5)')
    plt.plot(x, y2, label=f'Layer 2 (threshold=0.6)')
    plt.axvline(x=0.5, color='r', linestyle='--', alpha=0.5)
    plt.axvline(x=0.6, color='g', linestyle='--', alpha=0.5)
    plt.xlabel('Input Intensity')
    plt.ylabel('Output Intensity')
    plt.title('Nonlinear Layer Functions')
    plt.legend()
    plt.grid(True)
    
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
        
        # 最终输出 - 只显示原始输出
        output = x10
        final_intensity = torch.abs(output[0]).cpu().numpy()
        
        # 添加原始输出
        plt.subplot(2, 3, 3)
        plt.imshow(final_intensity)
        plt.colorbar()
        plt.title('Final Output (Original)')
        
        # 在原始输出上添加10个规定区域的标注框
        for i in range(10):
            # 获取当前区域的起始位置
            start_x = start_row[i]
            start_y = start_col[i]
            # 绘制矩形框
            rect = plt.Rectangle((start_y, start_x), square_size, square_size, 
                               fill=False, edgecolor='red', linewidth=1)
            plt.gca().add_patch(rect)
        
        # 1. 可视化相位调制层的权重
        plt.subplot(2, 3, 4)
        plt.imshow(model.mod1.phase_values.detach().cpu().numpy())
        plt.colorbar()
        plt.title('Phase Modulation Layer 1')
        
        plt.subplot(2, 3, 5)
        plt.imshow(model.mod2.phase_values.detach().cpu().numpy())
        plt.colorbar()
        plt.title('Phase Modulation Layer 2')
        
        plt.subplot(2, 3, 6)
        plt.imshow(model.mod3.phase_values.detach().cpu().numpy())
        plt.colorbar()
        plt.title('Phase Modulation Layer 3')
    
    plt.tight_layout()
    plt.savefig('model_visualization.png')
    plt.close()
    
    # 打印阈值信息
    print(f"Threshold values:")
    print(f"Layer 1: {model.nonlinear1.threshold.item():.4f}")
    print(f"Layer 2: {model.nonlinear2.threshold.item():.4f}")
    print(f"Final output intensity range: {final_intensity.min():.4f} to {final_intensity.max():.4f}")
    print(f"Selected sample {sample_idx} from training set with label {label}")

if __name__ == "__main__":
    # 指定模型文件路径
    model_path = "weights_large_uniform.pt"  # 或者 "best_model.pt"
    visualize_weights(model_path) 