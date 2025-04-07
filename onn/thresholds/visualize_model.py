import torch
import matplotlib.pyplot as plt
import numpy as np
from help import OpticalNetwork, N, L, lmbda, z, device

def visualize_weights(model_path):
    # 加载模型
    model = OpticalNetwork(N, L, lmbda, z).to(device)
    model.load_state_dict(torch.load(model_path))
    
    # 创建一个图形，包含多个子图
    fig = plt.figure(figsize=(15, 10))
    
    # 1. 可视化相位调制层的权重
    plt.subplot(2, 3, 1)
    plt.imshow(model.mod1.phase_values.detach().cpu().numpy())
    plt.colorbar()
    plt.title('Phase Modulation Layer 1')
    
    plt.subplot(2, 3, 2)
    plt.imshow(model.mod2.phase_values.detach().cpu().numpy())
    plt.colorbar()
    plt.title('Phase Modulation Layer 2')
    
    plt.subplot(2, 3, 3)
    plt.imshow(model.mod3.phase_values.detach().cpu().numpy())
    plt.colorbar()
    plt.title('Phase Modulation Layer 3')
    
    # 2. 可视化阈值
    plt.subplot(2, 3, 4)
    thresholds = [model.nonlinear1.threshold.item(), model.nonlinear2.threshold.item()]
    plt.bar(['Layer 1', 'Layer 2'], thresholds)
    plt.title('Threshold Values')
    plt.ylabel('Threshold')
    
    # 3. 可视化一个示例输入通过网络的传播过程
    # 创建一个简单的输入（例如高斯光束）
    x = np.linspace(-1, 1, N)
    y = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, y)
    input_field = torch.exp(-(X**2 + Y**2)).to(device)
    input_field = input_field.unsqueeze(0).unsqueeze(0)  # 添加batch和channel维度
    
    # 记录传播过程中的场
    with torch.no_grad():
        # 第一层传播
        x1 = model.mod1(input_field)
        x2 = model.propagate1(x1)
        x3 = model.nonlinear1(x2)
        
        # 可视化第一层非线性后的结果
        plt.subplot(2, 3, 5)
        plt.imshow(torch.abs(x3[0,0]).cpu().numpy())
        plt.colorbar()
        plt.title('After First Nonlinear Layer')
        
        # 第二层传播
        x4 = model.propagate2(x3)
        x5 = model.mod2(x4)
        x6 = model.propagate3(x5)
        x7 = model.nonlinear2(x6)
        
        # 可视化第二层非线性后的结果
        plt.subplot(2, 3, 6)
        plt.imshow(torch.abs(x7[0,0]).cpu().numpy())
        plt.colorbar()
        plt.title('After Second Nonlinear Layer')
    
    plt.tight_layout()
    plt.savefig('model_visualization.png')
    plt.close()
    
    # 打印阈值信息
    print(f"Threshold values:")
    print(f"Layer 1: {model.nonlinear1.threshold.item():.4f}")
    print(f"Layer 2: {model.nonlinear2.threshold.item():.4f}")

if __name__ == "__main__":
    # 指定模型文件路径
    model_path = "weights_large_uniform.pt"  # 或者 "best_model.pt"
    visualize_weights(model_path) 