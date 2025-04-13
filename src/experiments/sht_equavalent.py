import numpy as np
import jax
import jax.numpy as jnp
import s2fft
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def test_sphere_transform_identity(L=10, feature_dim=3, sampling="mw"):
    """
    测试球面数据在变换过程中是否保持恒等性
    
    参数:
        L: 球谐展开的最大度数/球面网格分辨率
        feature_dim: 特征维度
        sampling: 采样方案 ("mw"=Driscoll-Healy, "gl"=Gauss-Legendre)
    """
    print(f"测试参数: L={L}, feature_dim={feature_dim}, sampling={sampling}")
    
    # 1. 生成球面上的随机数据 - 直接在球面网格上生成
    # 在球面网格上生成随机数据，形状为 (L, 2*L-1, feature_dim)
    rng = jax.random.PRNGKey(42)
    sphere_data = jax.random.normal(rng, (L, 2*L-1, feature_dim))
    
    # 2. 将数据扁平化
    flattened_data = sphere_data.reshape(-1, feature_dim)
    print(f"原始扁平化数据形状: {flattened_data.shape}")
    
    # 3. 重塑回球面表示
    reshaped_data = flattened_data.reshape(L, 2*L-1, feature_dim)
    print(f"重塑数据形状: {reshaped_data.shape}")
    
    # 4. 应用球谐变换 (SHT)
    transformed_data = jax.vmap(lambda x: s2fft.forward(x, L, method="jax", 
                                                        spin=0, sampling=sampling, 
                                                        reality=True), 
                              in_axes=2)(reshaped_data)
    transformed_data = transformed_data.transpose(1, 2, 0)  # 调整轴顺序
    print(f"球谐变换后数据形状: {transformed_data.shape}")
    
    # 5. 应用逆球谐变换
    inverse_transformed = jax.vmap(lambda x: s2fft.inverse(x, L, method="jax", 
                                                         spin=0, sampling=sampling, 
                                                         reality=True), 
                                 in_axes=2)(transformed_data)
    inverse_transformed = inverse_transformed.transpose(1, 2, 0)  # 调整轴顺序
    print(f"逆变换后数据形状: {inverse_transformed.shape}")
    
    # 6. 重新扁平化
    reflattened_data = inverse_transformed.reshape(-1, feature_dim)
    print(f"重新扁平化数据形状: {reflattened_data.shape}")
    
    # 7. 比较原始数据和处理后数据
    diff = jnp.abs(flattened_data - reflattened_data)
    max_diff = jnp.max(diff)
    mean_diff = jnp.mean(diff)
    
    print(f"最大差异: {max_diff}")
    print(f"平均差异: {mean_diff}")
    
    # 8. 可视化比较
    return {
        "original": flattened_data,
        "processed": reflattened_data,
        "max_diff": max_diff,
        "mean_diff": mean_diff
    }

def visualize_results(results, point_count=100):
    """可视化原始数据和处理后数据的比较"""
    orig = np.array(results["original"])[:point_count]
    proc = np.array(results["processed"])[:point_count]
    
    # 创建点对点比较图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 对于每个特征维度创建散点图
    for i in range(min(3, orig.shape[1])):
        ax = axes[i]
        ax.scatter(orig[:, i], proc[:, i], alpha=0.5)
        
        # 添加理想线 (y=x)
        min_val = min(orig[:, i].min(), proc[:, i].min())
        max_val = max(orig[:, i].max(), proc[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        ax.set_xlabel(f'original dimension {i}')
        ax.set_ylabel(f'processed dimension {i}')
        ax.set_title(f'dimension {i} comparison')
        ax.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # 如果是3D数据，创建3D散点图比较
    if orig.shape[1] >= 3:
        fig = plt.figure(figsize=(12, 6))
        
        # 原始数据
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.scatter(orig[:, 0], orig[:, 1], orig[:, 2], c='b', marker='o', alpha=0.6)
        ax1.set_title('original data (3D)')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        
        # 处理后数据
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.scatter(proc[:, 0], proc[:, 1], proc[:, 2], c='r', marker='^', alpha=0.6)
        ax2.set_title('processed data (3D)')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        
        plt.tight_layout()
        plt.show()

# 运行测试
def run_tests():
    # 测试不同的分辨率
    resolutions = [8, 16, 32]
    for L in resolutions:
        results = test_sphere_transform_identity(L=L)
        print(f"\n分辨率 L={L} 的最大差异: {results['max_diff']:.8f}")
        print(f"分辨率 L={L} 的平均差异: {results['mean_diff']:.8f}")
        print("-" * 50)
    
    # 为最后一个测试结果可视化
    visualize_results(results)

if __name__ == "__main__":
    run_tests()