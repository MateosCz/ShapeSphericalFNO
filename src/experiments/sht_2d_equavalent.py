import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import numpy as np
import jax
import jax.numpy as jnp
import s2fft
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from src.utils.sht_helper import resize_flm
jax.config.update("jax_enable_x64", True)

def test_sht_identity(L=10, feature_dim=3, sampling="mw", reality=True, method="jax"):
    """
    测试s2fft球谐变换的恒等性（不使用扁平化）
    
    参数:
        L: 球谐展开的最大度数/球面网格分辨率
        feature_dim: 特征维度
        sampling: 采样方案 ("mw"=Driscoll-Healy, "gl"=Gauss-Legendre)
        reality: 是否使用实数值变换
        method: 计算方法
    """
    print(f"测试参数: L={L}, feature_dim={feature_dim}, sampling={sampling}, reality={reality}")
    
    # 1. 生成球面网格上的随机数据
    rng = jax.random.PRNGKey(42)
    sphere_data = jax.random.normal(rng, (L, 2*L-1, feature_dim))
    print(f"原始数据形状: {sphere_data.shape}")
    
    # 2. 直接应用SHT (对每个特征维度分别应用)
    transformed_data = jax.vmap(lambda x: s2fft.forward(x, L, method=method, 
                                                        spin=0, sampling=sampling, 
                                                        reality=reality), 
                               in_axes=2)(sphere_data)
    # 调整轴顺序以便检查
    transformed_data = transformed_data.transpose(1, 2, 0)
    print(f"SHT后数据形状: {transformed_data.shape}")
    
    # 3. 应用逆SHT
    inverse_transformed = jax.vmap(lambda x: s2fft.inverse(x, L, method=method, 
                                                          spin=0, sampling=sampling, 
                                                          reality=reality), 
                                  in_axes=2)(transformed_data)
    inverse_transformed = inverse_transformed.transpose(1, 2, 0)
    print(f"逆SHT后数据形状: {inverse_transformed.shape}")
    
    # 4. 计算恢复误差
    diff = jnp.abs(sphere_data - inverse_transformed)
    max_diff = jnp.max(diff)
    mean_diff = jnp.mean(diff)
    
    print(f"最大差异: {max_diff}")
    print(f"平均差异: {mean_diff}")
    
    # 5. 检查频谱内容 - 帮助诊断问题
    if feature_dim == 1:  # 为简单起见，只展示一个特征维度的频谱
        spectrum = jnp.abs(transformed_data[:, :, 0])
        print(f"频谱幅值统计: min={jnp.min(spectrum)}, max={jnp.max(spectrum)}, mean={jnp.mean(spectrum)}")
    
    return {
        "original": sphere_data,
        "spectrum": transformed_data,
        "recovered": inverse_transformed,
        "max_diff": max_diff,
        "mean_diff": mean_diff
    }

def visualize_sht_results(results, slice_idx=0):
    """可视化SHT变换恢复效果"""
    orig = np.array(results["original"])
    recov = np.array(results["recovered"])
    
    # 检查并处理复数数据
    if np.iscomplexobj(recov):
        print("检测到复数数据，将只显示实部")
        recov = np.real(recov)  # 只取实部
        # 或者使用幅值: recov = np.abs(recov)

    # 对特定切片进行可视化比较
    feature_dim = orig.shape[2]
    
    # 1. 绘制原始数据与恢复数据的对比图
    fig, axes = plt.subplots(feature_dim, 2, figsize=(12, 4*feature_dim))
    if feature_dim == 1:
        axes = axes.reshape(1, 2)
    
    for i in range(feature_dim):
        # 原始数据
        im1 = axes[i, 0].imshow(orig[:, :, i], cmap='viridis')
        axes[i, 0].set_title(f'original feature {i}')
        plt.colorbar(im1, ax=axes[i, 0])
        
        # 恢复数据
        im2 = axes[i, 1].imshow(recov[:, :, i], cmap='viridis')
        axes[i, 1].set_title(f'recovered feature {i}')
        plt.colorbar(im2, ax=axes[i, 1])
    
    plt.tight_layout()
    plt.show()
    
    # 2. 绘制残差图
    fig, axes = plt.subplots(1, feature_dim, figsize=(6*feature_dim, 5))
    if feature_dim == 1:
        axes = [axes]
    
    for i in range(feature_dim):
        # 如果原始数据和恢复数据类型不同，需要特别处理
        if np.iscomplexobj(orig) != np.iscomplexobj(recov):
            if np.iscomplexobj(orig):
                orig_data = np.real(orig[:, :, i])
            else:
                orig_data = orig[:, :, i]
            
            if np.iscomplexobj(recov):
                recov_data = np.real(recov[:, :, i])
            else:
                recov_data = recov[:, :, i]
            
            residual = np.abs(orig_data - recov_data)
        else:
            residual = np.abs(orig[:, :, i] - recov[:, :, i])
        
        im = axes[i].imshow(residual, cmap='hot')
        axes[i].set_title(f'residual feature {i}')
        plt.colorbar(im, ax=axes[i])
    
    plt.tight_layout()
    plt.show()
    
    # 3. 如果是3D数据，绘制一个切片的散点图比较
    if feature_dim >= 3:
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # 处理可能的复数数据
        x_orig = np.real(orig[:, slice_idx, 0].flatten()) if np.iscomplexobj(orig) else orig[:, slice_idx, 0].flatten()
        y_orig = np.real(orig[:, slice_idx, 1].flatten()) if np.iscomplexobj(orig) else orig[:, slice_idx, 1].flatten()
        
        x_recov = np.real(recov[:, slice_idx, 0].flatten()) if np.iscomplexobj(recov) else recov[:, slice_idx, 0].flatten()
        y_recov = np.real(recov[:, slice_idx, 1].flatten()) if np.iscomplexobj(recov) else recov[:, slice_idx, 1].flatten()
        
        ax.scatter(x_orig, y_orig, c='blue', label='original', alpha=0.7)
        ax.scatter(x_recov, y_recov, c='red', label='recovered', alpha=0.7, marker='x')
        ax.set_title(f'feature 0-1 scatter plot comparison (slice {slice_idx})')
        ax.set_xlabel('feature 0')
        ax.set_ylabel('feature 1')
        ax.legend()
        ax.grid(True)
        plt.show()

def test_parameter_variations():
    """测试不同参数组合下的SHT恒等性"""
    # 测试不同的分辨率
    print("=== 测试不同分辨率 ===")
    resolutions = [8, 16, 32]
    for L in resolutions:
        results = test_sht_identity(L=L)
        print(f"分辨率 L={L} 的最大差异: {results['max_diff']:.8f}")
        print("-" * 50)
    
    # 测试不同的采样方案
    print("\n=== 测试不同采样方案 ===")
    sampling_methods = ["mw", "gl", "healpix"]
    for sampling in sampling_methods:
        try:
            results = test_sht_identity(sampling=sampling)
            print(f"采样方案 {sampling} 的最大差异: {results['max_diff']:.8f}")
        except Exception as e:
            print(f"采样方案 {sampling} 出错: {e}")
        print("-" * 50)
    
    # 测试reality参数
    print("\n=== 测试reality参数 ===")
    for reality in [True, False]:
        results = test_sht_identity(reality=reality)
        print(f"reality={reality} 的最大差异: {results['max_diff']:.8f}")
        print("-" * 50)
    
    # 可视化最后一次测试结果
    print("\n=== 可视化最后一次测试结果 ===")
    visualize_sht_results(results)
    
    return results

def test_sht_with_padding(L=10, feature_dim=3, sampling="mw", reality=True, method="jax", target_L=None):
    """
    测试s2fft球谐变换并使用频域零填充的效果
    
    参数:
        L: 球谐展开的初始最大度数
        target_L: 频域零填充后的目标分辨率，默认为2*L
        feature_dim: 特征维度
        sampling: 采样方案 ("mw"=Driscoll-Healy, "gl"=Gauss-Legendre)
        reality: 是否使用实数值变换
        method: 计算方法
    """
    if target_L is None:
        target_L = 2 * L
        
    print(f"测试参数: L={L}, target_L={target_L}, feature_dim={feature_dim}, sampling={sampling}, reality={reality}")
    
    # 1. 生成球面网格上的随机数据
    rng = jax.random.PRNGKey(42)
    sphere_data = jax.random.normal(rng, (L, 2*L-1, feature_dim))
    print(f"原始数据形状: {sphere_data.shape}")
    
    # 2. 直接应用SHT
    transformed_data = jax.vmap(lambda x: s2fft.forward(x, L, method=method, 
                                                      spin=0, sampling=sampling, 
                                                      reality=reality), 
                               in_axes=2)(sphere_data)
    transformed_data = transformed_data.transpose(1, 2, 0)
    print(f"SHT后数据形状: {transformed_data.shape}")
    
    # 3. 使用resize_flm进行频域零填充
    
    padded_spectrum = resize_flm(transformed_data, target_L)
    print(f"频域零填充后形状: {padded_spectrum.shape}")
    
    # 4. 应用逆SHT（使用目标分辨率）
    padded_inverse = jax.vmap(lambda x: s2fft.inverse(x, target_L, method=method, 
                                                    spin=0, sampling=sampling, 
                                                    reality=reality), 
                            in_axes=2)(padded_spectrum)
    padded_inverse = padded_inverse.transpose(1, 2, 0)
    print(f"零填充后逆SHT数据形状: {padded_inverse.shape}")
    
    # 5. 为比较，提取与原始数据相同大小的区域
    # 计算需要裁剪的边界
    lat_start = (target_L - L) // 2
    lat_end = lat_start + L
    lon_start = (2*target_L - 1 - (2*L - 1)) // 2
    lon_end = lon_start + (2*L - 1)
    
    cropped_data = padded_inverse[lat_start:lat_end, lon_start:lon_end, :]
    print(f"裁剪后数据形状: {cropped_data.shape}")
    
    # 6. 计算恢复误差
    # 直接逆变换的误差
    regular_error = jax.vmap(lambda x: s2fft.inverse(x, L, method=method, 
                                                   spin=0, sampling=sampling, 
                                                   reality=reality), 
                           in_axes=2)(transformed_data)
    regular_error = regular_error.transpose(1, 2, 0)
    
    regular_diff = jnp.abs(sphere_data - regular_error)
    regular_max_diff = jnp.max(regular_diff)
    regular_mean_diff = jnp.mean(regular_diff)
    
    # 频域零填充后的误差
    padded_diff = jnp.abs(sphere_data - cropped_data)
    padded_max_diff = jnp.max(padded_diff)
    padded_mean_diff = jnp.mean(padded_diff)
    
    print(f"常规逆变换 - 最大差异: {regular_max_diff}")
    print(f"常规逆变换 - 平均差异: {regular_mean_diff}")
    print(f"零填充逆变换 - 最大差异: {padded_max_diff}")
    print(f"零填充逆变换 - 平均差异: {padded_mean_diff}")
    
    # 7. 检查频谱内容
    if feature_dim == 1:
        reg_spectrum = jnp.abs(transformed_data[:, :, 0])
        pad_spectrum = jnp.abs(padded_spectrum[:, :, 0])
        print(f"原始频谱统计: min={jnp.min(reg_spectrum)}, max={jnp.max(reg_spectrum)}")
        print(f"填充频谱统计: min={jnp.min(pad_spectrum)}, max={jnp.max(pad_spectrum)}")
    
    return {
        "original": sphere_data,
        "spectrum": transformed_data,
        "padded_spectrum": padded_spectrum,
        "regular_recovered": regular_error,
        "padded_recovered": padded_inverse,
        "cropped_recovered": cropped_data,
        "regular_max_diff": regular_max_diff,
        "regular_mean_diff": regular_mean_diff,
        "padded_max_diff": padded_max_diff,
        "padded_mean_diff": padded_mean_diff
    }

def visualize_padding_comparison(results, slice_idx=0):
    """比较常规SHT和零填充SHT的效果"""
    orig = np.array(results["original"])
    reg_recov = np.array(results["regular_recovered"])
    pad_recov = np.array(results["cropped_recovered"])
    
    # 处理可能的复数数据
    if np.iscomplexobj(reg_recov):
        print("检测到复数数据，将显示实部")
        reg_recov = np.real(reg_recov)
    if np.iscomplexobj(pad_recov):
        pad_recov = np.real(pad_recov)
    
    feature_dim = orig.shape[2]
    
    # 1. 对比原始、常规恢复和零填充恢复的数据
    fig, axes = plt.subplots(feature_dim, 3, figsize=(15, 4*feature_dim))
    if feature_dim == 1:
        axes = axes.reshape(1, 3)
    
    for i in range(feature_dim):
        # 原始数据
        im1 = axes[i, 0].imshow(orig[:, :, i], cmap='viridis')
        axes[i, 0].set_title(f'原始数据 特征{i}')
        plt.colorbar(im1, ax=axes[i, 0])
        
        # 常规恢复
        im2 = axes[i, 1].imshow(reg_recov[:, :, i], cmap='viridis')
        axes[i, 1].set_title(f'常规SHT恢复 特征{i}')
        plt.colorbar(im2, ax=axes[i, 1])
        
        # 零填充恢复
        im3 = axes[i, 2].imshow(pad_recov[:, :, i], cmap='viridis')
        axes[i, 2].set_title(f'零填充SHT恢复 特征{i}')
        plt.colorbar(im3, ax=axes[i, 2])
    
    plt.tight_layout()
    plt.show()
    
    # 2. 残差图比较
    fig, axes = plt.subplots(feature_dim, 2, figsize=(12, 4*feature_dim))
    if feature_dim == 1:
        axes = axes.reshape(1, 2)
    
    for i in range(feature_dim):
        # 常规恢复残差
        residual1 = np.abs(orig[:, :, i] - reg_recov[:, :, i])
        im1 = axes[i, 0].imshow(residual1, cmap='hot')
        axes[i, 0].set_title(f'常规SHT残差 特征{i} (最大:{np.max(residual1):.6f})')
        plt.colorbar(im1, ax=axes[i, 0])
        
        # 零填充恢复残差
        residual2 = np.abs(orig[:, :, i] - pad_recov[:, :, i])
        im2 = axes[i, 1].imshow(residual2, cmap='hot')
        axes[i, 1].set_title(f'零填充SHT残差 特征{i} (最大:{np.max(residual2):.6f})')
        plt.colorbar(im2, ax=axes[i, 1])
    
    plt.tight_layout()
    plt.show()
    
    # 3. 频谱比较 (只对第一个特征)
    if feature_dim >= 1:
        orig_spectrum = np.array(results["spectrum"])[:, :, 0]
        padded_spectrum = np.array(results["padded_spectrum"])[:, :, 0]
        
        # 处理复数值
        if np.iscomplexobj(orig_spectrum):
            orig_spectrum = np.abs(orig_spectrum)
        if np.iscomplexobj(padded_spectrum):
            padded_spectrum = np.abs(padded_spectrum)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        im1 = axes[0].imshow(np.log1p(orig_spectrum), cmap='viridis')
        axes[0].set_title('原始频谱 (log scale)')
        plt.colorbar(im1, ax=axes[0])
        
        im2 = axes[1].imshow(np.log1p(padded_spectrum), cmap='viridis')
        axes[1].set_title('零填充频谱 (log scale)')
        plt.colorbar(im2, ax=axes[1])
        
        plt.tight_layout()
        plt.show()

def test_padding_variations():
    """测试不同零填充参数的效果"""
    # 基础参数测试
    print("=== 测试基础零填充效果 ===")
    L = 16
    target_L_values = [L*2, L*4]
    
    for target_L in target_L_values:
        results = test_sht_with_padding(L=L, target_L=target_L)
        print(f"L={L} -> target_L={target_L}:")
        print(f"  常规恢复错误: {results['regular_mean_diff']:.8f}")
        print(f"  零填充恢复错误: {results['padded_mean_diff']:.8f}")
        print("-" * 50)
        
        # 可视化最后一个结果
        visualize_padding_comparison(results)
    
    # 测试不同的reality参数
    print("\n=== 测试reality参数对零填充的影响 ===")
    for reality in [True, False]:
        results = test_sht_with_padding(L=L, target_L=L*2, reality=reality)
        print(f"reality={reality}:")
        print(f"  常规恢复错误: {results['regular_mean_diff']:.8f}")
        print(f"  零填充恢复错误: {results['padded_mean_diff']:.8f}")
        print("-" * 50)
    
    return results

if __name__ == "__main__":
    # final_results = test_parameter_variations()
    padding_results = test_padding_variations()