import jax.numpy as jnp

def zero_pad_flm(flm, target_L):
    """
    将球谐系数从带限L扩展到目标带限target_L
    
    Args:
        flm: 原始球谐系数，形状为(L, 2L-1, d)
        target_L: 目标带限
        
    Returns:
        扩展后的球谐系数，形状为(target_L, 2*target_L-1, d)
    """
    L = flm.shape[0]
    d = flm.shape[2] if len(flm.shape) > 2 else 1
    
    # 创建目标尺寸的零数组
    padded_flm = jnp.zeros((target_L, 2*target_L-1, d), dtype=flm.dtype)
    
    # 计算m值的中心位置（在新数组中）
    center_new = target_L - 1
    center_old = L - 1
    
    # 复制原始系数
    for ell in range(L):
        # m的范围是[-ell, ell]
        m_start = center_old - ell  # 原始数组中的起始索引
        m_end = center_old + ell + 1  # 原始数组中的结束索引
        
        # 新数组中的对应位置
        new_m_start = center_new - ell
        new_m_end = center_new + ell + 1
        
        padded_flm = padded_flm.at[ell, new_m_start:new_m_end].set(flm[ell, m_start:m_end])
    
    return padded_flm