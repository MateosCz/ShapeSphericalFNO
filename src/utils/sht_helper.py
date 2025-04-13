import jax.numpy as jnp

def resize_flm(flm, target_L):
    """
    将球谐系数从带限Lbian   到目标带限target_L
    
    Args:
        flm: 原始球谐系数，形状为(L, 2L-1, d)
        target_L: 目标带限
        
    Returns:
        扩展后的球谐系数，形状为(target_L, 2*target_L-1, d)
    """
    L = flm.shape[0]
    d = flm.shape[2] if len(flm.shape) > 2 else 1
    
    # 创建目标尺寸的零数组
    resized_flm = jnp.zeros((target_L, 2*target_L-1, d), dtype=flm.dtype)
    
    # 计算m值的中心位置（在新数组中）
    center_new = target_L - 1
    center_old = L - 1
    if L < target_L:
    # 复制原始系数
        for ell in range(L):
            # m的范围是[-ell, ell]
            m_start = center_old - ell  # 原始数组中的起始索引
            m_end = center_old + ell + 1  # 原始数组中的结束索引
            
            # 新数组中的对应位置
            new_m_start = center_new - ell
            new_m_end = center_new + ell + 1
            
            resized_flm = resized_flm.at[ell, new_m_start:new_m_end].set(flm[ell, m_start:m_end])
    else:
        # 缩小球谐系数
        for ell in range(target_L):
            # m的范围是[-ell, ell]
            m_start = center_old - ell  # 原始数组中的起始索引
            m_end = center_old + ell + 1  # 原始数组中的结束索引
            
            # 新数组中的对应位置
            new_m_start = center_new - ell
            new_m_end = center_new + ell + 1
            
            resized_flm = resized_flm.at[ell, new_m_start:new_m_end].set(flm[ell, m_start:m_end])
    
    return resized_flm


def resize_spatial(f, target_L):
    """
    将空间域数据从带限L扩展到目标带限target_L，通过在周围对称补零
    
    Args:
        f: 原始空间域数据，形状为(L, 2L-1, d)
        target_L: 目标带限
        
    Returns:
        扩展后的空间域数据，形状为(target_L, 2*target_L-1, d)
    """
    L = f.shape[0]
    current_phi_size = 2 * L - 1
    target_phi_size = 2 * target_L - 1
    
    # 创建目标数组
    resized_f = jnp.zeros((target_L, target_phi_size)) + 0j if f.dtype == jnp.complex128 else jnp.zeros((target_L, target_phi_size))
    
    if len(f.shape) > 2:  # 处理多通道情况
        resized_f = jnp.zeros((target_L, target_phi_size, f.shape[2]), dtype=f.dtype)
    # 计算纬度方向的填充参数
    pad_lat = target_L - L
    start_lat = pad_lat // 2
    end_lat = start_lat + L
    
    # 计算经度方向的填充参数（总是偶数填充）
    pad_lon = target_phi_size - current_phi_size
    start_lon = pad_lon // 2  # 因为pad_lon = 2*(target_L - L)
    end_lon = start_lon + current_phi_size
    
    # 执行填充操作
    return resized_f.at[start_lat:end_lat, 
                         start_lon:end_lon].set(f)

def Legendre_Polynomial(x, L):
    """
    计算Legendre多项式
    
    Args:
        x: 输入值
        L: 多项式阶数
        
    Returns:
        计算结果
    """

    if L == 0:
        return jnp.ones_like(x)
    elif L == 1:
        return x
    else:
        return ((2*L-1)*x*Legendre_Polynomial(x, L-1) - (L-1)*Legendre_Polynomial(x, L-2)) / L

def Legendre_Polynomial_Derivative(x, L):
    """
    计算Legendre多项式的导数
    
    Args:
        x: 输入值
        L: 多项式阶数
        
    Returns:
        计算结果
    """

    if L == 0:
        return jnp.zeros_like(x)
    elif L == 1:
        return jnp.ones_like(x)
    else:
        return (L/x)*Legendre_Polynomial(x, L-1) - ((L-1)/x)*Legendre_Polynomial(x, L-2)

