# -*- coding: utf-8 -*-
"""
Created on 2025/4/23 21:05

@author: Prince
"""
"""
Duplication of the vector fitting algorithm in python (http://www.sintef.no/Projectweb/VECTFIT/)

All credit goes to Bjorn Gustavsen for his MATLAB implementation, and the following papers


 [1] B. Gustavsen and A. Semlyen, "Rational approximation of frequency
     domain responses by Vector Fitting", IEEE Trans. Power Delivery,
     vol. 14, no. 3, pp. 1052-1061, July 1999.

 [2] B. Gustavsen, "Improving the pole relocating properties of vector
     fitting", IEEE Trans. Power Delivery, vol. 21, no. 3, pp. 1587-1592,
     July 2006.

 [3] D. Deschrijver, M. Mrozowski, T. Dhaene, and D. De Zutter,
     "Macromodeling of Multiport Systems Using a Fast Implementation of
     the Vector Fitting Method", IEEE Microwave and Wireless Components
     Letters, vol. 18, no. 6, pp. 383-385, June 2008.
"""
__author__ = 'Phil Reinhold''Prince'

# vectfit3.py module.
"""
*** FastRelaxed Vector Fitting for Python v1.3***

vectfit3.py is the implementation of Fast Relaxed Vector Fitting algortihm on python. The original code was written 
in Matlab eviroment. The pourpose of this algorithm is to compute a rational approximation from tabuled data in the 
frequency domain for scalar or vectorized problems. The resulting model can be expressed in either state-space form 
or pole-residue form.

    - Original Matlab code autor: Bjorn Gustavsen (08/2008)
    - Transcripted and adapted by: Sebastian Loaiza (03/2024)
    - Last revision by: Sebastian Loaiza (01/2025)

 * References:

    [1] B. Gustavsen and A. Semlyen, "Rational approximation of frequency       
        domain responses by Vector Fitting", IEEE Trans. Power Delivery,        
        vol. 14, no. 3, pp. 1052-1061, July 1999.

    [2] B. Gustavsen, "Improving the pole relocating properties of vector
        fitting", IEEE Trans. Power Delivery, vol. 21, no. 3, pp. 1587-1592,
        July 2006.

    [3] D. Deschrijver, M. Mrozowski, T. Dhaene, and D. De Zutter,
        "Macromodeling of Multiport Systems Using a Fast Implementation of
        the Vector Fitting Method", IEEE Microwave and Wireless Components 
        Letters, vol. 18, no. 6, pp. 383-385, June 2008.

    [4] B. Gustavsen, "User's Guide for vectfit3.m (Fast, Relaxed Vector 
        fitting)", SINTEF Energy Research, N-7465 Trondheim, Norway, 2008. Aviable 
        online: https://www.sintef.no/en/software/vector-fitting/downloads/#menu
        accesed on: 2/2/2024

 * Changes:

    - All options for vectfit3 configuration are defined as boolean variables, except asymp which has 3 posible states
    - A new option, "lowert_mat" is added for vectfit3 configuration. This indicates when F(s) samples belong to a 
      lower triangular matrix function, that reduces the number of elements to fit for a symmetric matrix function.
    - A new option, "RMO_data" is added for vecfit3 configuration. This shows that matrix function elements are saved in 
      Row Major Order into F(s).
    - New options mentioned before and "cmplx_ss" flag are also included in SER.
    - A new method to sort the poles computed during the identification process is implemented.
    - tri2full() function is renamed to flat2full(). It is modified to consider asymmetric matrix problems as well 
      depending on the status of "lower_mat" and "RMO_data" flags.
    - ss2pr() function is replaced by to buildRES(). The new function just compute residues matrixes becouse vectfit returns 
      the poles already.
"""
# 代码1部分：矢量拟合算法
import re
import pandas as pd
import numpy as np
import schemdraw
import schemdraw.elements as elm
# 避免使用 "from pylab import *" 污染命名空间，统一使用 numpy/matplotlib 的命名
from numpy.linalg import eigvals, lstsq
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple

def parse_complex(s: str) -> complex:
    """
    将形如 "1+2j"、"1-2j" 的字符串解析为复数。
    这里先去除空白并修正类似 "1+-2j" 的格式，保证 complex() 能正常解析。
    """
    s = re.sub(r'\s+', '', s)
    s = re.sub(r'(?<=\d)\+?(?=-)', '', s)  # 可选：跟你类里的保持一致
    return complex(s)
def cc(z: complex) -> complex:
    """复共轭（显式函数便于阅读与调试）。"""
    return z.conjugate()

def model(s, poles, residues, d, h):
    """
    频域有理函数模型：
        F(s) = Σ r_k/(s - p_k) + d + s*h
    """
    return sum(r / (s - p) for p, r in zip(poles, residues)) + d + s * h

def vectfit_step(f, s, poles, conj_tol=1e-10):
    """
    f = complex data to fit
    s = j*frequency
    poles = initial poles guess
        note: All complex poles must come in sequential complex conjugate pairs
    returns adjusted poles
    """
    N = len(poles)
    Ns = len(s)

    # 用整型数组标识极点类型，避免浮点比较问题
    cindex = np.zeros(N, dtype=np.int8)
    # cindex is:
    #   - 0 for real poles
    #   - 1 for the first of a complex-conjugate pair
    #   - 2 for the second of a cc pair
    for i, p in enumerate(poles):
        if p.imag != 0:
            if i == 0 or cindex[i-1] != 1:
                # 使用容差判断共轭，避免浮点误差导致断言失败
                if not np.isclose(cc(poles[i]), poles[i+1], atol=conj_tol, rtol=0):
                    raise ValueError(
                        "Complex poles must come in conjugate pairs within tolerance: %s, %s" % (poles[i], poles[i+1])
                    )
                cindex[i] = 1
            else:
                cindex[i] = 2

    # First linear equation to solve. See Appendix A
    # 使用 complex128 保持数值精度，减轻高阶拟合时的病态问题
    A = np.zeros((Ns, 2 * N + 2), dtype=np.complex128)
    for i, p in enumerate(poles):
        if cindex[i] == 0:
            A[:, i] = 1/(s - p)
        elif cindex[i] == 1:
            A[:, i] = 1/(s - p) + 1/(s - cc(p))
        elif cindex[i] == 2:
            A[:, i] = 1j/(s - p) - 1j/(s - cc(p))
        else:
            raise RuntimeError("cindex[%s] = %s" % (i, cindex[i]))

        A [:, N+2+i] = -A[:, i] * f

    A[:, N] = 1
    A[:, N+1] = s

    # Solve Ax == b using pseudo-inverse
    b = f
    A = np.vstack((np.real(A), np.imag(A)))
    b = np.concatenate((np.real(b), np.imag(b)))
    x, residuals, rnk, s = lstsq(A, b, rcond=-1)

    residues = x[:N]
    d = x[N]
    h = x[N+1]

    # We only want the "tilde" part in (A.4)
    x = x[-N:]

    # Calculation of zeros: Appendix B
    A = np.diag(poles)
    b = np.ones(N)
    c = x
    for i, (ci, p) in enumerate(zip(cindex, poles)):
        if ci == 1:
            x, y = np.real(p), np.imag(p)
            A[i, i] = A[i+1, i+1] = x
            A[i, i+1] = -y
            A[i+1, i] = y
            b[i] = 2
            b[i+1] = 0
            #cv = c[i]
            #c[i,i+1] = real(cv), imag(cv)

    H = A - np.outer(b, c)
    H = np.real(H)
    new_poles = np.sort(eigvals(H))
    # 稳定化处理：若实部为正则进行镜像映射
    unstable = np.real(new_poles) > 0
    new_poles[unstable] -= 2 * np.real(new_poles)[unstable]
    return new_poles

# Dear gods of coding style, I sincerely apologize for the following copy/paste
def calculate_residues(f, s, poles, rcond=-1, conj_tol=1e-10):
    Ns = len(s)
    N = len(poles)

    # 用整型数组标识极点类型，避免浮点比较问题
    cindex = np.zeros(N, dtype=np.int8)
    for i, p in enumerate(poles):
        if p.imag != 0:
            if i == 0 or cindex[i-1] != 1:
                # 使用容差判断共轭，避免浮点误差导致断言失败
                if not np.isclose(cc(poles[i]), poles[i+1], atol=conj_tol, rtol=0):
                    raise ValueError(
                        "Complex poles must come in conjugate pairs within tolerance: %s, %s" % (poles[i], poles[i+1])
                    )
                cindex[i] = 1
            else:
                cindex[i] = 2

    # use the new poles to extract the residues
    A = np.zeros((Ns, N + 2), dtype=np.complex128)
    for i, p in enumerate(poles):
        if cindex[i] == 0:
            A[:, i] = 1/(s - p)
        elif cindex[i] == 1:
            A[:, i] = 1/(s - p) + 1/(s - cc(p))
        elif cindex[i] == 2:
            A[:, i] = 1j/(s - p) - 1j/(s - cc(p))
        else:
            raise RuntimeError("cindex[%s] = %s" % (i, cindex[i]))

    A[:, N] = 1
    A[:, N+1] = s
    # Solve Ax == b using pseudo-inverse
    b = f
    A = np.vstack((np.real(A), np.imag(A)))
    b = np.concatenate((np.real(b), np.imag(b)))
    cA = np.linalg.cond(A)
    if cA > 1e13:
        print ('Warning!: Ill Conditioned Matrix. Consider scaling the problem down')
        print ('Cond(A)', cA)
    x, residuals, rnk, s = lstsq(A, b, rcond=rcond)

    # Recover complex values
    # 保持 complex128 精度，避免精度损失
    x = np.complex128(x)
    for i, ci in enumerate(cindex):
       if ci == 1:
           r1, r2 = x[i:i+2]
           x[i] = r1 - 1j*r2
           x[i+1] = r1 + 1j*r2

    residues = x[:N]
    d = np.real(x[N])
    h = np.real(x[N + 1])
    return residues, d, h

def print_params(poles, residues, d, h):
    cfmt = "{0.real:g} + {0.imag:g}j"
    print ("poles: " + ", ".join(cfmt.format(p) for p in poles))
    print ("residues: " + ", ".join(cfmt.format(r) for r in residues))
    print ("offset: {:g}".format(d))
    print ("slope: {:g}".format(h))

def vectfit_auto(
    f,
    s,
    n_poles=20,
    n_iter=20,
    show=False,
    inc_real=False,
    loss_ratio=1e-2,
    rcond=-1,
    track_poles=False,
    weights: Optional[np.ndarray] = None,
    conj_tol: float = 1e-10,
    verbose: bool = False,
):
    """
    自动矢量拟合主流程。
    - weights: 频率点权重（例如 1/|F| 或 1/sqrt(|F|)），用于改善数值稳定性与误差分配。
    - conj_tol: 共轭判定容差。
    - verbose: 是否输出调试信息。
    """
    w = np.imag(s)
    pole_locs = np.linspace(w[0], w[-1], n_poles + 2)[1:-1]
    lr = loss_ratio
    init_poles = poles = np.concatenate([[p * (-lr + 1j), p * (-lr - 1j)] for p in pole_locs])

    if inc_real:
        poles = np.concatenate((poles, [1]))

    # 对输入数据进行加权（若提供 weights）
    if weights is not None:
        # 保证权重形状与 f/s 相同，避免广播错误
        weights = np.asarray(weights).reshape(-1)
        if weights.shape[0] != f.shape[0]:
            raise ValueError("weights 的长度必须与 f/s 的长度一致")
        f_fit = f * weights
    else:
        f_fit = f

    poles_list = []
    converged = False
    for it in range(n_iter):
        old_poles = poles.copy()
        poles = vectfit_step(f_fit, s, poles, conj_tol=conj_tol)
        poles_list.append(poles)
        
        # 收敛性检查
        # 对极点排序以确保对应位置比较（vectfit_step内部已对eigenvalues排序，但为了保险再做一次）
        # 注意：由于复数排序的模糊性，应该基于模或实部/虚部综合判断，但通常norm(diff)足够
        # 这里简单使用整体向量的相对变化率
        diff_norm = np.linalg.norm(poles - old_poles)
        base_norm = np.linalg.norm(old_poles)
        if base_norm > 0:
            change = diff_norm / base_norm
        else:
            change = diff_norm
            
        if verbose:
            print(f"Iter {it+1}/{n_iter}: Pole Change = {change:.6e}")
            
        if change < 1e-5: # 收敛阈值
            converged = True
            if verbose:
                print(f"Converged at iteration {it+1}")
            break

    residues, d, h = calculate_residues(f_fit, s, poles, rcond=rcond, conj_tol=conj_tol)

    if verbose and not converged:
        print(f"Warning: Maximum iterations ({n_iter}) reached without full convergence.")

    if verbose:
        print_params(poles, residues, d, h)

    if track_poles:
        return poles, residues, d, h, np.array(poles_list)

    # print_params(poles, residues, d, h)
    return poles, residues, d, h

def check_passivity(s, poles, residues, d, h):
    """
    检查模型的无源性 (Passivity Check)。
    对于导纳/阻抗函数，无源性要求 Re[Y(s)] >= 0 对于所有 Re[s] >= 0。
    这里主要检查虚轴上的实部是否非负。
    
    返回: (is_passive, min_real_part, violation_freq_hz)
    """
    # 重新计算拟合模型的响应
    resp = model(s, poles, residues, d, h)
    real_part = np.real(resp)
    
    min_real = np.min(real_part)
    is_passive = min_real >= -1e-12 # 允许微小的数值误差
    
    violation_freq = None
    if not is_passive:
        idx = np.argmin(real_part)
        w = np.imag(s[idx])
        violation_freq = w / (2 * np.pi)
        
    return is_passive, min_real, violation_freq

def vectfit_auto_rescale(f, s, **kwargs):
    """
    对 s 和 f 做尺度缩放后再拟合，改善病态问题。
    采用 max(|s|), max(|f|) 避免末点为 0 时除零。
    """
    s_scale = np.max(np.abs(s))
    f_scale = np.max(np.abs(f))
    if s_scale == 0 or f_scale == 0:
        raise ValueError("s 或 f 的尺度为 0，无法进行缩放拟合")
    # print ('SCALED')
    poles_s, residues_s, d_s, h_s = vectfit_auto(f / f_scale, s / s_scale, **kwargs)
    poles = poles_s * s_scale
    residues = residues_s * f_scale * s_scale
    d = d_s * f_scale
    h = h_s * f_scale / s_scale
    # print ('UNSCALED')
    # 是否打印由上层 verbose 控制；此处不强制输出
    return poles, residues, d, h

def calculate_error_metrics(f_orig, f_fitted):
    """
    计算拟合误差指标
    返回包含多种误差定义的字典
    """
    diff = np.abs(f_orig - f_fitted)
    abs_orig = np.abs(f_orig)
    
    # 防止分母为0
    valid_idx = abs_orig > 1e-15
    
    # 1. 均方根绝对误差 (RMS Absolute Error)
    rms_abs = np.sqrt(np.mean(diff**2))
    
    # 2. 最大绝对误差 (Max Absolute Error)
    max_abs = np.max(diff)
    
    # 3. 均方根相对误差 (RMS Relative Error) - 最常用的评估指标
    if np.any(valid_idx):
        rel_diff = diff[valid_idx] / abs_orig[valid_idx]
        rms_rel = np.sqrt(np.mean(rel_diff**2))
        max_rel = np.max(rel_diff)
    else:
        rms_rel = np.inf
        max_rel = np.inf
        
    return {
        'rms_abs': rms_abs,
        'max_abs': max_abs,
        'rms_rel': rms_rel,
        'max_rel': max_rel
    }

def vectfit_find_best_order(f, s, min_poles=2, max_poles=40, step=2, target_error=1e-4, **kwargs):
    """
    自动寻找最优拟合阶数 (n_poles)。
    
    参数:
        f, s: 频域数据
        min_poles, max_poles, step: 阶数扫描范围和步长
        target_error: 目标 RMS 相对误差。如果某阶数满足此误差，则认为足够好并停止搜索（优先选择低阶模型）。
                      如果想找绝对最小误差，可将此值设为 0。
        **kwargs: 传递给 vectfit_auto_rescale 的其他参数 (如 n_iter)
        
    返回:
        best_poles, best_residues, best_d, best_h, best_metrics
    """
    best_result = None
    best_metrics = None
    best_error_score = float('inf') # 用于比较的误差分数
    best_order = -1
    
    history = [] # 记录 (order, err)

    print(f"\n[Auto-Fit] 开始自动寻找最优阶数 (范围: {min_poles} ~ {max_poles}, 步长: {step})...")
    print(f"{'Order':<6} | {'RMS Rel Error':<15} | {'Max Rel Error':<15} | {'Status'}")
    print("-" * 60)
    
    # 遍历阶数
    for n in range(min_poles, max_poles + 1, step):
        # 执行拟合
        # 注意：使用 vectfit_auto_rescale 以获得数值稳定性
        try:
            poles, residues, d, h = vectfit_auto_rescale(f, s, n_poles=n, verbose=False, **kwargs)
            
            # 计算拟合数据
            f_fitted = model(s, poles, residues, d, h)
            
            # 计算误差
            metrics = calculate_error_metrics(f, f_fitted)
            err_score = metrics['rms_rel'] # 主要参考 RMS 相对误差
            
            status = ""
            if err_score < best_error_score:
                best_error_score = err_score
                best_order = n
                best_result = (poles, residues, d, h)
                best_metrics = metrics
                status = "*" # 标记为当前最佳
            
            print(f"{n:<6} | {metrics['rms_rel']:<15.6e} | {metrics['max_rel']:<15.6e} | {status}")
            history.append((n, err_score))

            # 检查是否满足目标误差 (早停机制)
            if err_score < target_error:
                print(f"-" * 60)
                print(f"[Auto-Fit] 阶数 {n} 已满足目标误差 ({target_error})。停止搜索。")
                break
                
        except Exception as e:
            print(f"{n:<6} | {'Failed':<15} | {str(e)}")

    print(f"[Auto-Fit] 最优结果: 阶数 = {best_order}, RMS相对误差 = {best_error_score:.6e}")
    return best_result[0], best_result[1], best_result[2], best_result[3], best_metrics

class SystemAnalyzer:
    DEFAULT_TOLERANCE = 1e-16

    def __init__(self):
        self.data = {
            'poles': [],
            'residues': [],
            'offset': 0.0,
            'slope': 0.0
        }
        self.output_data = {
            'offset': None,
            'slope': None,
            'rl_pairs': [],
            'rlc_pairs': [],
            'rc_params': None,
            'rl_params': [],
            'rlc_params': []
        }
        self.rc_num = 0
        self.rl_num = 0
        self.rlc_num = 0
        self.valid = False

    def load_fitting_result(self, poles, residues, d, h):
        """
        直接加载矢量拟合的结果数据，无需进行字符串转换。
        """
        self.data['poles'] = poles
        self.data['residues'] = residues
        
        # 确保 d 和 h 是实数标量（如果是复数且虚部极小，取实部）
        self.data['offset'] = d.real if isinstance(d, complex) else d
        self.data['slope'] = h.real if isinstance(h, complex) else h
        
        self.output_data['offset'] = self.data['offset']
        self.output_data['slope'] = self.data['slope']
        
        # 执行分析流程
        self.classify_poles()
        self.calculate_parameters()
        self.valid = True

    def classify_poles(self):
        poles = self.data['poles']
        residues = self.data['residues']
        n = len(poles)
        processed = [False] * n
        epsilon = 1e-9 # 适当放宽公差以适应浮点误差

        # 1. 识别实数极点 (RL电路)
        for i in range(n):
            if processed[i]:
                continue
            # 极点虚部极小 -> 实极点
            if abs(poles[i].imag) < epsilon:
                # 生成 ID: a, b, c...
                self.output_data['rl_pairs'].append({
                    'id': chr(97 + len(self.output_data['rl_pairs'])), 
                    'pole': poles[i],
                    'residue': residues[i]
                })
                processed[i] = True

        # 2. 识别共轭复数极点对 (RLC电路)
        for i in range(n):
            if processed[i]:
                continue
            
            # 寻找共轭配对
            found_pair = False
            for j in range(i + 1, n):
                if not processed[j]:
                    # 判定条件：实部接近，虚部互为相反数
                    if (abs(poles[i].real - poles[j].real) < epsilon and 
                        abs(poles[i].imag + poles[j].imag) < epsilon):
                        
                        self.output_data['rlc_pairs'].append({
                            'id': len(self.output_data['rlc_pairs']) + 1,
                            'poles': [poles[i], poles[j]],
                            'residues': [residues[i], residues[j]]
                        })
                        processed[i] = True
                        processed[j] = True
                        found_pair = True
                        break
            
            if not found_pair:
                # 如果找不到配对，可能是孤立的复数极点，这在物理系统中不应出现（除非根据实数处理）
                print(f"警告: 发现未配对的复数极点: {poles[i]}")

    def calculate_parameters(self):
        # 1. 计算 RC 并联支路 (由 offset 和 slope 决定)
        d = self.output_data['offset']
        h = self.output_data['slope']
        
        if abs(d) > 1e-20:
            # 假设模型为 Y(s) = ... + d + s*h
            # d 对应电导 G = 1/R, h 对应电容 C
            R_val = 1.0 / d
            C_val = h
            
            # 物理性检查
            if R_val < 0 or C_val < 0:
                print(f"警告: RC并联支路包含非物理(负值)参数: R={R_val:.2e}, C={C_val:.2e}")

            self.output_data['rc_params'] = {
                'R': R_val,
                'C': C_val
            }
            self.rc_num = 1
        else:
            self.output_data['rc_params'] = None
            self.rc_num = 0

        # 2. 计算 RL 串联支路
        self.output_data['rl_params'] = []
        for item in self.output_data['rl_pairs']:
            p = item['pole'].real
            r = item['residue'].real # 实极点的留数应为实数
            
            if abs(r) < 1e-15:
                continue
                
            # 导纳形式 Y(s) = 1/(L*s + R) = (1/L) / (s + R/L)
            # 留数 r = 1/L -> L = 1/r
            # 极点 p = -R/L -> R = -p * L = -p/r
            L = 1.0 / r
            R = -p / r
            
            # 物理性检查
            if L < 0 or R < 0:
                 print(f"警告: RL支路 {item['id']} 包含非物理(负值)参数: L={L:.2e}, R={R:.2e}")

            self.output_data['rl_params'].append({
                'id': item['id'],
                'L': L,
                'R': R
            })
        self.rl_num = len(self.output_data['rl_params'])

        # 3. 计算 RLC 支路
        self.output_data['rlc_params'] = []
        for item in self.output_data['rlc_pairs']:
            # 取其中一个极点和留数（通常取虚部为正的，或直接取第一个）
            p = item['poles'][0]
            k = item['residues'][0] # k为留数
            
            # 使得留数 k = k' + j*k''
            # 极点 p = -alpha + j*beta
            
            # 标准二阶项对应物理参数计算
            # 公式依据具体的等效电路拓扑在此处实现
            # 使用原代码逻辑：
            p_real, p_imag = p.real, p.imag
            c_real, c_imag = k.real, k.imag
            denom = p_real**2 + p_imag**2
            
            try:
                L = 1.0 / (2 * c_real)
                R = -p_real / c_real
                C_val = (2 * c_real) / denom
                b = -2 * (c_real * p_real + c_imag * p_imag)
                g_m = b / denom
                
                # 物理性检查 (受控源 g_m 和 b 可以为负，但无源元件 R, L, C 应为正)
                if L < 0 or R < 0 or C_val < 0:
                    print(f"警告: RLC支路 {item['id']} 包含非物理(负值)无源元件: L={L:.2e}, R={R:.2e}, C={C_val:.2e}")

                self.output_data['rlc_params'].append({
                    'id': item['id'],
                    'L': L,
                    'R': R,
                    'C': C_val,
                    'b': b,
                    'g_m': g_m
                })
            except ZeroDivisionError:
                print(f"警告: RLC电路 {item['id']} 参数计算出现除零错误。")
                
        self.rlc_num = len(self.output_data['rlc_params'])

    def print_report(self):
        """打印美观的分析报告"""
        if not self.valid:
            print("目前没有有效的分析数据。")
            return

        print("\n" + "="*60)
        print(f"{'VECTOR FITTING & EQUIVALENT CIRCUIT ANALYSIS REPORT':^60}")
        print("="*60)
        
        # 摘要
        print(f"\n[电路结构摘要]")
        print(f"{'  类型':<15} | {'数量':<10}")
        print("-" * 30)
        print(f"{'  RC 并联支路':<15} | {self.rc_num:<10}")
        print(f"{'  RL 串联支路':<15} | {self.rl_num:<10}")
        print(f"{'  RLC 单元':<15} | {self.rlc_num:<10}")

        # 1. RC 并联参数
        if self.output_data['rc_params']:
            print("\n" + "-"*60)
            print("[RC 并联支路 (Parallel RC Branch)]")
            rc = self.output_data['rc_params']
            print(f"  R = {rc['R']:.6e} Ω")
            print(f"  C = {rc['C']:.6e} F (Slope)")
            print(f"  (Offset G = {self.output_data['offset']:.6e} S)")

        # 2. RL 串联参数
        if self.output_data['rl_params']:
            print("\n" + "-"*60)
            print("[RL 串联支路 (Series RL Branches)]")
            print(f"{'  ID':<5} | {'L (H)':<15} | {'R (Ω)':<15}")
            print("-" * 40)
            for p in self.output_data['rl_params']:
                print(f"  {p['id'].upper():<5} | {p['L']:<15.4e} | {p['R']:<15.4e}")

        # 3. RLC 参数
        if self.output_data['rlc_params']:
            print("\n" + "-"*60)
            print("[RLC 受控源宽频等效支路 (RLC Branches)]")
            header = f"{'  ID':<4} | {'R (Ω)':<11} | {'L (H)':<11} | {'C (F)':<11} | {'b':<11} | {'g_m (S)':<11}"
            print(header)
            print("-" * len(header))
            for p in self.output_data['rlc_params']:
                print(f"  {p['id']:<4} | {p['R']:<11.4e} | {p['L']:<11.4e} | {p['C']:<11.4e} | {p['b']:<11.4e} | {p['g_m']:<11.4e}")
        
        print("="*60 + "\n")

def plot_fitting_result(s, f_orig, f_fitted, metrics=None):
    """
    绘制原始数据与拟合数据的幅频和相频响应对比图。
    """
    magnitude_f = np.abs(f_orig)
    magnitude_fitted = np.abs(f_fitted)
    phase_f = np.degrees(np.angle(f_orig))
    phase_fitted = np.degrees(np.unwrap(np.angle(f_fitted)))

    w = np.imag(s)
    freq_hz = w / (2 * np.pi)

    title_suffix = ""
    if metrics:
        title_suffix = f" (RMS Error: {metrics['rms_rel']:.2e})"

    plt.figure(figsize=(10, 8))
    
    # 幅频响应
    plt.subplot(2, 1, 1)
    plt.semilogx(freq_hz, 20 * np.log10(magnitude_f), label='Original', color='b', linewidth=1.5, alpha=0.8)
    plt.semilogx(freq_hz, 20 * np.log10(magnitude_fitted), '--', label='Fitted', color='r', linewidth=1.5, alpha=0.8)
    plt.title(f'Magnitude Response{title_suffix}')
    plt.ylabel('Magnitude [dB]')
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.legend()

    # 相频响应
    plt.subplot(2, 1, 2)
    plt.semilogx(freq_hz, phase_f, label='Original', color='b', linewidth=1.5, alpha=0.8)
    plt.semilogx(freq_hz, phase_fitted, '--', label='Fitted', color='r', linewidth=1.5, alpha=0.8)
    plt.title('Phase Response')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Phase [degrees]')
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    # 自动调整相频响应的 y 轴范围，避免过大空白
    plt.autoscale(axis='y', tight=True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def run_pipeline_case2():
    """
    测试案例2: Escalar measured response of a transformer
    """
    print("\n运行测试案例 2: Transformer Data Analysis")
    csv_path = r".\TRANSF_DATA.csv"
    
    try:
        Mdata = pd.read_csv(csv_path).to_numpy()
        # 根据原始代码逻辑提取数据
        # 取前160行，第0列幅度，第1列角度
        test_f = Mdata[:160, 0] * np.exp(1j * np.deg2rad(Mdata[:160, 1]))
        w = 2 * np.pi * np.linspace(0, 10e6, 401)[1:161]
        test_s = 1j * w
    except Exception as e:
        print(f"数据加载失败: {e}")
        return

    # 矢量拟合 (使用自动寻优)
    # poles, residues, d, h = vectfit_auto_rescale(test_f, test_s, verbose=False)
    poles, residues, d, h, metrics = vectfit_find_best_order(
        test_f, 
        test_s, 
        min_poles=2, 
        max_poles=32, 
        step=2, 
        target_error=5e-4 # 设定一个期望的精度目标，满足即停
    )
    
    # 生成拟合曲线用于绘图
    fitted = model(test_s, poles, residues, d, h)

    # 打印最终误差
    print(f"最终拟合误差 (RMS Relative): {metrics['rms_rel']:.6e}")
    print(f"最终拟合误差 (Max Relative): {metrics['max_rel']:.6e}")

    # 系统参数分析与电路生成
    print("正在进行等效电路参数计算...")
    analyzer = SystemAnalyzer()
    analyzer.load_fitting_result(poles, residues, d, h)
    analyzer.print_report()

    # 绘图
    plot_fitting_result(test_s, test_f, fitted, metrics=metrics)

if __name__ == '__main__':
    # 可以在此处切换不同的测试案例函数
    run_pipeline_case2()
    
    # 其他测试案例（如代码中保留的注释）可以按照类似 run_pipeline_case2 的方式封装


    # 测试案例1 18th order frequency response F(s) of one dimension
    # test_s = 1j*np.linspace(1, 1e5, 800)
    #
    # test_poles = [
    #     -4500,
    #     -41000,
    #     -100+5000j, -100-5000j,
    #     -120+15000j, -120-15000j,
    #     -3000+35000j, -3000-35000j,
    #     -200+45000j, -200-45000j,
    #     -1500+45000j, -1500-45000j,
    #     -500+70000j, -500-70000j,
    #     -1000+73000j, -1000-73000j,
    #     -2000+90000j, -2000-90000j,
    # ]
    # test_residues = [
    #     -3000,
    #     -83000,
    #     -5+7000j, -5-7000j,
    #     -20+18000j, -20-18000j,
    #     6000+45000j, 6000-45000j,
    #     40+60000j, 40-60000j,
    #     90+10000j, 90-10000j,
    #     50000+80000j, 50000-80000j,
    #     1000+45000j, 1000-45000j,
    #     -5000+92000j, -5000-92000j
    # ]
    # test_d = .2
    # test_h = 2e-5
    #
    # test_f = sum(c/(test_s - a) for c, a in zip(test_residues, test_poles))
    # test_f += test_d + test_h*test_s

    # # 定义CSV文件路径
    # csv_path = r".\frequency_response_data.csv"
    # Mdata = pd.read_csv(csv_path)
    # # 将DataFrame转换为numpy数组以便后续处理
    # Mdata = Mdata.to_numpy()
    # # 计算频率响应样本
    # # 直接使用Measured_Magnitude_Linear
    # magnitude_linear = Mdata[:134, 1]  # Measured_Magnitude_Linear 是第二列
    # # 将相位从度转换为弧度
    # phase_radians = Mdata[:134, 2] * np.pi / 180  # Measured_Phase_Deg 转换为弧度
    # # 计算复数频率响应 f(s)
    # test_f = magnitude_linear * np.exp(1j * phase_radians)
    # # 提取频率数据：Frequency_Hz
    # test_s = Mdata[:134, 0]  # 第一列是Frequency_Hz
    # # 将频率转换为复数域s=jω
    # test_s = 1j * 2 * np.pi * test_s

    # # 定义CSV文件路径
    # csv_path = r".\impedance_pu.csv"
    # Mdata = pd.read_csv(csv_path)
    # # 将DataFrame转换为numpy数组以便后续处理
    # Mdata = Mdata.to_numpy()
    # # 计算频率响应样本
    # # 直接使用Measured_Magnitude_Linear
    # magnitude_linear = Mdata[:,1]  # Measured_Magnitude_Linear 是第二列
    # # 将相位从度转换为弧度
    # phase_degrees = Mdata[:, 2]
    # # 计算复数频率响应 f(s)
    # test_f = magnitude_linear * np.exp(1j * phase_degrees * np.pi / 180)
    # # 提取频率数据：Frequency_Hz
    # test_s = Mdata[:, 0]  # 第一列是Frequency_Hz
    # # 将频率转换为复数域s=jω
    # test_s = 1j * 2 * np.pi * test_s


    # #测试案例2 Escalar measured response of a transformer
    # csv_path = r".\TRANSF_DATA.csv"
    # # 使用pandas读取CSV文件到DataFrame
    # Mdata = pd.read_csv(csv_path)
    # # 将DataFrame转换为numpy数组以便后续处理
    # Mdata = Mdata.to_numpy()
    # # 计算f(s)的样本：取前160行数据，第0列为幅度，第1列为角度（转换为弧度）
    # test_f = Mdata[:160, 0] * np.exp(1j * Mdata[:160, 1] * pi / 180)
    # w = 2 * pi * np.linspace(0, 10e6, 401)[1:161]
    # # 将频率转换为复数域s=jω
    # test_s = 1j * w

    # #测试案例3 Elementwise approximation of a 6x6 admitance matrix
    # csv_path = r".\SYSADMITANCE_DATA.csv"  # local path! Update with your own path
    # Mdata = pd.read_csv(csv_path)  # Measured data
    # Mdata = np.ravel(Mdata.to_numpy())  # Data transfered into a numpy array
    # N = int(Mdata[0])  # Number of frequency samples
    # Ysys = np.zeros((6, 6, N), dtype=np.complex128)  # Samples of matrix Y(s) imported from the file
    # s = np.zeros(N, dtype=np.complex128)  # Complex frequency points of evaluation for Y(s)
    #
    # # Y(s) and s are compressed in row-major order into Mdata array. Samples are organized in blocks of 73 numbers, inside each block the first
    # # number matchs the complex frequency samples and the remaining ones corresponds to each element in the admitance matrix Y(s). first comes
    # # the real part and then the imaginary part.
    # k = 0  # sample index
    # for i in range(1, Mdata.size, 73):
    #     s[k] = 1j * Mdata[i]  # complex frequency sample
    #     for row in range(6):
    #         ind = row * 12 + i
    #         Ysys[row, :, k] = Mdata[ind + 1:ind + 12:2] + 1j * Mdata[ind + 2:ind + 13:2]  # entire row is append to Y(s)
    #     k += 1
    # # Stacking Y(s) data as elements of a frequency domain function F(s).
    # # Due to Y(s) is symmetric only the members into the lower trinagular submatrix Y(s) are taken
    # F = np.zeros((21, N), dtype=np.complex128)
    # k = 0  # element index
    # for row in range(6):
    #     for col in range(row, 6):
    #         F[k, :] = Ysys[row, col, :]  # all samples are in z axis
    #         k += 1
    #
    # print("\nFrequency domain samples of F(s) = \n", F)
    # print("f(s) shape = ", F.shape, "\ndata type in f(s) = ", type(F), type(F[0, 0]))
    #
    # weights = 1 / np.sqrt(np.abs(F))  # Weighting with inverse of the square root of the magnitude of F(s)
    # n = 50  # Order of approximation
    # w = s.imag  # Angular frequency samples in rad/s
    # # Starting poles generation:
    # Bet = np.linspace(w[0], w[N - 1], int(n / 2))
    # poles = np.zeros(n, dtype=np.complex128)
    # # setting poles as complex conjugated pairs
    # for k in range(int(n / 2)):
    #     alf = -Bet[k] / 100
    #     poles[2 * k] = alf - 1j * Bet[k]
    #     poles[2 * k + 1] = alf + 1j * Bet[k]
    #
    # print(
    #     "A better set of initial poles are obtained by fitting the weighted column sum of the first column of Y(s)\n * Applying 5 iterations of vector fitting...")
    # # These weights are common for all column elements
    # g = np.zeros(N, dtype=np.complex128)
    # for k in range(6):
    #     g = g + F[k, :] / np.linalg.norm(F[k, :])
    # weights_g = 1 / np.abs(g)
    # test_f = g
    # test_s = s

    # #测试案例4 Elementwise approximation of a 3x3 propagation matrix of an aerial transmission line.\nIt corresponds to single propagation mode and time delay is already extracted
    # # Importing data from a .csv file (lineConstants_H0.csv)
    # csv_path = r".\MODEH_DATA.csv"  # local path! Update with your own path
    # # Pandas data frame with H(w) samples in the frequency domain:
    # # Columns organization: index , OMEGA(Ang. frequency), H_00REAL(1st element's real part), H_00IMAG(1st element's imaginary part), ... H_01REAL(2nd element's real part), ...
    # Hdata = pd.read_csv(csv_path)
    # w = np.ravel(Hdata.loc[:, "OMEGA"].to_numpy())  # Angular frequency samples
    # s = 1j * w  # Complex frequency samples
    # N = w.size  # Number of samples
    # Hw = np.zeros((3, 3, N), dtype=np.complex128)  # Propagation matrix in the frequency domain
    # # Copying data into H:
    # k = 2
    # for row in range(3):
    #     for col in range(3):
    #         Hw[row, col, :] = np.ravel(Hdata.iloc[:, k].to_numpy()) + 1j * np.ravel(
    #             Hdata.iloc[:, k + 1].to_numpy())  # elements are read in RMO
    #         k += 2
    #
    # # Stacking H(s) data as elements of a frequency domain function F(s).
    # # Due to H(s) is asymmetric, F(s) is a flattened version of H(s). Row Major Ordering is used to map H(s) elements into F(s)
    # F = np.zeros((3 * 3, N), dtype=np.complex128)
    # k = 0  # element index
    # for row in range(3):
    #     for col in range(3):
    #         F[k, :] = Hw[row, col, :]  # all frequency samples are in z axis
    #         k += 1
    #
    # print("\nFrequency domain samples of F(s) = \n", F)
    # print("f(s) shape = ", F.shape, "\ndata type in f(s) = ", type(F), type(F[0, 0]))
    #
    # weights = np.ones(N, dtype=np.float64)  # No samples weighting
    # n = 35  # Order of approximation
    # # Starting poles generation:
    # Bet = np.logspace(np.log10(w[0]), np.log10(w[N - 1]), int(n / 2))
    # poles = np.zeros(n, dtype=np.complex128)
    # # setting poles as complex conjugated pairs
    # for k in range(int(n / 2)):
    #     alf = -Bet[k] / 100
    #     poles[2 * k] = alf - 1j * Bet[k]
    #     poles[2 * k + 1] = alf + 1j * Bet[k]
    #
    # print("A set of initial poles are obtained by fitting trace of Hi\n * Applying 10 iterations of vector fitting...")
    # # Using H trace to identify initial poles:
    # trH = np.zeros(N, dtype=np.complex128)
    # for k in range(3):
    #     trH = trH + Hw[k, k, :]
    # test_f = trH
    # test_s = s