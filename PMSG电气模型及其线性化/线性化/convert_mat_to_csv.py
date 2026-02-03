import scipy.io
import numpy as np
import pandas as pd
import os
import glob
from scipy import signal

# ================= 配置部分 =================
# 输入文件夹：MATLAB 生成的 mat 文件所在路径
# INPUT_DIR = r"your root"  # 请修改为您的实际路径，例如 "e:\\ruanjian\\..."
# 替换为实际的路径，使用原始字符串（r前缀）避免转义字符问题
INPUT_DIR = r"E:\ruanjian\GitHubDesktop\Vector-fitting-and-equivalent-circuit-transformation\your root"
# 输出文件夹：生成的 CSV 存放路径
OUTPUT_DIR = os.path.join(INPUT_DIR, "csv_data")

# 频率扫描范围 (Hz)
F_START = 0.1
F_END = 5000.0  # 根据风机带宽调整，例如 5kHz
NUM_POINTS = 500 # 频率点数量

# 关注的输入输出通道索引 (0-based)
# 输入 u_ss = [v_alpha, v_beta, omega_best, i_gqref]
# 我们通常只关心 v_alpha(0), v_beta(1) 对输出的影响
INPUT_IDXS = [0, 1] 
# 输出假设为 [i_alpha, i_beta] (2维)
OUTPUT_IDXS = [0, 1]
# ===========================================

def process_mat_files():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"创建输出目录: {OUTPUT_DIR}")

    # 查找所有 .mat 文件
    mat_files = glob.glob(os.path.join(INPUT_DIR, "*.mat"))
    print(f"找到 {len(mat_files)} 个 .mat 文件。")

    # 生成对数分布的频率向量
    freqs_hz = np.logspace(np.log10(F_START), np.log10(F_END), NUM_POINTS)
    w_rad = 2 * np.pi * freqs_hz  # 转换为角频率 rad/s

    for mat_path in mat_files:
        try:
            # 1. 加载 MAT 文件
            data = scipy.io.loadmat(mat_path)
            
            # 提取 ABCD 矩阵
            # 注意：scipy.io 加载的可能是 numpy 矩阵或数组，需稍作处理
            if 'sys' in data:
                # 如果存的是 sys 对象结构体，可能需要解析 struct
                # 但您代码里也分别存了 A, B, C, D，直接用单独的变量更方便
                A = data['A']
                B = data['B']
                C = data['C']
                D = data['D']
            else:
                # 兼容性处理
                A = data['A']
                B = data['B']
                C = data['C']
                D = data['D']

            # 2. 构建状态空间系统
            # scipy.signal.StateSpace(A, B, C, D)
            sys = signal.StateSpace(A, B, C, D)

            # 3. 计算频率响应
            # signal.freqresp 计算 H(jw) = C(jwI - A)^-1 B + D
            # w: 频率点
            # returns: w, h (complex frequency result)
            # 这里的 h 形状通常是 (outputs, inputs, freq_points) 或 (freq_points)
            w_out, h_out = signal.freqresp(sys, w=w_rad)

            # h_out shape: (n_outputs, n_inputs, n_freqs)
            
            # 4. 提取数据并保存为 DataFrame
            # 我们需要保存 Y11, Y12, Y21, Y22 (即 alpha-alpha, alpha-beta ...)
            # 对应 h_out[out_idx, in_idx, :]
            
            df_dict = {
                'Frequency_Hz': freqs_hz,
                'Omega_rad': w_rad
            }

            # 遍历我们关心的通道
            # Y11: Input 0 -> Output 0
            # Y12: Input 1 -> Output 0
            # Y21: Input 0 -> Output 1
            # Y22: Input 1 -> Output 1
            
            for out_i in OUTPUT_IDXS:
                for in_j in INPUT_IDXS:
                    resp_complex = h_out[out_i, in_j, :]
                    
                    label = f'Y{out_i+1}{in_j+1}' # 例如 Y11, Y12
                    
                    # 保存实部和虚部，这对矢量拟合最精确
                    df_dict[f'{label}_Real'] = np.real(resp_complex)
                    df_dict[f'{label}_Imag'] = np.imag(resp_complex)
                    
                    # 顺便保存幅值和相位（VF.py 示例里用了这个，方便肉眼看）
                    df_dict[f'{label}_Mag'] = np.abs(resp_complex)
                    df_dict[f'{label}_Phase_Deg'] = np.degrees(np.angle(resp_complex))

            df = pd.DataFrame(df_dict)

            # 保存 CSV
            base_name = os.path.basename(mat_path).replace('.mat', '.csv')
            csv_path = os.path.join(OUTPUT_DIR, base_name)
            df.to_csv(csv_path, index=False)
            
            print(f"已处理: {base_name}")

        except Exception as e:
            print(f"处理文件 {os.path.basename(mat_path)} 时出错: {e}")

if __name__ == "__main__":
    process_mat_files()
