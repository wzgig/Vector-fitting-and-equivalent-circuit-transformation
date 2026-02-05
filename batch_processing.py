import os
import glob
import re
import pandas as pd
import numpy as np
import VF
import concurrent.futures
import time

# --- 配置 ---
INPUT_DIR = os.path.join("your_root", "csv_data")
OUTPUT_FILE = "equivalent_circuit_parameters_optimized.csv"
# 目标元素
ELEMENTS = ["Y11", "Y12", "Y21", "Y22"]

# 正则匹配模式
FILENAME_PATTERN = re.compile(
    r"iP(\d+)_iV(\d+)_iQ(\d+)_iX(\d+)__P([+-]?\d+m)_Q([+-]?\d+m)_V([+-]?\d+m)_xi([+-]?\d+md)"
)

def parse_milli_token(tok: str) -> float:
    """将 '-300m' 解析为 -0.3"""
    tok = tok.strip()
    if tok.endswith("md"):
        tok = tok[:-2]
    elif tok.endswith("m"):
        tok = tok[:-1]
    return float(tok) / 1000.0

def process_single_file(fpath):
    """
    单个文件处理函数，用于并行调用
    返回: (成功列表, 失败信息)
    """
    basename = os.path.basename(fpath)
    file_results = []
    
    # 1. 解析文件名元数据
    match = FILENAME_PATTERN.search(basename)
    conditions = {}
    if match:
        conditions = {
            'iP': match.group(1),
            'iV': match.group(2),
            'iQ': match.group(3),
            'iX': match.group(4),
            'P':  parse_milli_token(match.group(5)),
            'Q':  parse_milli_token(match.group(6)),
            'V':  parse_milli_token(match.group(7)),
            'xi': parse_milli_token(match.group(8)),
        }
    else:
        conditions = {k: np.nan for k in ['iP', 'iV', 'iQ', 'iX', 'P', 'Q', 'V', 'xi']}

    try:
        df = pd.read_csv(fpath)
        
        # 频率向量 (复数s)
        if 'Frequency_Hz' not in df.columns:
            return [], f"{basename}: Missing 'Frequency_Hz' column"
            
        freq_hz = df['Frequency_Hz'].values
        s_vec = 1j * 2 * np.pi * freq_hz
        
        for elem in ELEMENTS:
            real_col = f"{elem}_Real"
            imag_col = f"{elem}_Imag"
            
            if real_col not in df.columns or imag_col not in df.columns:
                continue
            
            # 2. 构建复数响应
            f_vec = df[real_col].values + 1j * df[imag_col].values
            
            # [优化重构]
            # 现在我们只需通知 VF.py 使用 "反向幅值加权" 策略即可
            # 无需在业务代码中手动计算 weights 数组，保持了业务逻辑的整洁和算法的封装性
            
            # 4. 执行矢量拟合
            poles, residues, d, h, metrics = VF.vectfit_find_best_order(
                f_vec, s_vec, 
                min_poles=2, max_poles=40, step=2, 
                target_error=1e-5, 
                weighting_policy='none', # <--- 优雅的接口调用
                silent=True
            )
            
            # [优化] 5. 检查无源性 (Passivity Check)
            is_passive, min_real, viol_freq = VF.check_passivity(s_vec, poles, residues, d, h)

            # 6. 系统参数提取
            analyzer = VF.SystemAnalyzer()
            analyzer.load_fitting_result(poles, residues, d, h)
            
            # 基础元数据 (对每一行都重复，方便后续 Pandas 分析)
            base_info = {
                'Filename': basename,
                **conditions,
                'Element': elem,
                'RMS_Rel_Error': metrics['rms_rel'],
                'Max_Rel_Error': metrics['max_rel'],
                'Order': len(poles),
                'Is_Passive': is_passive,       # 新增指标
                'Min_Real_Part': min_real       # 新增指标
            }

            # 收集电路参数
            # 并联 RC
            if analyzer.output_data['rc_params']:
                row = base_info.copy()
                row.update({
                    'Branch_Type': 'RC_Parallel',
                    'Branch_ID': 'Parallel',
                    'R': analyzer.output_data['rc_params']['R'],
                    'C': analyzer.output_data['rc_params']['C']
                })
                file_results.append(row)
            
            # 串联 RL
            if analyzer.output_data['rl_params']:
                for item in analyzer.output_data['rl_params']:
                    row = base_info.copy()
                    row.update({
                        'Branch_Type': 'RL_Series',
                        'Branch_ID': item['id'],
                        'R': item['R'],
                        'L': item['L']
                    })
                    file_results.append(row)

            # 串联 RLC
            if analyzer.output_data['rlc_params']:
                for item in analyzer.output_data['rlc_params']:
                    row = base_info.copy()
                    row.update({
                        'Branch_Type': 'RLC_Series',
                        'Branch_ID': item['id'],
                        'R': item['R'],
                        'L': item['L'],
                        'C': item['C']
                    })
                    file_results.append(row)
                    
        return file_results, None

    except Exception as e:
        return [], f"{basename}: {str(e)}"

def run_batch():
    # 查找所有 CSV
    csv_files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))
    total_files = len(csv_files)
    print(f"找到 {total_files} 个 CSV 文件，准备开始并行处理...")

    all_data = []
    errors = []
    
    start_time = time.time()

    # 使用多进程可以避开 Python GIL，利用多核 CPU
    # max_workers 默认设为 None (CPU核心数)，可根据内存情况调整
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # 提交任务
        futures = {executor.submit(process_single_file, f): f for f in csv_files}
        
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            res, err = future.result()
            if res:
                all_data.extend(res)
            if err:
                errors.append(err)
            
            # 进度打印
            if (i + 1) % 10 == 0:
                print(f"进度: {i + 1}/{total_files} ({(i + 1)/total_files*100:.1f}%)")

    end_time = time.time()
    print(f"\n处理完成，耗时: {end_time - start_time:.2f} 秒")

    if errors:
        print(f"\n出现 {len(errors)} 个错误:")
        for e in errors[:5]: # 只打印前5个错误
            print(f"  - {e}")
        if len(errors) > 5:
            print("  ... (更多错误见日志)")

    # 保存结果
    if all_data:
        print("\n保存结果文件...")
        full_df = pd.DataFrame(all_data)
        
        # 定义列顺序
        cols_order = ['Filename', 'iP', 'iV', 'iQ', 'iX', 'P', 'Q', 'V', 'xi', 
                      'Element', 'Is_Passive', 'Min_Real_Part', # 放在显眼位置
                      'Branch_Type', 'Branch_ID', 
                      'R', 'L', 'C', 
                      'RMS_Rel_Error', 'Max_Rel_Error', 'Order']
        
        # 确保所有列存在（填补 NaN）
        for c in cols_order:
            if c not in full_df.columns:
                full_df[c] = None
                
        # 分文件保存 (可选：按 Element 拆分)
        # 这里演示保存为一个完整大表，方便筛选
        # 如果文件过大，也可以按原来的逻辑 split
        for elem in ELEMENTS:
            elem_df = full_df[full_df['Element'] == elem].copy()
            if not elem_df.empty:
                # 重新排序列
                elem_df = elem_df[cols_order]
                fname = OUTPUT_FILE.replace(".csv", f"_{elem}.csv")
                elem_df.to_csv(fname, index=False)
                print(f"  -> {fname} ({len(elem_df)} rows)")
    else:
        print("未生成任何有效数据。")

if __name__ == "__main__":
    # Windows 下使用 multiprocess 必须放在 if __name__ == "__main__": 下
    run_batch()