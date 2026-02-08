
import os
import pandas as pd
import numpy as np
import VF

# --- 配置 ---
INPUT_FILE = os.path.join("your_root", "csv_data", "iP01_iV01_iQ01_iX01__P-300m_Q-1000m_V+900m_xi-10000md.csv")
TARGET_ELEMENT = "Y11"

def print_separator(title):
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def main():
    print(f"Processing single file: {INPUT_FILE}")
    
    if not os.path.exists(INPUT_FILE):
        print("Error: File not found.")
        return

    try:
        df = pd.read_csv(INPUT_FILE)
        
        # 1. 准备数据
        if 'Frequency_Hz' not in df.columns:
            print("Error: Missing 'Frequency_Hz' column")
            return
            
        freq_hz = df['Frequency_Hz'].values
        s_vec = 1j * 2 * np.pi * freq_hz
        
        real_col = f"{TARGET_ELEMENT}_Real"
        imag_col = f"{TARGET_ELEMENT}_Imag"
        
        if real_col not in df.columns or imag_col not in df.columns:
            print(f"Error: Columns for {TARGET_ELEMENT} not found.")
            return

        f_vec = df[real_col].values + 1j * df[imag_col].values
        
        print_separator(f"Step 1: Vector Fitting for {TARGET_ELEMENT}")
        print("Target Order: 6 (3 pairs of poles)")
        
        # 2. 执行矢量拟合
        # 固定6阶(6个极点)，需设置 min/max_poles=3 (因为代码中 n_poles 是对数)
        poles, residues, d, h, metrics = VF.vectfit_find_best_order(
            f_vec, s_vec, 
            min_poles=3, max_poles=3, step=1, 
            target_error=1e-5, 
            weighting_policy='none',
            silent=False 
        )

        # 3. 打印有理函数表达式
        print_separator("Step 2: Rational Function Model")
        print("Model format: F(s) = Sum(r_k / (s - p_k)) + d + s*h")
        print("\n[Identified Parameters]")
        print(f"Propportional term (h): {h:.6e}")
        print(f"Constant term      (d): {d:.6e}")
        print(f"\nPoles (p_k) and Residues (r_k):")
        print(f"{'Pole (Real + Imag*j)':<35} | {'Residue (Real + Imag*j)':<35}")
        print("-" * 75)
        for p, r in zip(poles, residues):
            p_str = f"{p.real:.4e} {p.imag:+.4e}j"
            r_str = f"{r.real:.4e} {r.imag:+.4e}j"
            print(f"{p_str:<35} | {r_str:<35}")

        print(f"\n[Fitting Quality]")
        print(f"RMS Relative Error: {metrics['rms_rel']*100:.4f} %")
        print(f"Max Relative Error: {metrics['max_rel']*100:.4f} %")

        # 4. 检查无源性
        print_separator("Step 3: Passivity Check")
        is_passive, min_real, viol_freq = VF.check_passivity(s_vec, poles, residues, d, h)
        if is_passive:
            print("Status: PASSIVE (Physical validity confirmed)")
        else:
            print("Status: NOT PASSIVE")
            print(f"Min Real Part: {min_real:.4e}")

        # 5. 等效电路参数提取
        print_separator("Step 4: Equivalent Circuit Synthesis")
        analyzer = VF.SystemAnalyzer()
        analyzer.load_fitting_result(poles, residues, d, h)
        
        print("Detected Circuit Branches:")
        
        # 并联 RC
        if analyzer.output_data['rc_params']:
            p = analyzer.output_data['rc_params']
            print("\n[Parallel RC Branch] (from d and h)")
            print(f"  R_p: {p['R']:.4f} Ohm (Conductance G = {1/p['R']:.4e} S)")
            print(f"  C_p: {p['C']:.4e} F")
        
        # 串联 RL
        if analyzer.output_data['rl_params']:
            print("\n[Series RL Branches] (Real poles)")
            for i, item in enumerate(analyzer.output_data['rl_params']):
                print(f"  Branch {i+1}:")
                print(f"    R: {item['R']:.4f} Ohm")
                print(f"    L: {item['L']:.4e} H")

        # 串联 RLC
        if analyzer.output_data['rlc_params']:
            print("\n[Series RLC Branches] (Complex pole pairs)")
            for i, item in enumerate(analyzer.output_data['rlc_params']):
                print(f"  Branch {i+1}:")
                print(f"    R: {item['R']:.4f} Ohm")
                print(f"    L: {item['L']:.4e} H")
                print(f"    C: {item['C']:.4e} F")
                # 如果有 b, g_m (受控源参数), 也可以展示
                if item.get('b', 0) != 0 or item.get('g_m', 0) != 0:
                   print(f"    (Active parts: b={item.get('b',0):.4e}, g_m={item.get('g_m',0):.4e})")

        print("\nProcessing complete.")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
