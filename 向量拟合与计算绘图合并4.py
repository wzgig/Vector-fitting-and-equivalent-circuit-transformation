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
from pylab import *
from numpy.linalg import eigvals, lstsq
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple

def parse_complex(s):
    """增强型复数解析函数"""
    s = re.sub(r'\s+', '', s)
def cc(z):
    return z.conjugate()

def model(s, poles, residues, d, h):
    return sum(r/(s-p) for p, r in zip(poles, residues)) + d + s*h

def vectfit_step(f, s, poles):
    """
    f = complex data to fit
    s = j*frequency
    poles = initial poles guess
        note: All complex poles must come in sequential complex conjugate pairs
    returns adjusted poles
    """
    N = len(poles)
    Ns = len(s)

    cindex = zeros(N)
    # cindex is:
    #   - 0 for real poles
    #   - 1 for the first of a complex-conjugate pair
    #   - 2 for the second of a cc pair
    for i, p in enumerate(poles):
        if p.imag != 0:
            if i == 0 or cindex[i-1] != 1:
                assert cc(poles[i]) == poles[i+1], ("Complex poles must come in conjugate pairs: %s, %s" % (poles[i], poles[i+1]))
                cindex[i] = 1
            else:
                cindex[i] = 2

    # First linear equation to solve. See Appendix A
    A = zeros((Ns, 2*N+2), dtype=np.complex64)
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
    A = vstack((real(A), imag(A)))
    b = concatenate((real(b), imag(b)))
    x, residuals, rnk, s = lstsq(A, b, rcond=-1)

    residues = x[:N]
    d = x[N]
    h = x[N+1]

    # We only want the "tilde" part in (A.4)
    x = x[-N:]

    # Calculation of zeros: Appendix B
    A = diag(poles)
    b = ones(N)
    c = x
    for i, (ci, p) in enumerate(zip(cindex, poles)):
        if ci == 1:
            x, y = real(p), imag(p)
            A[i, i] = A[i+1, i+1] = x
            A[i, i+1] = -y
            A[i+1, i] = y
            b[i] = 2
            b[i+1] = 0
            #cv = c[i]
            #c[i,i+1] = real(cv), imag(cv)

    H = A - outer(b, c)
    H = real(H)
    new_poles = sort(eigvals(H))
    unstable = real(new_poles) > 0
    new_poles[unstable] -= 2*real(new_poles)[unstable]
    return new_poles

# Dear gods of coding style, I sincerely apologize for the following copy/paste
def calculate_residues(f, s, poles, rcond=-1):
    Ns = len(s)
    N = len(poles)

    cindex = zeros(N)
    for i, p in enumerate(poles):
        if p.imag != 0:
            if i == 0 or cindex[i-1] != 1:
                assert cc(poles[i]) == poles[i+1], ("Complex poles must come in conjugate pairs: %s, %s" % poles[i:i+1])
                cindex[i] = 1
            else:
                cindex[i] = 2

    # use the new poles to extract the residues
    A = zeros((Ns, N+2), dtype=np.complex128)
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
    A = vstack((real(A), imag(A)))
    b = concatenate((real(b), imag(b)))
    cA = np.linalg.cond(A)
    if cA > 1e13:
        print ('Warning!: Ill Conditioned Matrix. Consider scaling the problem down')
        print ('Cond(A)', cA)
    x, residuals, rnk, s = lstsq(A, b, rcond=rcond)

    # Recover complex values
    x = np.complex64(x)
    for i, ci in enumerate(cindex):
       if ci == 1:
           r1, r2 = x[i:i+2]
           x[i] = r1 - 1j*r2
           x[i+1] = r1 + 1j*r2

    residues = x[:N]
    d = x[N].real
    h = x[N+1].real
    return residues, d, h

def print_params(poles, residues, d, h):
    cfmt = "{0.real:g} + {0.imag:g}j"
    print ("poles: " + ", ".join(cfmt.format(p) for p in poles))
    print ("residues: " + ", ".join(cfmt.format(r) for r in residues))
    print ("offset: {:g}".format(d))
    print ("slope: {:g}".format(h))

def vectfit_auto(f, s, n_poles=20, n_iter=20, show=False,
                 inc_real=False, loss_ratio=1e-2, rcond=-1, track_poles=False):
    w = imag(s)
    pole_locs = linspace(w[0], w[-1], n_poles+2)[1:-1]
    lr = loss_ratio
    init_poles = poles = concatenate([[p*(-lr + 1j), p*(-lr - 1j)] for p in pole_locs])

    if inc_real:
        poles = concatenate((poles, [1]))

    poles_list = []
    for _ in range(n_iter):
        poles = vectfit_step(f, s, poles)
        poles_list.append(poles)

    residues, d, h = calculate_residues(f, s, poles, rcond=rcond)

    if track_poles:
        return poles, residues, d, h, np.array(poles_list)

    # print_params(poles, residues, d, h)
    return poles, residues, d, h

def vectfit_auto_rescale(f, s, **kwargs):
    s_scale = abs(s[-1])
    f_scale = abs(f[-1])
    # print ('SCALED')
    poles_s, residues_s, d_s, h_s = vectfit_auto(f / f_scale, s / s_scale, **kwargs)
    poles = poles_s * s_scale
    residues = residues_s * f_scale * s_scale
    d = d_s * f_scale
    h = h_s * f_scale / s_scale
    # print ('UNSCALED')
    print_params(poles, residues, d, h)
    return poles, residues, d, h

class SystemAnalyzer:
    DEFAULT_TOLERANCE = 1e-16

    def __init__(self):
        self.data = {
            'poles': [],
            'residues': [],
            'offset': None,
            'slope': None
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

    @staticmethod
    def parse_complex(s: str) -> complex:
        s = re.sub(r'\s+', '', s)
        s = re.sub(r'(?<=\d)\+?(?=-)', '', s)
        try:
            return complex(s)
        except ValueError:
            raise ValueError(f"无效的复数格式: '{s}'")

    @staticmethod
    def parse_key_value(line: str) -> Tuple[str, str]:
        key_part, _, value_part = line.partition(':')
        return key_part.strip().lower(), value_part.strip()

    def analyze_input(self) -> List[str]:
        print("请输入系统参数（格式如下，输入空行结束）:")
        print("poles: -1+0j, 2-3j\nresidues: ...\noffset: ...\nslope: ...")
        input_lines = []
        while (line := input().strip()):
            input_lines.append(line)
        return input_lines

    def process_lines(self, input_lines: List[str]):
        for line in input_lines:
            key, value = self.parse_key_value(line)
            if key == 'poles':
                items = re.split(r',\s*(?![^()]*\))', value)
                self.data['poles'] = [self.parse_complex(item.strip()) for item in items]
            elif key == 'residues':
                items = re.split(r',\s*(?![^()]*\))', value)
                self.data['residues'] = [self.parse_complex(item.strip()) for item in items]
            elif key == 'offset':
                self.data['offset'] = float(value)
                self.output_data['offset'] = self.data['offset']
            elif key == 'slope':
                self.data['slope'] = float(value)
                self.output_data['slope'] = self.data['slope']

        if len(self.data['poles']) != len(self.data['residues']):
            raise ValueError("极点与留数数量不匹配")

    def classify_poles(self):
        processed = [False] * len(self.data['poles'])
        epsilon = self.DEFAULT_TOLERANCE

        # 处理实数极点
        for i, pole in enumerate(self.data['poles']):
            if processed[i]:
                continue
            if abs(pole.imag) < epsilon:
                self.output_data['rl_pairs'].append({
                    'id': chr(97 + len(self.output_data['rl_pairs'])),
                    'pole': self.data['poles'][i],
                    'residue': self.data['residues'][i]
                })
                processed[i] = True

        # 处理共轭对
        for i in range(len(self.data['poles'])):
            if not processed[i]:
                found = False
                for j in range(i+1, len(self.data['poles'])):
                    if not processed[j] and \
                       abs(self.data['poles'][i].real - self.data['poles'][j].real) < epsilon and \
                       abs(self.data['poles'][i].imag + self.data['poles'][j].imag) < epsilon:
                        self.output_data['rlc_pairs'].append({
                            'id': len(self.output_data['rlc_pairs']) + 1,
                            'poles': [self.data['poles'][i], self.data['poles'][j]],
                            'residues': [self.data['residues'][i], self.data['residues'][j]]
                        })
                        processed[i] = True
                        processed[j] = True
                        found = True
                        break
                if not found and not processed[i]:
                    raise ValueError(f"未配对的复数极点: {self.data['poles'][i]}")

    def calculate_parameters(self):
        # 计算RC参数
        if self.output_data['offset'] is not None and self.output_data['slope'] is not None:
            try:
                self.output_data['rc_params'] = {
                    'R': 1.0 / self.output_data['offset'],
                    'C': self.output_data['slope']
                }
                self.rc_num = 1
            except ZeroDivisionError:
                raise ValueError("Slope值不能为零")
        else:
            self.rc_num = 0

        # 计算RL参数
        for rl_pair in self.output_data['rl_pairs']:
            pole = rl_pair['pole'].real
            residue = rl_pair['residue'].real
            if abs(residue) < self.DEFAULT_TOLERANCE:
                raise ValueError(f"RL电路{rl_pair['id']}的留数接近零")
            L = 1.0 / residue
            R = -pole / residue
            self.output_data['rl_params'].append({
                'id': rl_pair['id'],
                'L': L,
                'R': R
            })
        self.rl_num = len(self.output_data['rl_pairs'])

        # 计算RLC参数
        for rlc_pair in self.output_data['rlc_pairs']:
            p = rlc_pair['poles'][0]
            c = rlc_pair['residues'][0]
            p_real = p.real
            p_imag = p.imag
            c_real = c.real
            c_imag = c.imag
            denom = p_real**2 + p_imag**2
            if denom < self.DEFAULT_TOLERANCE:
                raise ValueError(f"RLC电路{rlc_pair['id']}极点模长接近零")
            try:
                params = {
                    'id': rlc_pair['id'],
                    'L': 1.0 / (2 * c_real),
                    'R': -p_real / c_real,
                    'C': (2 * c_real) / denom,
                    'b': -2 * (c_real * p_real + c_imag * p_imag),
                    'g_m': (-2 * (c_real * p_real + c_imag * p_imag)) / denom
                }
                self.output_data['rlc_params'].append(params)
            except ZeroDivisionError as e:
                raise ValueError(f"RLC电路{rlc_pair['id']}参数计算错误: {e}")
        self.rlc_num = len(self.output_data['rlc_pairs'])

    def analyze(self, input_lines=None):
        try:
            if input_lines is None:
                input_lines = self.analyze_input()
            self.process_lines(input_lines)
            self.classify_poles()
            self.calculate_parameters()
            self.valid = True
        except Exception as e:
            print(f"参数分析错误: {str(e)}")
            self.valid = False

class CircuitDrawer:
    def __init__(self, analyzer: SystemAnalyzer):
        self.analyzer = analyzer
        # 添加编码问题修复配置
        schemdraw.config.use_unicode = False  # 禁用Unicode符号
        schemdraw.config.font = 'Arial'  # 使用标准字体

    def draw_combined_circuits(self):
        rc_num = self.analyzer.rc_num
        rl_num = self.analyzer.rl_num
        rlc_num = self.analyzer.rlc_num

        d = schemdraw.Drawing()
        d.config(
            fontsize=12,
            unit=1,
            font='Arial',  # 强制使用标准字体
            color='black'  # 确保颜色设置兼容
        )

        d = schemdraw.Drawing()
        d.config(fontsize=12, unit=1)
        vin_points = []
        vo_points = []
        x_offset = 0

        if rc_num:
            with d:
                vin = elm.Dot().at([x_offset, 0]).label('Vp', loc='top')
                d += vin
                vin_points.append(vin.start)
                d += elm.Line().down()
                d.push()
                d += elm.Line().right()
                d += elm.Line().down().length(1)
                d += elm.Resistor().down().label('R')
                d += elm.Line().down().length(1)
                d += elm.Line().left()
                d.pop()
                d += elm.Line().left()
                d += elm.Line().down().length(1)
                d += elm.Capacitor().down().label('C')
                d += elm.Line().down().length(1)
                d += elm.Line().right()
                d += elm.Line().down()
                vo = elm.Dot().label('Vn', loc='bottom')
                d += vo
                vo_points.append(vo.end)
            x_offset += 4

        for i in range(rl_num):
            suffix = chr(97 + i)
            with d:
                vin = elm.Dot().at([x_offset, 0]).label(f'Vp_{suffix}', loc='top')
                d += vin
                vin_points.append(vin.start)
                d += elm.Line().down()
                d += elm.Resistor().down().label(f'R_{suffix}')
                d += elm.Line().down()
                d += elm.Inductor().down().label(f'L_{suffix}')
                d += elm.Line().down()
                vo = elm.Dot().label(f'Vn_{suffix}', loc='bottom')
                d += vo
                vo_points.append(vo.end)
            x_offset += 4

        for i in range(rlc_num):
            suffix = i + 1
            with d:
                vin = elm.Dot().at([x_offset, 0]).label(f'Vp_{suffix}', loc='top')
                d += vin
                vin_points.append(vin.start)
                d += elm.Line().down(0.5)
                d += elm.Dot().label('', loc='left')
                d.push()
                d += elm.Line().right(2)
                d += elm.Line().down(1.5)
                d += elm.SourceControlledI().down().label(f'gmVc{suffix}', loc='bottom', rotate=-90)
                d += elm.Line().down(1.5)
                d += elm.Line().left(2)
                d.pop()
                d += elm.Line().down(0.5)
                d += elm.Inductor().down().label(f'L_{suffix}')
                d += elm.Capacitor().down().label(f'C_{suffix}').label(['+', f'Vc{suffix}', '-'], 'bottom')
                d += elm.Resistor().down().label(f'R_{suffix}')
                d += elm.Line().down(0.5)
                d += elm.Dot().label('', loc='left')
                d += elm.Line().down(0.5)
                vo = elm.Dot().label(f'Vn_{suffix}', loc='bottom')
                d += vo
                vo_points.append(vo.end)
            x_offset += 4.5

        if vin_points:
            with d:
                total_span = x_offset - 4.5
                d += elm.Line().at(vin_points[0]).right(total_span)
                d += elm.Line().at(vo_points[0]).right(total_span)

        d.draw()

if __name__ == '__main__':
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


    #测试案例2 Escalar measured response of a transformer
    csv_path = r".\TRANSF_DATA.csv"
    # 使用pandas读取CSV文件到DataFrame
    Mdata = pd.read_csv(csv_path)
    # 将DataFrame转换为numpy数组以便后续处理
    Mdata = Mdata.to_numpy()
    # 计算f(s)的样本：取前160行数据，第0列为幅度，第1列为角度（转换为弧度）
    test_f = Mdata[:160, 0] * np.exp(1j * Mdata[:160, 1] * pi / 180)
    w = 2 * pi * np.linspace(0, 10e6, 401)[1:161]
    # 将频率转换为复数域s=jω
    test_s = 1j * w

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


    vectfit_auto(test_f, test_s)

    poles, residues, d, h = vectfit_auto_rescale(test_f, test_s)
    fitted = model(test_s, poles, residues, d, h)

    # 构造输入参数行
    input_lines = [
        f'poles: {", ".join(f"{p.real:g}{p.imag:+g}j" for p in poles)}',
        f'residues: {", ".join(f"{r.real:g}{r.imag:+g}j" for r in residues)}',
        f'offset: {d:g}',
        f'slope: {h:g}'
    ]

    # 系统参数分析与电路生成（代码2的主程序部分）
    analyzer = SystemAnalyzer()
    analyzer.analyze(input_lines=input_lines)
    if analyzer.valid:
        print("\n=== 系统参数分析报告 ===")
        print(f"* RC电路数量: {analyzer.rc_num}")
        print(f"* RL电路数量: {analyzer.rl_num}")
        print(f"* RLC电路数量: {analyzer.rlc_num}")

        if analyzer.output_data['offset'] is not None and analyzer.output_data['slope'] is not None:
            print("\n[Offset & Slope]")
            print(f"Offset: {analyzer.output_data['offset']:.5f}")
            print(f"Slope: {analyzer.output_data['slope']:.5e}")

        if analyzer.output_data['rl_pairs']:
            print("\n[RL串联电路参数]")
            for pair in analyzer.output_data['rl_pairs']:
                pole = pair['pole']
                residue = pair['residue']
                pole_str = f"{pole.real:.5f}" if abs(pole.imag) < SystemAnalyzer.DEFAULT_TOLERANCE else f"{pole.real:.5f}{pole.imag:+.5f}j"
                res_str = f"{residue.real:.5f}" if abs(residue.imag) < SystemAnalyzer.DEFAULT_TOLERANCE else f"{residue.real:.5f}{residue.imag:+.5f}j"
                print(f"RL_{pair['id'].upper()}:")
                print(f"  极点: {pole_str}")
                print(f"  留数: {res_str}")

        if analyzer.output_data['rlc_pairs']:
            print("\n[RLC受控源电路参数]")
            for pair in analyzer.output_data['rlc_pairs']:
                p1, p2 = pair['poles']
                r1, r2 = pair['residues']
                p1_str = f"{p1.real:.5f}{p1.imag:+.5f}j"
                p2_str = f"{p2.real:.5f}{p2.imag:+.5f}j"
                r1_str = f"{r1.real:.5f}{r1.imag:+.5f}j"
                r2_str = f"{r2.real:.5f}{r2.imag:+.5f}j"
                print(f"RLC_{pair['id']}:")
                print(f"  极点对: {p1_str}")
                print(f"         {p2_str}")
                print(f"  留数对: {r1_str}")
                print(f"         {r2_str}")

        if analyzer.output_data['rc_params']:
            print("\n[RC并联电路参数]")
            print(f"R = {analyzer.output_data['rc_params']['R']:.4e} Ω")
            print(f"C = {analyzer.output_data['rc_params']['C']:.4e} F")

        if analyzer.output_data['rl_params']:
            print("\n[RL串联电路参数]")
            for param in analyzer.output_data['rl_params']:
                print(f"RL_{param['id'].upper()}:")
                print(f"  L_{param['id'].upper()} = {param['L']:.4e} H")
                print(f"  R_{param['id'].upper()} = {param['R']:.4e} Ω")

        if analyzer.output_data['rlc_params']:
            print("\n[RLC受控源电路参数]")
            for param in analyzer.output_data['rlc_params']:
                print(f"RLC_{param['id']}:")
                print(f"  L_{param['id']} = {param['L']:.4e} H")
                print(f"  R_{param['id']} = {param['R']:.4e} Ω")
                print(f"  C_{param['id']} = {param['C']:.4e} F")
                print(f"  b_{param['id']} = {param['b']:.4e}")
                print(f"  g_m_{param['id']} = {param['g_m']:.4e} S\n")

        try:
            print("\n生成电路图中...")
            drawer = CircuitDrawer(analyzer)
            drawer.draw_combined_circuits()
            print("电路图已生成！")
        except Exception as e:
            print(f"绘图失败: {str(e)}")

    magnitude_f = np.abs(test_f)
    magnitude_fitted = np.abs(fitted)
    phase_f = np.degrees(np.angle(test_f))  # 消除跳变并转度数
    phase_fitted = np.degrees(np.unwrap(np.angle(fitted)))  # 消除跳变并转度数

    # 创建幅频响应图
    plt.figure(figsize=(10, 8))
    w = imag(test_s)
    freq_hz = w / (2 * np.pi)
    # 幅频响应
    plt.subplot(2, 1, 1)
    plt.semilogx(freq_hz, 20 * np.log10(magnitude_f), label='Original', color='b')
    plt.semilogx(freq_hz, 20 * np.log10(magnitude_fitted), '--', label='Fitted', color='r')
    plt.title('Magnitude Response')
    plt.xlabel('Frequency [Hz]')  # 修改单位
    plt.ylabel('Magnitude [dB]')
    plt.grid(True, which='both', linestyle='--')
    plt.legend()
    plt.tight_layout()
    # 相频响应
    plt.subplot(2, 1, 2)
    plt.semilogx(freq_hz, phase_f, label='Original', color='b')
    plt.semilogx(freq_hz, phase_fitted, '--', label='Fitted', color='r')
    plt.title('Phase Response')
    plt.xlabel('Frequency [Hz]')  # 修改单位
    plt.ylabel('Phase [degrees]')
    plt.grid(True, which='both', linestyle='--')
    # 自动调整相频响应的 y 轴范围
    plt.autoscale(axis='y', tight=True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 显示图形
    plt.show()
    """
    Completed by 2025/4/24 17:04
    @author: Prince
    主程序代码中有四个测试案例，前三个合适，第四个一般（534-679行，用哪个就取消注释哪个，实际工程中测出的频率响应的格式应当与案例2相符）
    修改初始预测极点与拟合次数在vectfit_auto函数中 229行
    """