%% pf_Yaa_robust.m
% 适用于连续时间 tf 的高阶部分分式分解（导纳/阻抗都可）
% 核心：zpk 方式更稳 + 归一化 + minreal清理 + 误差验证 + 高精度兜底

clear; clc;
run("Admittancesingle.mlx")
%% ===== 0) 你的系统：Yaa (已存在于工作区) =====
% 假设你已经有 Yaa = tf(...)

assert(isa(Yaa,'tf'), 'Yaa 必须是 tf 对象');
assert(Yaa.Ts == 0, '当前脚本仅针对连续系统（Ts=0）');

%% ===== 1) 先做最小实现：清理潜在零极点抵消（非常重要）=====
% 容差可调：如果你后面发现抵消过多/过少，调大或调小 tol
tol_cancel = 1e-8;
Yc = minreal(Yaa, tol_cancel);

%% ===== 2) 取系数并做归一化（减少病态）=====
[num, den] = tfdata(Yc, 'v');

% 分母首项归一化（roots/residue 稳定性会好一些）
den = den / den(1);
num = num / den(1);

% 可选：进一步"尺度归一化"（对高阶大跨度有帮助）
% 把系数整体缩放到量级接近 1（不改变传递函数）
scale_den = 10.^round(log10(max(abs(den))));
scale_num = 10.^round(log10(max(abs(num))));
den = den / scale_den;
num = num / scale_num;
% 注意：这里等价于把系统整体乘了 (scale_den/scale_num)，要补回
gain_scale = scale_den / scale_num;

%% ===== 3) 优先用 zpk 方式求极点/零点，再由极点做 residue =====
% zpk 的内部算法对极点求解往往比直接对多项式 roots 稳一些
Z = zero(tf(num, den));
P = pole(tf(num, den));
Kzpk = dcgain(zpk(Z,P,1)); %#ok<NASGU> % 仅用于检查，不强依赖

% 直接项（严格真分式才是 residue 的"干净输入"）
[q, r] = deconv(num, den);     % num = q*den + r
% residue 只对 r/den 做（避免 K 干扰）
[R, P2, K0] = residue(r, den); %#ok<ASGLU>

% 补回缩放增益
R = R * gain_scale;
q = q * gain_scale;

% 合并直接项
K = q;

%% ===== 4) 结果输出 =====
disp('=== Residues (R) ==='); disp(R);
disp('=== Poles (P) ===');    disp(P2);
disp('=== Direct Terms (K) ==='); disp(K);

%% ===== 5) 反算验证：不要用"系数逐项相等"判断对不对，应该看系统误差 =====
[num_rec, den_rec] = residue(R, P2, K);
Y_rec = tf(num_rec, den_rec);

% 与 minreal 后的原系统对比（更公平）
E = minreal(Yc - Y_rec, 1e-7);

disp('=== 误差系统 E(s) = Yc - Y_rec （做过 minreal） ===');
E

% 给出频域相对误差（工程上更直观）
w = logspace(0, 6, 2000);  % 1 到 1e6 rad/s，可按你系统带宽改
H1 = squeeze(freqresp(Yc, w));
H2 = squeeze(freqresp(Y_rec, w));

relErr = max(abs(H1 - H2) ./ max(abs(H1), 1e-12));
absErr = max(abs(H1 - H2));

fprintf('\nmax relative error over w grid = %.3e\n', relErr);
fprintf('max absolute  error over w grid = %.3e\n', absErr);

%% ===== 6) 如果误差仍然偏大：启用"高精度兜底"建议 =====
% 一般在以下情况需要：
% - 极点非常接近（成对/簇）
% - 多重极点（重复根）
% - 系数跨度更大或阶数更高
%
% 你可以在误差 > 1e-6 之类的阈值时启用符号高精度方案（见下方注释）
%
%{
if relErr > 1e-6
    warning('相对误差偏大，建议使用符号高精度做兜底。');

    syms s
    Ns = poly2sym(num / gain_scale, s); % 注意：这里 num/den 是做过缩放的，先除回去
    Ds = poly2sym(den, s);
    Ysym = (Ns/Ds) * gain_scale;

    % 求极点（高精度）
    poles_sym = vpa(solve(Ds == 0, s), 50);

    % 留数：对每个极点计算（假设极点不重复；若重复需扩展公式）
    R_sym = sym(zeros(size(poles_sym)));
    for k = 1:numel(poles_sym)
        R_sym(k) = vpa(limit((s - poles_sym(k))*Ysym, s, poles_sym(k)), 50);
    end

    disp('=== vpa Poles ==='); disp(poles_sym);
    disp('=== vpa Residues ==='); disp(R_sym);
end
%}

% 原系统的极点（不 minreal）
p0 = pole(Yaa)

% minreal 后的极点
p1 = pole(minreal(Yaa, 1e-8))

disp("原系统极点数: " + numel(p0));
disp("minreal后极点数: " + numel(p1));
disp("原系统是否有共轭对（看虚部）:");
disp(p0(abs(imag(p0))>1e-6));
