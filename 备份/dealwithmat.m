%% ================= 配置部分 =================
% % 输入文件夹：MATLAB 生成的 mat 文件所在路径
% INPUT_DIR = "your root";   % <- 改成你的路径，例如 "E:\ruanjian\..."
% 
% % 输出文件夹：生成的 CSV 存放路径
% OUTPUT_DIR = fullfile(INPUT_DIR, "csv_data");
% 
% % 频率扫描范围 (Hz)
% F_START   = 1e-1;
% F_END     = 1e5;   % 根据风机带宽调整，例如 5kHz
% NUM_POINTS = 2000;     % 频率点数量
% 
% % 关注的输入输出通道索引（MATLAB 是 1-based）
% % Python: INPUT_IDXS=[0,1] OUTPUT_IDXS=[0,1]
% INPUT_IDXS  = [1, 2];  % v_alpha, v_beta
% OUTPUT_IDXS = [1, 2];  % i_alpha, i_beta
% %% ===========================================
% 
% % 创建输出目录
% if ~isfolder(OUTPUT_DIR)
%     mkdir(OUTPUT_DIR);
%     fprintf("创建输出目录: %s\n", OUTPUT_DIR);
% end
% 
% % 查找所有 .mat 文件
% matFiles = dir(fullfile(INPUT_DIR, "*.mat"));
% fprintf("找到 %d 个 .mat 文件。\n", numel(matFiles));
% 
% % 生成对数分布的频率向量
% freqs_hz = logspace(log10(F_START), log10(F_END), NUM_POINTS).';  % 列向量
% w_rad    = 2*pi*freqs_hz;                                         % rad/s
% 
% for k = 1:numel(matFiles)
%     matPath = fullfile(matFiles(k).folder, matFiles(k).name);
%     try
%         % 1) 加载 MAT 文件
%         S = load(matPath);
% 
%         % --- 提取 ABCD ---
%         % 情况A：mat里直接有 A,B,C,D
%         if isfield(S,"A") && isfield(S,"B") && isfield(S,"C") && isfield(S,"D")
%             A = S.A; B = S.B; C = S.C; D = S.D;
% 
%         % 情况B：mat里存的是 sys（ss对象）
%         elseif isfield(S,"sys")
%             sys0 = S.sys;
%             [A,B,C,D] = ssdata(sys0);
% 
%         else
%             error("MAT 文件中找不到 A,B,C,D 或 sys。");
%         end
% 
%         % 2) 构建状态空间系统
%         sys = ss(A, B, C, D);
% 
%         % 3) 计算频率响应
%         % freqresp 返回大小: [ny, nu, N]
%         H = freqresp(sys, w_rad);
% 
%         % 4) 组织表格并导出 CSV
%         T = table(freqs_hz, w_rad, 'VariableNames', {'Frequency_Hz','Omega_rad'});
% 
%         % 遍历关心的通道：输出(1,2) 输入(1,2) -> Y11,Y12,Y21,Y22
%         for out_i = OUTPUT_IDXS
%             for in_j = INPUT_IDXS
%                 resp = squeeze(H(out_i, in_j, :));  % N x 1 复数
% 
%                 % label: Y{out}{in}，例如 Y11, Y12
%                 label = sprintf("Y%d%d", out_i, in_j);
% 
%                 T.(label + "_Real")      = real(resp);
%                 T.(label + "_Imag")      = imag(resp);
%                 T.(label + "_Mag")       = abs(resp);
%                 T.(label + "_Phase_Deg") = rad2deg(angle(resp));
%             end
%         end
% 
%         % 保存 CSV
%         [~, baseName, ~] = fileparts(matFiles(k).name);
%         csvPath = fullfile(OUTPUT_DIR, baseName + ".csv");
%         writetable(T, csvPath);
% 
%         fprintf("已处理: %s\n", baseName + ".csv");
% 
%     catch ME
%         fprintf("处理文件 %s 时出错: %s\n", matFiles(k).name, ME.message);
%     end
% end

clear,clc,tic
%% ================= 配置部分 =================
INPUT_DIR  = "your_root";  
OUTPUT_DIR = fullfile(INPUT_DIR, "csv_data");

F_START    = 1e-1;
F_END      = 1e5;
NUM_POINTS = 2000;

INPUT_IDXS  = [1 2];   % v_alpha, v_beta
OUTPUT_IDXS = [1 2];   % i_alpha, i_beta

SAVE_MAGPHASE = false;   % VF.py 不需要，建议 false
USE_PARALLEL  = false;   % 文件多可以 truef
%% ============================================

%% 创建输出目录
if ~isfolder(OUTPUT_DIR)
    mkdir(OUTPUT_DIR);
    fprintf("创建输出目录: %s\n", OUTPUT_DIR);
end

%% 查找 MAT 文件
matFiles = dir(fullfile(INPUT_DIR, "*.mat"));
fprintf("找到 %d 个 .mat 文件。\n", numel(matFiles));

%% 频率向量
freqs_hz = logspace(log10(F_START), log10(F_END), NUM_POINTS).';
w_rad    = 2*pi*freqs_hz;

%% 选择循环方式
if USE_PARALLEL
    parfor k = 1:numel(matFiles)
        processOneFile(matFiles(k), freqs_hz, w_rad, ...
                       INPUT_IDXS, OUTPUT_IDXS, ...
                       OUTPUT_DIR, SAVE_MAGPHASE);
    end
else
    for k = 1:numel(matFiles)
        processOneFile(matFiles(k), freqs_hz, w_rad, ...
                       INPUT_IDXS, OUTPUT_IDXS, ...
                       OUTPUT_DIR, SAVE_MAGPHASE);
    end
end

fprintf("=== 全部处理完成 ===\n");
toc
%% ==========================================================
function processOneFile(fileInfo, freqs_hz, w_rad, ...
                        INPUT_IDXS, OUTPUT_IDXS, ...
                        OUTPUT_DIR, SAVE_MAGPHASE)

    matPath = fullfile(fileInfo.folder, fileInfo.name);

    try
        %% 1) 加载文件
        S = load(matPath);

        %% 2) 提取系统
        if isfield(S,"A") && isfield(S,"B") && isfield(S,"C") && isfield(S,"D")
            A = double(S.A);
            B = double(S.B);
            C = double(S.C);
            D = double(S.D);

        elseif isfield(S,"sys")
            [A,B,C,D] = ssdata(S.sys);
            A = double(A); B = double(B);
            C = double(C); D = double(D);

        else
            error("找不到 A,B,C,D 或 sys");
        end

        sys = ss(A,B,C,D);

        %% 3) 输入输出维度检查
        [ny, nu] = size(D);

        if max(INPUT_IDXS) > nu || max(OUTPUT_IDXS) > ny
            error("通道越界：系统输入=%d 输出=%d", nu, ny);
        end

        %% 4) 计算频率响应
        H = freqresp(sys, w_rad);   % ny × nu × N

        %% 5) 构造输出 table（一次性分配列）
        T = table(freqs_hz, 'VariableNames', {'Frequency_Hz'});

        for out_i = OUTPUT_IDXS
            for in_j = INPUT_IDXS

                resp = squeeze(H(out_i, in_j, :));
                label = sprintf("Y%d%d", out_i, in_j);

                % VF.py 最需要：Real + Imag
                T.(label+"_Real") = real(resp);
                T.(label+"_Imag") = imag(resp);

                % 可选输出幅值相位
                if SAVE_MAGPHASE
                    T.(label+"_Mag")   = abs(resp);
                    T.(label+"_Phase") = rad2deg(angle(resp));
                end
            end
        end

        %% 6) 写 CSV
        [~, baseName, ~] = fileparts(fileInfo.name);
        csvPath = fullfile(OUTPUT_DIR, baseName + ".csv");

        writetable(T, csvPath);

        fprintf("已处理: %s\n", baseName);

    catch ME
        fprintf("处理失败: %s → %s\n", fileInfo.name, ME.message);
    end
end
