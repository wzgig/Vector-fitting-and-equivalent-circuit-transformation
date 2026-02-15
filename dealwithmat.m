clear,clc,tic
%% ================= 配置部分 =================
INPUT_DIR  = "your_root_small";  
OUTPUT_DIR = fullfile(INPUT_DIR, "csv_data");

F_START    = 1e-1;
F_END      = 1e5;
NUM_POINTS = 2000;

INPUT_IDXS  = [1 2];   % v_alpha, v_beta
OUTPUT_IDXS = [1 2];   % i_alpha, i_beta

SAVE_MAGPHASE = false;   % VF.py 不需要，建议 false
USE_PARALLEL  = false;   % 文件多可以 true
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

%% ========== 新增：初始化CSV计数器 ==========
csvCount = 0;                  % 普通循环计数器
successFlags = false(numel(matFiles), 1);  % 并行循环标记数组

%% 选择循环方式
if USE_PARALLEL
    parfor k = 1:numel(matFiles)
        % 并行循环中用数组存储成功标记（不能直接累加）
        successFlags(k) = processOneFile(matFiles(k), freqs_hz, w_rad, ...
                       INPUT_IDXS, OUTPUT_IDXS, ...
                       OUTPUT_DIR, SAVE_MAGPHASE);
    end
    csvCount = sum(successFlags);  % 统计成功数量
else
    for k = 1:numel(matFiles)
        % 普通循环直接累加成功次数
        csvCount = csvCount + processOneFile(matFiles(k), freqs_hz, w_rad, ...
                       INPUT_IDXS, OUTPUT_IDXS, ...
                       OUTPUT_DIR, SAVE_MAGPHASE);
    end
end

fprintf("=== 全部处理完成 ===\n");
%% ========== 新增：显示生成的CSV数量 ==========
fprintf("成功生成 %d 个 CSV 文件。\n", csvCount);
toc

%% ==========================================================
% ========== 修改：函数添加返回值，标记是否成功生成CSV ==========
function success = processOneFile(fileInfo, freqs_hz, w_rad, ...
                        INPUT_IDXS, OUTPUT_IDXS, ...
                        OUTPUT_DIR, SAVE_MAGPHASE)
    % 初始化返回值：默认失败（0）
    success = 0;
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
        
        %% ========== 新增：标记为成功 ==========
        success = 1;

    catch ME
        fprintf("处理失败: %s → %s\n", fileInfo.name, ME.message);
        % 失败时保持success=0
    end
end