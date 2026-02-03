clear;
clc;

f    = 60;
Fnom    = f;
Np  = 48;
w_g = 2*pi*f;
Sbase = 2e6;
Vbase = 690;
Ibase = Sbase/(Vbase);
wbase = w_g;
Zbase = Vbase^2/Sbase;
Lbase = Zbase/w_g;%电感基值
Cbase = 1/(wbase*Zbase);%电容基值
Tbase = Sbase/(w_g/Np);
U_dcref = 1150;
Kp_PLL = 250;
Ki_PLL = 3200;
ugq_ref = 0;
v_w = 12;
%% ---- 输入参数 ----
P  = -1;        % 有功功率（标幺值）。发电模式为负值（按你的约定）
Q  = 0;         % 无功功率（标幺值）
V  = 1;    % 电压幅值（标幺值）
xi = -0.2;     % 电压相角（弧度）

%% ---- 电网电压的α-β坐标系表示（功率不变框架下v_alpha、v_beta取值相同；缩放系数体现在p,q方程中） ----
v_ab    = V * exp(1i*xi);
v_alpha = real(v_ab);  % 电压α轴分量
v_beta  = imag(v_ab);  % 电压β轴分量

%% ---- 从p-q方程求解i_alpha、i_beta（等功率 => 不含3/2系数） ----
% p = v_alpha*i_alpha + v_beta*i_beta
% q = v_beta*i_alpha  - v_alpha*i_beta

v2 = v_alpha^2 + v_beta^2;   % 等于V²

if v2 < 1e-12
    error('电压幅值过小：无法计算电流参考值。');
end

% 逆映射公式（等功率）：
% [i_alpha; i_beta] = 1/v2 * [ v_alpha  v_beta;
%                              v_beta  -v_alpha] * [P; Q]
i_vec   = 1 / v2 * [ v_alpha,  v_beta;
                     v_beta,  -v_alpha ] * [P; Q];

i_alpha = i_vec(1);
i_beta  = i_vec(2);
i_ab    = i_alpha + 1i*i_beta;

syms PLL_int thetapll
%% ========== 坐标变换（GSC的电压观测） ==========
% PLL
theta_g = thetapll;         % 网侧PLL估计的同步角
% 构造变换矩阵，将d-q轴量转换为定轴坐标系（用于测量或控制）
Tg_alphabeta_dq = [cos(theta_g) sin(theta_g); -sin(theta_g) cos(theta_g)];
vg_dq_real = Vbase * Tg_alphabeta_dq * [v_alpha; v_beta];  % GSC中的实际电压
% 变换后各轴上的电压值提取
v_g_d = vg_dq_real(1);          % 网侧d轴电压
v_g_q = vg_dq_real(2);         % 网侧q轴电压

%% ========== xi求解 ==========
eqnq = Ki_PLL*(v_g_q/Vbase - ugq_ref) == 0;
[pllg] = solve(eqnq, thetapll);
pllg_v   = vpa(pllg);
pllg_r   = real(pllg_v);
xi_v = vpa(xi);
[~, idx] = min(abs(pllg_r - xi_v));
thetapll = pllg_r(idx);
disp(thetapll)
PLL_int = 0;

v_g_d = round(eval(v_g_d));
v_g_q = round(eval(v_g_q));
%% ---- 主程序 ----
run("PMSGRef_and_Para.m");
Tg_dq_alphabeta = [cos(theta_g) -sin(theta_g); sin(theta_g) cos(theta_g)];
ig_dig_alphabeta = (1/Ibase)*Tg_dq_alphabeta*[i_gd;i_gq];

i_galpha = ig_dig_alphabeta(1);
i_gbeta = ig_dig_alphabeta(2);

eqn1 = (Te - T_m - D_m*omega_m) / J == 0;
eqn2 = u_sq - R_s*i_sq - omega_e*Psi_sd == 0;
eqn3 = u_sd - R_s*i_sd + omega_e*Psi_sq == 0;
eqn4 = i_sdref - i_sd == 0;
eqn5 = i_sqref - i_sq == 0;
eqn6 = n_ref - n == 0;
eqn7 = (u_sdref - u_sd)/T_d == 0;
eqn8 = (u_sqref - u_sq)/T_d == 0;
eqn9 = (P_s - P_dc)/(C_dc*U_dc) == 0;
eqn10 = U_dc - U_dcref == 0;
eqn11 = (v_g_d - u_gd - R_g*i_gd + w_g*L_g*i_gq)/L_g == 0;
eqn12 = (v_g_q - u_gq - R_g*i_gq - w_g*L_g*i_gd)/L_g == 0;
eqn13 = i_gdref - i_gd == 0;
eqn14 = i_gqref - i_gq == 0;
eqn15 = i_galpha == i_alpha;
eqn16 = i_gbeta == i_beta;

[omega_m, Psi_sq, Psi_sd, Id_stator_int, Iq_stator_int, Speed_int,...
    u_sd, u_sq, U_dc, Udc_int, i_gd, i_gq, Id_grid_int, Iq_grid_int,...
    omega_best, i_gqref]...
                      = vpasolve(eval([eqn1,eqn2,eqn3,eqn4,eqn5,eqn6, ...
                      eqn7,eqn8,eqn9,eqn10,eqn11,eqn12,eqn13,eqn14, ...
                      eqn15,eqn16]),...
                      [omega_m Psi_sq Psi_sd Id_stator_int Iq_stator_int Speed_int ...
                      u_sd u_sq U_dc Udc_int i_gd i_gq Id_grid_int Iq_grid_int ...
                      omega_best i_gqref],...
                      [-inf,inf;-inf,inf;-inf,inf;-inf,inf;-inf,inf;-inf,inf;-inf,inf;-inf,inf; ...
                      -inf,inf;-inf,inf;-inf,inf;-inf,inf;-inf,inf;-inf,inf; ...
                      -inf,inf;-inf,inf]);

inistate = [omega_m Psi_sq Psi_sd Id_stator_int Iq_stator_int Speed_int...
    u_sd u_sq U_dc Udc_int i_gd i_gq Id_grid_int Iq_grid_int PLL_int thetapll]'
state_size = size(inistate,1);

omega_best = double(omega_best);
i_gqref = double(i_gqref);
disp(omega_best);disp(i_gqref);

dt = 0.00001;
tspan=dt:dt:10;
options = odeset('RelTol',1e-12,'AbsTol',...
    1e-12*ones(1,state_size));
[t,x] = ode45(@(t,x) PMSG(t,x,omega_best, i_gqref),tspan,double(inistate),options);

% run("tuoMWRef_and_Para.m");

%%%%%%%%%% states


omega_m = x(:,1);
Psi_sq = x(:,2);
Psi_sd = x(:,3);
Id_stator_int = x(:,4);
Iq_stator_int = x(:,5);
Speed_int = x(:,6);
u_sd = x(:,7);
u_sq = x(:,8);
U_dc = x(:,9);
Udc_int = x(:,10);
i_gd = x(:,11);
i_gq = x(:,12);
Id_grid_int = x(:,13);
Iq_grid_int = x(:,14);
PLL_int = x(:,15);
thetapll = x(:,16);

disp(thetapll(1));

figure(1);
plot(tspan, eval(P_m)/Sbase, 'DisplayName', 'P_m');
hold on;
plot(tspan, eval(-P_s)/Sbase, 'DisplayName', 'P_s');
hold on;
% plot(tspan, eval(-P_e)/Sbase, 'DisplayName', 'P_e');
legend('Location', 'best', 'Interpreter', 'tex');

figure(2);
plot(tspan, eval(P_g/Sbase), 'DisplayName', 'P_g^{pu}');
hold on;
plot(tspan, eval(Q_g/Sbase), 'DisplayName', 'P_g^{pu}');
hold on;
legend('Location', 'best', 'Interpreter', 'tex');   % Location参数指定图例位置，'best'表示自动选择最佳位置