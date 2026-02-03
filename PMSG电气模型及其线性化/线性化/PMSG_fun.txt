function [dotx, yy] = PMSG_fun(t,x,u)

%%%%%%%%%%%%%%%%%%%%%%%%%% Rename %%%%%%%%%%%%%%%%%%%%%%%%
omega_m = x(1);
Psi_sq  = x(2);
Psi_sd  = x(3);
Id_stator_int = x(4);
Iq_stator_int = x(5);
Speed_int     = x(6);
u_sd          = x(7);
u_sq          = x(8);
U_dc        = x(9);
Udc_int     = x(10);
i_gd        = x(11);
i_gq        = x(12);
Id_grid_int = x(13);
Iq_grid_int = x(14);
PLL_int   = x(15);
thetapll  = x(16);  

%%%%%%%%%%%%%%%%%%%%%%%%%%  Paras  %%%%%%%%%%%%%%%%%%%%%%%%
v_alpha = u(1);   % the global voltage should be input
v_beta = u(2);
omega_best = u(3);
i_gqref = u(4);

f    = 60;
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

R_g = 0.05;
L_g = 7e-3;
C_dc = 1e-3;

Psi_f = 3.88889;%ac           % 转子磁链 
L_sd = 1.8e-3;%ac         % d轴电感 
L_sq = 1.8e-3;%ac         % q轴电感 
R_s = 0.0026;%ac          % 定子电阻
beta = 0;                 % 桨距角，单位：度（固定值
pitch = beta;
J = 35000;%ac
D_m = 0.078;%ac           % 自阻尼系数
R_t = 36.6;
% rho = 1.12;
v_w = 12;

Kp_Id_stator = 1;
Ki_Id_stator = 12;
Kp_Iq_stator = 1;
Ki_Iq_stator = 12;
Kp_Speed = 100;
Ki_Speed = 220;
Kp_Udc = 1;
Ki_Udc = 10;
Kp_Id_grid = 1;
Ki_Id_grid = 15;
Kp_Iq_grid = 1;
Ki_Iq_grid = 15;

Kp_PLL = 250;
Ki_PLL = 3200;
T_d = 1/6000;

%%%%%%%%%%%%%%%%%%%%%%%%%% Algebra %%%%%%%%%%%%%%%%%%%%%%%%
%% ========== 坐标变换（GSC的电压观测） ==========
% PLL
theta_g = thetapll;         % 网侧PLL估计的同步角
% 构造变换矩阵，将d-q轴量转换为定轴坐标系（用于测量或控制）
Tg_alphabeta_dq = [cos(theta_g) sin(theta_g); -sin(theta_g) cos(theta_g)];
vg_dq_real = Vbase * Tg_alphabeta_dq * [v_alpha; v_beta];  % GSC中的实际电压
% 变换后各轴上的电压值提取
v_g_d = vg_dq_real(1);          % 网侧d轴电压
v_g_q = vg_dq_real(2);         % 网侧q轴电压

% omega_best = 8.1*v_w/R_t;%最佳叶尖速比
n_ref = omega_best*30/pi;
ugq_ref = 0;

n = omega_m * 30/pi;
omega_e = omega_m*Np;

i_sd = (Psi_sd - Psi_f)/L_sd;
i_sq = Psi_sq/L_sq;

Te = Np*(Psi_f*i_sq);

u1 = 0.5*pi;
u2 = 1.225;
lambda = (R_t*omega_m)/(v_w);
lambda_i = 1/(1/(lambda+0.08*beta)-0.035/(beta^3+1));
Cp = 0.51763*(116/lambda_i-0.4*beta-5)*exp(-21/lambda_i)+0.006795*lambda;
P_m = u1*u2*(R_t^2)*(v_w^3)*Cp;
T_m = (-1)*(P_m)/omega_m;

% dy1 = n_ref - n;
% dx1 = i_sdref - i_sd;
% dx2 = i_sqref - i_sq;
i_sdref = 0;
i_sqref = Kp_Speed*(n_ref - n) + Ki_Speed*Speed_int;
u_sqref = Kp_Iq_stator*(i_sqref - i_sq)+Ki_Iq_stator*Iq_stator_int+omega_e*Psi_f+omega_e*L_sd*i_sd;
u_sdref = Kp_Id_stator*(i_sdref - i_sd)+Ki_Id_stator*Id_stator_int-omega_e*L_sq*i_sq;

P_s = (u_sd * i_sd + u_sq * i_sq);%(2-17)
Q_s = (u_sq * i_sd - u_sd * i_sq);%(2-18)
P_e = Te * omega_m;

i_gdref = Kp_Udc*(U_dc - U_dcref) + Ki_Udc*Udc_int;

u_gd = - Kp_Id_grid*(i_gdref - i_gd) - Ki_Id_grid*Id_grid_int - R_g*i_gd + v_g_d+i_gq*w_g*L_g;
u_gq = - Kp_Iq_grid*(i_gqref - i_gq) - Ki_Iq_grid*Iq_grid_int - R_g*i_gq + v_g_q-i_gd*w_g*L_g;

P_dc = (u_gd* i_gd + u_gq * i_gq);%(2-17)
P_g = (v_g_d * i_gd + v_g_q * i_gq);%(2-17)
Q_g = (v_g_q * i_gd - v_g_d * i_gq);%(2-18)
%%%%%%%%%%%%%%%%%%%%%%% Differential %%%%%%%%%%%%%%%%%%%%%
Tg_dq_alphabeta = [cos(theta_g) -sin(theta_g); sin(theta_g) cos(theta_g)];
ig_dq_alphabeta = (1/Ibase)*Tg_dq_alphabeta*[i_gd;i_gq];

i_galpha = ig_dq_alphabeta(1);
i_gbeta = ig_dq_alphabeta(2);

yy = [i_galpha; i_gbeta];


    dotx = [
        (Te - T_m - D_m*omega_m) / J;
        u_sq - R_s*i_sq - omega_e*Psi_sd;
        u_sd - R_s*i_sd + omega_e*Psi_sq;
        i_sdref - i_sd;
        i_sqref - i_sq;
        n_ref - n;
        (u_sdref - u_sd)/T_d;
        (u_sqref - u_sq)/T_d;
        (P_s - P_dc)/(C_dc*U_dc);
        U_dc - U_dcref;
        (v_g_d - u_gd - R_g*i_gd + w_g*L_g*i_gq)/L_g;
        (v_g_q - u_gq - R_g*i_gq - w_g*L_g*i_gd)/L_g;
        i_gdref - i_gd;
        i_gqref - i_gq;
        Ki_PLL*(v_g_q/Vbase - ugq_ref);
        PLL_int + Kp_PLL*(v_g_q/Vbase - ugq_ref);
    ];
end


