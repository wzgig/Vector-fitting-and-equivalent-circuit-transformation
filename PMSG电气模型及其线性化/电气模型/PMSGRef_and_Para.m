f    = 60;
Fnom    = f;
Np  = 48;
w_g = 2*pi*f;
Sbase = 2e6;
Pnom = Sbase;
Vbase = 690;
Ibase = Sbase/(Vbase);
wbase = w_g;
Zbase = Vbase^2/Sbase;
Lbase = Zbase/w_g;%电感基值
Cbase = 1/(wbase*Zbase);%电容基值
Tbase = Sbase/(w_g/Np);
U_dcref = 1150;

R_g = 0.05;%pu
L_g = 7e-3;%pu
C_dc = 1e-3;


Psi_f = 3.88889;%ac           % 转子磁链 
L_sd = 1.8e-3;%ac         % d轴电感 
L_sq = 1.8e-3;%ac         % q轴电感 
R_s = 0.0026;%ac          % 定子电阻
beta = 0;                 % 桨距角，单位：度（固定值
pitch = beta;
% v_w = 10.611186933034323618371655280757;               % 风速 (m/s)
J = 35000;%ac
D_m = 0.078;%ac           % 自阻尼系数
R_t = 36.6;
% rho = 1.12;
v_w =12;

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
T_m = 1/3000;
T_trq = 60;


%%%%%%%%%%%%%%%%%%%%%%%%%% Ref    %%%%%%%%%%%%%%%%%%%%%%%%
v_w = 12;
%%%%%%%%%%%%%%%%%%%%%%%%%% Algebra %%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%% algebra
% syms omega_m Psi_sq Psi_sd x1 x2 y1 u_sd u_sq xpll thetapll U_dc z1 i_gd i_gq z2 z3;
syms omega_m Psi_sq Psi_sd Id_stator_int Iq_stator_int Speed_int...
    u_sd u_sq U_dc Udc_int i_gd i_gq Id_grid_int Iq_grid_int;
syms omega_best i_gqref

% omega_best = 8.1*v_w/R_t;%最佳叶尖速比
n_ref = omega_best*30/pi;

n = omega_m * 30/pi;
omega_e = omega_m*Np;
omega_r = omega_m * (Np/(Fnom*2*pi));

i_sd = (Psi_sd - Psi_f)/L_sd;
i_sq = Psi_sq/L_sq;

Te = Np*(Psi_f*i_sq);
Tem = Te * (1/(Np*Pnom/(2*pi*Fnom)));

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