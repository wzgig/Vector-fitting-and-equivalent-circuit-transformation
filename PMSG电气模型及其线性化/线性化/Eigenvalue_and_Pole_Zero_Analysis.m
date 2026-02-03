%% ====== Plot eigenvalues (poles) / pole-zero map for state-space ======
% Requirements:
%   - If you only have A: run section 1 (eigenvalue plot).
%   - If you have A,B,C,D: run section 2 (pzmap).

%% -------- Section 0: bring your matrices here --------
% Example:
% A = ...;  B = ...;  C = ...;  D = ...;

%% -------- Section 1: eigenvalues of A (poles) in complex plane --------
lam = eig(A);

figure('Name','Eigenvalues of A (Poles)','Color','w'); clf;
plot(real(lam), imag(lam), 'x', 'MarkerSize', 10, 'LineWidth', 1.8); hold on;
xline(0,'--','LineWidth',1.2);
yline(0,'--','LineWidth',1.2);
grid on; axis equal;
xlabel('Real(\lambda)');
ylabel('Imag(\lambda)');
title('Eigenvalues of A (Poles)');

% Annotate unstable poles (real > 0)
idx_unstable = real(lam) > 0;
if any(idx_unstable)
    plot(real(lam(idx_unstable)), imag(lam(idx_unstable)), 'ro', ...
        'MarkerSize', 8, 'LineWidth', 1.6);
    legend('All poles','Re=0 axis','Imag=0 axis','Unstable poles','Location','best');
else
    legend('All poles','Re=0 axis','Imag=0 axis','Location','best');
end

% Print a quick summary
fprintf('Total poles: %d\n', numel(lam));
fprintf('Unstable poles (Re>0): %d\n', nnz(idx_unstable));
[~,I] = sort(real(lam),'descend');
disp('Poles sorted by real part (top 10):');
disp(lam(I(1:min(10,end))));

%% -------- Section 2: pole-zero map (needs A,B,C,D) --------
% Only run if you have B,C,D defined
if exist('B','var') && exist('C','var') && exist('D','var')
    sys = ss(A,B,C,D);

    figure('Name','Pole-Zero Map (pzmap)','Color','w'); clf;
    pzmap(sys); grid on;
    title('Pole-Zero Map (State-Space System)');

    % Optional: show damping / natural frequency
    figure('Name','Damping Report','Color','w'); clf;
    damp(sys); % prints to command window and draws a plot in newer MATLAB versions
end

%% -------- Optional: if you want zeros/poles numerically --------
% (needs A,B,C,D)
if exist('B','var') && exist('C','var') && exist('D','var')
    p = pole(sys);
    z = tzero(sys);
    fprintf('\nPoles (pole(sys)):\n'); disp(p);
    fprintf('Transmission zeros (tzero(sys)):\n'); disp(z);
end
