% Matlab script for the eigenvalue problem using Advanpix

Precision = 17*2;
mp.Digits(Precision);

% --- Parameters ---
mu = mp('1e-6');           % Viscosity parameter (Advanpix mp)
k_value = mp('1.05');      % Wavenumber (Advanpix mp)
k2 = k_value^2;            % Precompute k^2 (Advanpix mp)
target_eigenvalue = 0.03143596668 ; % + mp(4)*mu; % Target eigenvalue for search (Advanpix mp)
% 0.03143596668
% 0.0621256
% -0.4262968344

num_eigenvalues_to_find = 1; % Number of eigenvalues to find

Nmax = 6*round(double(sqrt(target_eigenvalue/mu)))

Nmax = 60000;              % Truncation size (maximum n)

disp(num2str([k_value mu Nmax], '%.2f\t%.2g\t%d'))

if_no_bc = 1;
if_include_additional_bc = 0;

% --- Initialize Matrices (Sparse) ---

R_mat = mp(sparse(Nmax, Nmax)); % Advanpix sparse matrix
Q_mat = mp(sparse(Nmax, Nmax)); % Advanpix sparse matrix
T_mat = mp(sparse(Nmax, Nmax)); % Advanpix sparse matrix

% --- Precompute Coefficients ---
S1 = compute_S1_mp(Nmax);
S2 = compute_S2_mp(S1, Nmax);
S3 = compute_S3_mp(S2, S1, Nmax);
R_coeffs = compute_R_mp(S2, Nmax, k2);
Q_coeffs = compute_Q_mp(S3, S1, Nmax, k2);
T_coeffs = compute_T_mp(S2, R_coeffs, Nmax, k2);

V_ = zeros(Nmax,1, 'mp');
V_(2) = mp(1);
for n=1:Nmax-2
    V_(n+2) = V_(n+1)*mp(2)*mp(2*n+1)/mp(n)/mp(n+1) + V_(n);
end


if if_include_additional_bc
    NNN = Nmax - 3;
else
    NNN = Nmax - 1;
end

if if_no_bc
    NNN = Nmax+1;
end
% --- Build Matrices ---
for n = 1:Nmax

    % --- R Matrix ---
    if n < NNN
        R_mat(n, n) = -R_coeffs.R_n0(n);  % Diagonal element

        if n - 2 >= 1
            R_mat(n, n - 2) = R_coeffs.R_np2(n-2); % R_{n-2, +2}
        end
        if n + 2 <= Nmax
            R_mat(n, n + 2) = R_coeffs.R_nm2(n+2); % R_{n+2, -2}
        end
    end

    % --- Q Matrix ---
    if n < NNN
        if n - 3 >= 1
            Q_mat(n, n - 3) = Q_coeffs.Q_np3(n-3);  % Q_{n-3, +3}
        end
        if n - 1 >= 1
            Q_mat(n, n - 1) = -Q_coeffs.Q_np1(n-1); % -Q_{n-1, +1}
        end
        if n + 1 <= Nmax
            Q_mat(n, n + 1) = Q_coeffs.Q_nm1(n+1);      % Q_{n+1, -1}
        end
        if n + 3 <= Nmax
            Q_mat(n, n + 3) = -Q_coeffs.Q_nm3(n+3);     % -Q_{n+3, -3}
        end
    end

    % --- T Matrix ---
    if n < NNN
        T_mat(n, n) = T_coeffs.T_n0(n);          % Diagonal element

        if n - 4 >= 1
            T_mat(n, n - 4) = T_coeffs.T_np4(n-4);  % T_{n-4, +4}
        end
        if n - 2 >= 1
            T_mat(n, n - 2) = -T_coeffs.T_np2(n-2); % -T_{n-2, +2}
        end
        if n + 2 <= Nmax
            T_mat(n, n + 2) = -T_coeffs.T_nm2(n+2); % -T_{n+2, -2}
        end
        if n + 4 <= Nmax
            T_mat(n, n + 4) = T_coeffs.T_nm4(n+4);  % T_{n+4, -4}
        end
    end

end

% --- Boundary Conditions (Last Two Rows of R_mat) ---
if ~if_no_bc
    for m = 0:floor(Nmax/2)
        n_even = 2*m + 1;  % Matlab index for n = 2m
        if n_even <= Nmax
            Q_mat(NNN, n_even) = mp((-1)^m);
            if if_include_additional_bc
                Q_mat(Nmax - 1, n_even) = mp((-1)^m)*V_(n_even);
            end
        end
    end
    for m = 0:floor((Nmax-1)/2)
        n_odd = 2*m + 2;   % Matlab index for n = 2m+1
        if n_odd <= Nmax
            Q_mat(NNN+1, n_odd) = mp((-1)^m);
            if if_include_additional_bc
                Q_mat(Nmax, n_odd) = mp((-1)^m)*V_(n_odd);
            end
        end
    end
    % Q_mat and T_mat last two rows are already zero (sparse initialization).
end

% --- Solve Eigenvalue Problem ---
M = Q_mat + mu * T_mat;
% [eigenvectors, eigenvalues_mat] = eig(full(M), full(R_mat)); % Use full matrices for eig - not needed for eigs
opts.tol   = mp('1e-100');   % Tight convergence tolerance (Advanpix mp)
opts.maxit = 20000;         % Increase maximum iterations
opts.disp  = 1;             % Suppress iterative display
opts.sigma = target_eigenvalue; % Set the shift for shift-and-invert mode

[A, eigenvalues_mat] = mpeigs(M, R_mat, num_eigenvalues_to_find, target_eigenvalue, opts);
% [A, eigenvalues_mat] = eig(full(M), full(R_mat));
eigenvalues = diag(eigenvalues_mat);

% --- Find Eigenvalue Closest to Target ---
if numel(eigenvalues) > 1
    [~, closest_eigenvalue_index] = min(abs(eigenvalues - target_eigenvalue));
    closest_eigenvalue = eigenvalues(closest_eigenvalue_index);
else
    closest_eigenvalue = eigenvalues;
end

% --- Display Result ---
disp(['Eigenvalue (gamma) closest to target ', num2str(double(target_eigenvalue), '%.3g'), ':']); % Use char for mp display
disp(num2str(double(closest_eigenvalue), '%.6g'));

% --- Eigenfunction Plotting (using double for plotting) ---

% A= A/A(1)*mp('pi')/mp(4);


% Base cases:
u = linspace(-mp(1), mp(1), 10001);
Su = NewtonCotes_mp(2, u);

y = linspace(-mp(8), mp(8), 16001);
Sy = NewtonCotes_mp(2, y);
u = tanh(y);
Su = Sy./cosh(y).^2;

Pm = ones(1, length(u), 'mp');     % P_0(u) = 1
P0 = u;                            % P_1(u) = u
EF_r = Pm * A(1);
EF_i = P0 * A(2);
ZT_r = Pm * mp(0);
ZT_i = P0 * mp(2) * A(2);

Sum_norm = mp('0');
for n = 2:round(Nmax)-1
    Pn = (mp(2*n-1) * u .* P0 - mp(n-1) * Pm) / mp(n);
    if ~mod(n,2)
        m=n/2;
        if ~mod(n,4)
            EF_r = EF_r + Pn * A(n+1);
            ZT_r = ZT_r + Pn * A(n+1)*mp(n)*mp(n+1);
        else
            EF_r = EF_r - Pn * A(n+1);
            ZT_r = ZT_r - Pn * A(n+1)*mp(n)*mp(n+1);
        end

        P_m2u = (Pn-1)./(1-u.^2);
        P_m2u(1) = 2*P_m2u(2)-P_m2u(3);
        P_m2u(end) = 2*P_m2u(end-1)-P_m2u(end-2);
        I2m = sum(P_m2u.*Su);
        Sum_norm = Sum_norm + (-1)^m*A(n+1)*I2m;

    else
        if ~mod(n-1,4)
            EF_i = EF_i + Pn * A(n+1);
            ZT_i = ZT_i + Pn * A(n+1)*mp(n)*mp(n+1);
        else
            EF_i = EF_i - Pn * A(n+1);
            ZT_i = ZT_i - Pn * A(n+1)*mp(n)*mp(n+1);
        end
    end
    Pm = P0;
    P0 = Pn;
end
ZT_r = -(1-u.^2) .* ZT_r - k2*EF_r;
ZT_i = -(1-u.^2) .* ZT_i - k2*EF_i;

%%

Norm = -mp('1')/k2/Sum_norm;
A  = Norm*A;
EF = EF_r + 1i*EF_i;
ZT = ZT_r + 1i*ZT_i;
EF = Norm*EF;
ZT = Norm*ZT;
EF_r = real(EF);
EF_i = imag(EF);
ZT_r = real(ZT);
ZT_i = imag(ZT);
%%
Norm2 = sum(ZT_r.*Sy);
EF_r = real(EF)/Norm2;
EF_i = imag(EF)/Norm2;
ZT_r = real(ZT)/Norm2;
ZT_i = imag(ZT)/Norm2;

%%
uo = linspace(-1,1,31);

% h = plot(u, ZT_r, u, ZT_i);
h = plot(y, ZT_r, 'b.', y, ZT_i, 'r.');
set(h(1:2), 'LineWidth', 2)
% hold on
% % plot(uo, 2*sqrt(1-uo.^2).^3/pi, 'ko');
% Norms = real(Syr*V_st);
% h2 = plot(y_r, real(V_st)/Norms, 'k:', y_r, imag(V_st)/Norms, 'k:')
% set(h2(1:2), 'LineWidth', 1)
% hold off

grid on
xlabel('$y$', 'FontSize', 18, 'Interpreter','latex')
ylabel('$\zeta$', 'FontSize', 18, 'Interpreter','latex')
% legend('Re', 'Im', 'Re ST', 'Im ST', 'FontSize', 14, 'Interpreter','latex')
title(num2str([Nmax, double(k_value), double(mu)], 'Nmax=%d,  k=%.2f,  mu=%.3g'), 'FontSize', 16) % Convert k_value, mu to double for title
axis([-1 1 -6 6])

%% Step 9: Save final results

[tmi, imax] = max(abs(A(31:end)));
imax = imax+30;
disp(num2str([imax], 'i_max = %d'))

% Prepare 'result' structure with all relevant data
result.Precision = Precision;
result.k_value = double(k_value);
result.mu = double(mu);
result.Nmax = Nmax;
result.imax = imax;
result.I = 1:length(A);
result.target_eigenvalue = target_eigenvalue;
result.eigenvalues = eigenvalues;
result.A = double(A);

result.u = double(u);
result.EF_r = double(EF_r);
result.EF_i = double(EF_i);
result.ZT_r = double(ZT_r);
result.ZT_i = double(ZT_i);

% Save 'result' 
filename_final = num2str([double(k_value), double(mu), Nmax, Precision], 'EF_k=%.2f_mu=%.2g_Nmax=%d_Precision=%d.mat');
save(filename_final, 'result');

disp('Computation and saving of results completed.');

%% Save to CSV
% --- User Input Section ---
filename = num2str([double(k_value), double(mu), Nmax, Precision], 'EP-k=%.2f_mu=%.2g_Nmax=%d_Precision=%d.csv');

separator_type = 'tab'; % Choose 'comma' or 'tab' as separator

% --- Data Preparation (Assume you have these variables already) ---
% Example data - REPLACE with your actual data

% --- Separator Selection ---
if strcmp(separator_type, 'comma')
    separator = ',';
    separator_head = ',';
elseif strcmp(separator_type, 'tab')
    separator = '\t\t';
    separator_head = '\t\t\t\t';
else
    error('Invalid separator_type. Choose ''comma'' or ''tab''.');
end

% --- File Opening ---
fileID = fopen(filename, 'w');
if fileID == -1
    error('Could not open file for writing.');
end

% --- First Line (Comment with Parameter Values) ---
fprintf(fileID, '# k_value: %.4g, mu: %.4g, EV_initial_guess: %.11f, EV: %.11f\n', ...
    k_value, mu, target_eigenvalue, eigenvalues);

% --- Second Line (Column Labels) ---
labels = {'y', 'u', 'Re\psi', 'Im\psi', 'Re\zeta', 'Im\zeta'};
label_line = strjoin(labels, separator_head);
fprintf(fileID, '%s\n', label_line);

% --- Data Columns Writing ---
num_rows = length(y); % Assuming all data columns have the same length

for i = 1:num_rows
    fprintf(fileID, '%.4f%s%.12f%s%.12f%s%.12f%s%.12f%s%.12f\n', ...
        y(i), separator, ...
        u(i), separator, ...
        result.EF_r(i), separator, ...
        result.EF_i(i), separator, ...
        result.ZT_r(i), separator, ...
        result.ZT_i(i));
end

% --- File Closing ---
fclose(fileID);

disp(['CSV file "' filename '" created successfully with ' separator_type ' separator.']);


%%
E_even = zeros(1, Nmax, 'mp');
E_even(1:4:end)=1;
E_even(3:4:end)=-1;
E_even*(A)
E_even*(V_.*A)
%%
% --- Helper Functions (Coefficient Computations) ---

function S1 = compute_S1_mp(Nmax)
    S1.S_nm1 = zeros(Nmax, 1, 'mp'); % Advanpix mp array
    S1.S_np1 = zeros(Nmax, 1, 'mp'); % Advanpix mp array
    for n = 0:Nmax-1
        S1.S_nm1(n+1) = mp(n) / mp(2*n + 1); % Advanpix mp division
        S1.S_np1(n+1) = mp(n+1) / mp(2*n + 1); % Advanpix mp division
    end
end

function S2 = compute_S2_mp(S1, Nmax)
    S2.S_nm2 = zeros(Nmax, 1, 'mp'); % Advanpix mp array
    S2.S_n0  = zeros(Nmax, 1, 'mp');  % Advanpix mp array
    S2.S_np2 = zeros(Nmax, 1, 'mp'); % Advanpix mp array
    for n = 0:Nmax-1
        S1_np1_m1 = mp(0);
        S1_nm1_p2 = mp(0);
        S1_nm1_m1 = mp(0);

        if n >= 1
            S1_np1_m1 = S1.S_np1(n);
            S1_nm1_m1 = S1.S_nm1(n);
        end
        if n + 2 <= Nmax
            S1_nm1_p2 = S1.S_nm1(n+2);
        end

        S2.S_nm2(n+1) = S1.S_nm1(n+1) * S1_nm1_m1;
        S2.S_n0(n+1)  = S1.S_nm1(n+1) * S1_np1_m1 + S1.S_np1(n+1) * S1_nm1_p2;
        if n + 2 <= Nmax
           S2.S_np2(n+1) = S1.S_np1(n+1) * S1.S_np1(n+2);
        end
    end
end

function S3 = compute_S3_mp(S2, S1, Nmax)
    S3.S_nm3 = zeros(Nmax, 1, 'mp'); % Advanpix mp array
    S3.S_nm1 = zeros(Nmax, 1, 'mp'); % Advanpix mp array
    S3.S_np1 = zeros(Nmax, 1, 'mp'); % Advanpix mp array
    S3.S_np3 = zeros(Nmax, 1, 'mp'); % Advanpix mp array
    for n = 0:Nmax-1
        S1_np1_p1_m2 = mp(0);
        S1_nm1_p1_p2 = mp(0);
        S1_np1_p1_p2 = mp(0);

        if n > 1
            S1_np1_p1_m2 = S1.S_np1(n-1);
        end
        if n + 3 <= Nmax
            S1_nm1_p1_p2 = S1.S_nm1(n+3);
            S1_np1_p1_p2 = S1.S_np1(n+3);
        end

        if n > 1
            S3.S_nm3(n+1) = S2.S_nm2(n+1) * S1.S_nm1(n-1);
        end
        S3.S_nm1(n+1) = S2.S_n0(n+1)  * S1.S_nm1(n+1) + S2.S_nm2(n+1) * S1_np1_p1_m2;
        S3.S_np1(n+1) = S2.S_n0(n+1)  * S1.S_np1(n+1) + S2.S_np2(n+1) * S1_nm1_p1_p2;
        S3.S_np3(n+1) = S2.S_np2(n+1)  * S1_np1_p1_p2;
    end
end

function R_coeffs = compute_R_mp(S2, Nmax, k2)
    R_coeffs.R_nm2 = zeros(Nmax, 1, 'mp'); % Advanpix mp array
    R_coeffs.R_n0  = zeros(Nmax, 1, 'mp');  % Advanpix mp array
    R_coeffs.R_np2 = zeros(Nmax, 1, 'mp'); % Advanpix mp array
    for n = 0:Nmax-1
        R_coeffs.R_nm2(n+1) =  mp(n)*(n+1) * S2.S_nm2(n+1); % Advanpix mp multiplication
        R_coeffs.R_n0(n+1)  =  mp(n)*(n+1) * (S2.S_n0(n+1) - mp(1)) - k2; % Advanpix mp operations
        R_coeffs.R_np2(n+1) =  mp(n)*(n+1) * S2.S_np2(n+1); % Advanpix mp multiplication
    end
end

function Q_coeffs = compute_Q_mp(S3, S1, Nmax, k2)
    Q_coeffs.Q_nm3 = zeros(Nmax, 1, 'mp'); % Advanpix mp array
    Q_coeffs.Q_nm1 = zeros(Nmax, 1, 'mp'); % Advanpix mp array
    Q_coeffs.Q_np1 = zeros(Nmax, 1, 'mp'); % Advanpix mp array
    Q_coeffs.Q_np3 = zeros(Nmax, 1, 'mp'); % Advanpix mp array
    for n = 0:Nmax-1
        Q_coeffs.Q_nm3(n+1) = (mp(n-1))*(n+2) * S3.S_nm3(n+1); % Advanpix mp multiplication
        Q_coeffs.Q_nm1(n+1) = (mp(n-1))*(n+2) * (S3.S_nm1(n+1) - S1.S_nm1(n+1)) - k2*S1.S_nm1(n+1); % Advanpix mp operations
        Q_coeffs.Q_np1(n+1) = (mp(n-1))*(n+2) * (S3.S_np1(n+1) - S1.S_np1(n+1)) - k2*S1.S_np1(n+1); % Advanpix mp operations
        Q_coeffs.Q_np3(n+1) = (mp(n-1))*(n+2) * S3.S_np3(n+1); % Advanpix mp multiplication
    end
end

function T_coeffs = compute_T_mp(S2, R_coeffs, Nmax, k2)
    T_coeffs.T_nm4 = zeros(Nmax, 1, 'mp'); % Advanpix mp array
    T_coeffs.T_nm2 = zeros(Nmax, 1, 'mp'); % Advanpix mp array
    T_coeffs.T_n0  = zeros(Nmax, 1, 'mp'); % Advanpix mp array
    T_coeffs.T_np2 = zeros(Nmax, 1, 'mp'); % Advanpix mp array
    T_coeffs.T_np4 = zeros(Nmax, 1, 'mp'); % Advanpix mp array
    for n = 0:Nmax-1
        Sm_nm2 = mp(0);
        S0_nm2 = mp(0);
        Sp_nm2 = mp(0);
        Sm_np2 = mp(0);
        S0_np2 = mp(0);
        Sp_np2 = mp(0);

        if n > 1
            Sm_nm2 = S2.S_nm2(n-1);
            S0_nm2 = S2.S_n0(n-1);
            Sp_nm2 = S2.S_np2(n-1);
        end
        if n + 2 < Nmax
            Sm_np2 = S2.S_nm2(n+3);
            S0_np2 = S2.S_n0(n+3);
            Sp_np2 = S2.S_np2(n+3);
        end

        T_coeffs.T_nm4(n+1) = (mp(n-2))*(n-1) * Sm_nm2       * R_coeffs.R_nm2(n+1); % Advanpix mp multiplication
        T_coeffs.T_nm2(n+1) = ((mp(n-2))*(n-1) * (S0_nm2 - mp(1)) - k2) * R_coeffs.R_nm2(n+1) + ... % Advanpix mp operations
                              mp(n)*(n+1) * S2.S_nm2(n+1)       * R_coeffs.R_n0(n+1); % Advanpix mp multiplication
        T_coeffs.T_n0(n+1)  = (mp(n-2))*(n-1) * Sp_nm2 * R_coeffs.R_nm2(n+1) + ... % Advanpix mp multiplication
                             (mp(n)*(n+1) * (S2.S_n0(n+1) - mp(1)) - k2) * R_coeffs.R_n0(n+1) + ... % Advanpix mp operations
                             (mp(n+2))*(n+3) * Sm_np2 * R_coeffs.R_np2(n+1); % Advanpix mp multiplication
        T_coeffs.T_np2(n+1) = (mp(n)*(n+1) * S2.S_np2(n+1) * R_coeffs.R_n0(n+1)  + ... % Advanpix mp multiplication
                             ((mp(n+2))*(n+3) * (S0_np2 - mp(1)) - k2) * R_coeffs.R_np2(n+1)); % Advanpix mp operations
        T_coeffs.T_np4(n+1) = (mp(n+2))*(n+3) * Sp_np2 * R_coeffs.R_np2(n+1); % Advanpix mp multiplication
    end
end
