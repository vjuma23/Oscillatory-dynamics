% =========================================================================
%  Turing and Floquet Stability Space Analysis
% =========================================================================
% By Victor Juma (vjuma23@gmail.com)
% DESCRIPTION:
% This script computes the stability boundaries for a Reaction-Diffusion (RD)
% system across a 2-parameter space (a1, a5). It identifies:
%   1. Uniform Steady States (USS).
%   2. Classical Turing Instability (Diffusion-driven instability of USS).
%   3. Oscillatory Dynamics (Limit Cycles).
%   4. Floquet-Turing Instability (Diffusion-driven instability of Limit Cycles).
%
% OUTPUTS:
%   - Matrices containing stability codes and critical diffusion values.
%   - Text files saved in a generated folder 'Turi_osc_spaces_a1_a5'.
%
% USAGE:
%   Adjust 'n_a' and 'n_b' for grid resolution.
%   Run the script to generate stability maps.
%   Requires Parallel Computing Toolbox (parfor).
% =========================================================================

clc; clear; close all;
format long g

%% Model Parameters
pp.a2 = 4.5;  pp.a3 = 0.1;   pp.a4 = 2.1053;
pp.a6 = 7.5;  pp.a8 = 7.5;   pp.a7 = 0.6075;  pp.gam = 1;

% Diffusion values to sweep
dd = [1, 10, 50, 60, 75, 100, 150, 200];

% Output setup
folder_name = fullfile(pwd, 'Turi_osc_spaces_a1_a5');
if ~exist(folder_name, 'dir'), mkdir(folder_name); end

% Grid setup
n_a = 1000; % Resolution for a1
n_b = 500;  % Resolution for a5
a1_vals = linspace(0.001, 26, n_a);
a5_vals = linspace(0.1, 10, n_b);

% Save axis values
writematrix(a1_vals, fullfile(folder_name, 'a1_vals.txt'));
writematrix(a5_vals, fullfile(folder_name, 'a5_vals.txt'));
writematrix(dd,     fullfile(folder_name, 'dd_vals.txt'));

n_dd = numel(dd);
wave_numbers = linspace(0, 7, 150); % Wavenumber range for instability check
opts = odeset('RelTol',1e-8,'AbsTol',1e-8);

% Flatten grid for parallel loop
numTotal = n_a * n_b;
[II, JJ] = ndgrid(1:n_a, 1:n_b);
ii_list = II(:);
jj_list = JJ(:);

% Preallocate results
stab3_vec = zeros(numTotal, n_dd);
stab5_vec = zeros(numTotal, n_dd);
Turing_vec = zeros(numTotal, n_dd);

fprintf('Starting parallel computation on %d grid points...\n', numTotal);

%% Main Parallel Loop
parfor idx = 1:numTotal
    % Progress print
    if idx == 1 || mod(idx,500) == 0 || idx == numTotal
        fprintf('Progress: %.1f%%\n', 100*idx/numTotal);
    end

    ii = ii_list(idx);
    jj = jj_list(idx);
    local_pp = pp;
    local_pp.a1 = a1_vals(ii);
    local_pp.a5 = a5_vals(jj);

    stab3_vals = zeros(1, n_dd);
    stab5_vals = zeros(1, n_dd);
    turing_vals = zeros(1, n_dd);

    for d_i = 1:n_dd
        d_val = dd(d_i);
        local_ppd = local_pp;

        % --- 1. Find Steady States using polynomial and count them ---
        pol = coef_pol(local_ppd);
        m_roots = roots(pol);
        m_roots = m_roots(imag(m_roots)==0 & m_roots>0 & m_roots<=local_ppd.a6);
        r_roots = (m_roots./(local_ppd.a8+m_roots)) .* ...
                  ((local_ppd.a7+local_ppd.a6-m_roots)./(local_ppd.a6-m_roots));
        my_roots = [r_roots, m_roots];
        my_roots = my_roots(my_roots(:,1)>0 & my_roots(:,1)<=local_ppd.a2, :);
        aaa = size(my_roots,1);

        traces = zeros(aaa,1);
        dets = zeros(aaa,1);
        stable_flags = false(aaa,1);
        turing_flags = false(aaa,1);
        satisfies_first_two_conditions = false(aaa,1);
        unstable_found = false;

        % --- 2. Check Linear Stability (Turing) ---
        for aai = 1:aaa
            R = my_roots(aai,1); M = my_roots(aai,2);
            J = jac_matrix_kamp(R, M, local_ppd);
            trJ = trace(J); detJ = det(J);

            traces(aai) = trJ;
            dets(aai)   = detJ;
            stable_flags(aai) = (trJ < 0) && (detJ > 0);
            if trJ > 0 && detJ > 0, unstable_found = true; end
            satisfies_first_two_conditions(aai) = (trJ < 0) && (detJ > 0);

            % Turing Conditions
            fu = (local_ppd.a1 * (local_ppd.a4*M*R^2 - local_ppd.a3*R^2 ...
                - 2*local_ppd.a2*local_ppd.a4*R*M - 2*local_ppd.a3*local_ppd.a4*R*M ...
                + local_ppd.a2^2*local_ppd.a4*M + local_ppd.a2*local_ppd.a3*local_ppd.a4*M)) ...
                / ((local_ppd.a4*M + R)^2 * (local_ppd.a3 + local_ppd.a2 - R)^2) ...
                - local_ppd.a5 / (local_ppd.a5 + R)^2;
            fv = -local_ppd.a1*local_ppd.a4*R*(local_ppd.a2-R) / ...
                ((local_ppd.a2 + local_ppd.a3 - R)*(local_ppd.a4*M + R)^2);
            gu = (local_ppd.a6 - M) / (local_ppd.a6 + local_ppd.a7 - M);
            gv = -local_ppd.a7*R / (local_ppd.a6 + local_ppd.a7 - M)^2 - local_ppd.a8/(local_ppd.a8 + M)^2;

            if satisfies_first_two_conditions(aai)
                if (d_val*fu + gv) > 0 && ((d_val*fu + gv)^2 > 4*d_val*(fu*gv - fv*gu))
                    turing_flags(aai) = true;
                end
            end
        end

        % Assign Turing Codes
        if aaa == 1
            if stable_flags(1)
                if turing_flags(1), turing_code = 4; else, turing_code = 6; end
            else
                turing_code = 0;
            end
        elseif aaa > 1
            if unstable_found
                turing_code = 0;
            else
                idx_check = find(satisfies_first_two_conditions);
                if isempty(idx_check)
                    turing_code = 7;
                else
                    if any(turing_flags(idx_check)), turing_code = 5; else, turing_code = 7; end
                end
            end
        else
            turing_code = 0;
        end
        turing_vals(d_i) = turing_code;

        % --- 3. Floquet Analysis (Limit Cycles) ---
        stab3_val = 0; stab5_val = 0;
        if aaa == 1 && any(~stable_flags)
            for aai = 1:aaa
                if ~stable_flags(aai)
                    stab3_val = floquet_analysis_region3(my_roots(aai,:), local_ppd, d_val, wave_numbers, opts);
                    break;
                end
            end
        elseif aaa > 1 && any(~stable_flags)
            for aai = 1:aaa
                if ~stable_flags(aai)
                    stab5_val = floquet_analysis_region5(my_roots(aai,:), local_ppd, d_val, wave_numbers, opts);
                    break;
                end
            end
        end
        stab3_vals(d_i) = stab3_val;
        stab5_vals(d_i) = stab5_val;
    end

    stab3_vec(idx, :) = stab3_vals;
    stab5_vec(idx, :) = stab5_vals;
    Turing_vec(idx, :) = turing_vals;
end

%% Saving Results
% Reshape back to 2D grid
stability_reg_3 = reshape(stab3_vec, [n_a, n_b, n_dd]);
stability_reg_5 = reshape(stab5_vec, [n_a, n_b, n_dd]);
Turing_vals     = reshape(Turing_vec, [n_a, n_b, n_dd]);

for d_i = 1:n_dd
    d_val = dd(d_i);
    fn3 = fullfile(folder_name, sprintf('stability_map_reg_3_Dv_%d.txt', d_val));
    fn5 = fullfile(folder_name, sprintf('stability_map_reg_5_Dv_%d.txt', d_val));
    fnT = fullfile(folder_name, sprintf('turing_map_d_%d.txt', d_val));
    writematrix(stability_reg_3(:,:,d_i), fn3);
    writematrix(stability_reg_5(:,:,d_i), fn5);
    writematrix(Turing_vals(:,:,d_i), fnT);
end

fprintf('\nComputation Complete. Results saved in %s\n', folder_name);


%% ========================================================================
%% LOCAL FUNCTIONS
%% ========================================================================

function result = floquet_analysis_region3(root, pp, d_val, wave_numbers, opts)
    % Analyzes stability of limit cycles for G.A.S. limit cycle
    R = root(1); M = root(2);

    % Step 1: Transient integration to approach limit cycle
    % NOTE: 'ode15s' is used here for stiff systems.
    % you may use 'ode23s' or any suitable. Users should adjust 'tspan' and time steps
    % to ensure the system has settled onto the limit cycle.
    tspan_prep = [0 500]; 
    x0_prep = [R; M] + 1e-2;
    try
        [~, x_prep] = ode15s(@(t, x) Nondim(t, x, pp), tspan_prep, x0_prep, opts);
        x0_cycle = x_prep(end, :).';
    catch
        result = 1; % ODE failed
        return;
    end

    % Step 2: Integrate to capture the full limit cycle
    % NOTE: For accurate spline interpolation in Floquet analysis, a dense output 
    % is often required. If using [0 100], ensure the solver takes enough steps, 
    % or explicitly define a time vector (e.g., 0:0.001:100).
    tspan_long = [0 500]; 
    try
        [t_long, x_long] = ode15s(@(t, x) Nondim(t, x, pp), tspan_long, x0_cycle, opts);
        [x_limit, t_limit, T_final, ~] = compute_limit_cycle(t_long, x_long);
    catch
        result = 1;
        return;
    end
    
    if isempty(x_limit) || isnan(T_final) || T_final < 0.1
        result = 1;
        return;
    end

    % Step 3: Interpolate solution on the limit cycle (if coarse time steps are used)
    [x_fun, y_fun] = spline_function(t_limit, x_limit);
    integration_time = linspace(t_limit(1), t_limit(end), 400);

    % or use directly t_limit as
    % integration_time = t_limit % if enough time resolution is used

    % Step 4: Check Floquet multipliers
    unstableFound = false;
    for w_idx = 1:numel(wave_numbers)
        k_val = wave_numbers(w_idx);
        max_eig_val = compute_wave_eig(k_val, x_fun, y_fun, pp, integration_time, opts, d_val);
        if max_eig_val > 1
            unstableFound = true;
            break;
        end
    end

    if ~unstableFound
        result = 2; % Stable Cycle
    else
        result = 3; % Unstable Cycle (FTDDI)
    end
end

function result = floquet_analysis_region5(root, pp, d_val, wave_numbers, opts)
    % Analyzes stability of limit cycles in L.A.S limit cycle (similar to  G.A.S.)
    R = root(1); M = root(2);

    % Step 1: Transient integration
    tspan_prep = [0 1000]; 
    x0_prep = [R; M] + 1e-2;
    try
        [~, x_prep] = ode15s(@(t, x) Nondim(t, x, pp), tspan_prep, x0_prep, opts);
        x0_cycle = x_prep(end, :).';
    catch
        result = 1; 
        return;
    end

    % Step 2: Cycle extraction
    tspan_long = [0 1000];
    try
        [t_long, x_long] = ode15s(@(t, x) Nondim(t, x, pp), tspan_long, x0_cycle, opts);
        [x_limit, t_limit, T_final, ~] = compute_limit_cycle(t_long, x_long);
    catch
        result = 1;
        return;
    end
    
    if isempty(x_limit) || isnan(T_final) || T_final < 0.1
        result = 1;
        return;
    end

    % Step 3: Interpolation
    [x_fun, y_fun] = spline_function(t_limit, x_limit);
    integration_time = linspace(t_limit(1), t_limit(end), 400);

    % or

    % integration_time = t_limit
    
    % Step 4: Floquet Multipliers
    unstableFound = false;
    for w_idx = 1:numel(wave_numbers)
        k_val = wave_numbers(w_idx);
        max_eig_val = compute_wave_eig(k_val, x_fun, y_fun, pp, integration_time, opts, d_val);
        if max_eig_val > 1
            unstableFound = true;
            break;
        end
    end

    if ~unstableFound
        result = 2; 
    else
        result = 3; 
    end
end

function max_eig_val = compute_wave_eig(k_val, x_fun, y_fun, pp, integration_time, opts, d_val)
    % Computes the maximum Floquet multiplier for a given wavenumber k
    if abs(k_val) < 1e-12
        max_eig_val = 0;
        return;
    end
    k_sqr = k_val^2;
    n = 2;
    Phi_0 = eye(n);
    Phi = Phi_0;
    
    % Integrate variational equations using ode15s (stiff solver)
    for col = 1:n
        [~, x_sol] = ode15s(@(tt, xx) kamp_linear_rd(tt, xx, x_fun, y_fun, pp, k_sqr, d_val), integration_time, Phi(:,col), opts);
        Phi(:,col) = x_sol(end, :)';
    end
    BB = Phi_0 \ Phi;
    eigs_val = abs(eig(BB));
    max_eig_val = max(eigs_val);
end

function [x_fun, y_fun] = spline_function(t_limit, x_limit)
    if size(x_limit, 2) ~= 2
        error('x_limit must be an Nx2 matrix with x and y coordinates.');
    end
    % Generate piecewise polynomial structures
    x_spline_pp = spline(t_limit, x_limit(:,1)); 
    y_spline_pp = spline(t_limit, x_limit(:,2)); 

    x_fun = @(t) ppval(x_spline_pp, t);
    y_fun = @(t) ppval(y_spline_pp, t);
end

function dydt = Nondim(~,y,pp)
    dydt=zeros(2,1);
    dydt(1)=pp.gam*((pp.a1*y(1)*(pp.a2-y(1)))/(((pp.a3+pp.a2-y(1))*(pp.a4*y(2)+y(1))))-y(1)/(pp.a5+y(1))); 
    dydt(2)=pp.gam*((y(1)*(pp.a6-y(2)))/(pp.a7+pp.a6-y(2))-y(2)/(pp.a8+y(2))); 
end

function Jac = jac_matrix_kamp(R, M, pp)
    fu=(pp.a1*(pp.a4*M*R^2-pp.a3*R^2-2*pp.a2*pp.a4*R*M-2*pp.a3*pp.a4*R*M+pp.a2^2*pp.a4*M+pp.a2*pp.a3*pp.a4*M))/...
        ((pp.a4*M+R)^2*(pp.a3+pp.a2-R)^2)-pp.a5/(pp.a5+R)^2;
    fv=-pp.a1*pp.a4*R*(pp.a2-R)/((pp.a2+pp.a3-R)*(pp.a4*M+R)^2);
    gu=(pp.a6-M)/(pp.a6+pp.a7-M);
    gv=-pp.a7*R/((pp.a6+pp.a7-M)^2)-pp.a8/(pp.a8+M)^2;
    Jac = pp.gam*[fu,  fv; gu,   gv];
end

function Xdot = kamp_linear_rd(t, x, x_fun, y_fun, pp, k_squared, d_val)
    xx = x_fun(t); yy = y_fun(t);
    J_reaction = jac_matrix_kamp(xx,yy,pp);
    D_matrix = -k_squared * [1, 0; 0, d_val];
    J_total = J_reaction + D_matrix;
    Xdot = J_total * x;
end

function [x_limit, t_limit, T_final, deviation] = compute_limit_cycle(t, x)
    if any(x(:,1) < 0) || any(x(:,2) < 0)
        x_limit = []; t_limit = []; T_final = NaN; deviation = NaN;
        return;
    end

    [~, peak_indices1] = findpeaks(x(:,1), 'MinPeakDistance', 5);
    [~, peak_indices2] = findpeaks(x(:,2), 'MinPeakDistance', 5);
    peak_times1 = t(peak_indices1);
    peak_times2 = t(peak_indices2);

    if length(peak_times1) < 2 || length(peak_times2) < 2
        x_limit = []; t_limit = []; T_final = NaN; deviation = NaN;
        return;
    end

    T_final1 = mean(diff(peak_times1(2:end)));
    T_final2 = mean(diff(peak_times2(2:end)));

    % Extract cycle based on peak times
    [x_limit1, t_limit1, dev1] = extract_cycle_segment(t, x, peak_times1);
    [x_limit2, t_limit2, dev2] = extract_cycle_segment(t, x, peak_times2);

    if dev1 < dev2
        x_limit = x_limit1; t_limit = t_limit1; T_final = T_final1; deviation = dev1;
    else
        x_limit = x_limit2; t_limit = t_limit2; T_final = T_final2; deviation = dev2;
    end
end

function [x_seg, t_seg, dev] = extract_cycle_segment(t, x, peaks)
    t_start = peaks(end-1);
    t_end = peaks(end);
    idx_start = find(t >= t_start, 1, 'first');
    idx_end = find(t <= t_end, 1, 'last');
    x_seg = x(idx_start:idx_end, :);
    t_seg = t(idx_start:idx_end) - t(idx_start);
    dev = norm(x_seg(1,:) - x_seg(end,:));
end

function out = coef_pol(pp)
    % Coefficients for characteristic polynomial of the steady state
    a1 = pp.a1; a2 = pp.a2; a3 = pp.a3; a4 = pp.a4; 
    a5 = pp.a5; a6 = pp.a6; a7 = pp.a7; a8 = pp.a8;

    AA1 = a2 * a4 + a3 * a4 - a4;
    AA2 = (-a1*a2*a5 - 3*a2*a4*a6 - a2*a4*a7 + 2*a2*a4*a8 - 3*a3*a4*a6 ...
        - a3*a4*a7 + 2*a3*a4*a8 - a1*a2 + a1*a5 + 3*a4*a6 + 2*a4*a7 - a4*a8 ...
        + a1 + a2 + a3 - 1);
    AA3 = (3*a1*a2*a5*a6 + a1*a2*a5*a7 - 2*a1*a2*a5*a8 + 3*a2*a4*a6^2 ...
        + 2*a2*a4*a6*a7 - 6*a2*a4*a6*a8 - 2*a2*a4*a7*a8 + a2*a4*a8^2 ...
        + 3*a3*a4*a6^2 + 2*a3*a4*a6*a7 - 6*a3*a4*a6*a8 - 2*a3*a4*a7*a8 ...
        + a3*a4*a8^2 + 3*a1*a2*a6 + 2*a1*a2*a7 - a1*a2*a8 - 3*a1*a5*a6 ...
        - 2*a1*a5*a7 + a1*a5*a8 - 3*a4*a6^2 - 4*a4*a6*a7 ...
        + 3*a4*a6*a8 - a4*a7^2 + 2*a4*a7*a8 - 3*a1*a6 - 3*a1*a7 ...
        - 3*a2*a6 - 2*a2*a7 + a2*a8 - 3*a3*a6 - 2*a3*a7 ...
        + a3*a8 + 3*a6 + 3*a7);
    AA4 = (-3*a1*a2*a5*a6^2 - 2*a1*a2*a5*a6*a7 + 6*a1*a2*a5*a6*a8 ...
        + 2*a1*a2*a5*a7*a8 - a1*a2*a5*a8^2 - a2*a4*a6^3 - a2*a4*a6^2*a7 ...
        + 6*a2*a4*a6^2*a8 + 4*a2*a4*a6*a7*a8 - 3*a2*a4*a6*a8^2 ...
        - a2*a4*a7*a8^2 - a3*a4*a6^3 - a3*a4*a6^2*a7 + 6*a3*a4*a6^2*a8 ...
        + 4*a3*a4*a6*a7*a8 - 3*a3*a4*a6*a8^2 - a3*a4*a7*a8^2 - 3*a1*a2*a6^2 ...
        - 4*a1*a2*a6*a7 + 3*a1*a2*a6*a8 - a1*a2*a7^2 + 2*a1*a2*a7*a8 ...
        + 3*a1*a5*a6^2 + 4*a1*a5*a6*a7 - 3*a1*a5*a6*a8 + a1*a5*a7^2 ...
        - 2*a1*a5*a7*a8 + a4*a6^3 + 2*a4*a6^2*a7 - 3*a4*a6^2*a8 ...
        + a4*a6*a7^2 - 4*a4*a6*a7*a8 - a4*a7^2*a8 + 3*a1*a6^2 ...
        + 6*a1*a6*a7 + 3*a1*a7^2 + 3*a2*a6^2 + 4*a2*a6*a7 - 3*a2*a6*a8 ...
        + a2*a7^2 - 2*a2*a7*a8 + 3*a3*a6^2 + 4*a3*a6*a7 - 3*a3*a6*a8 ...
        + a3*a7^2 - 2*a3*a7*a8 - 3*a6^2 - 6*a6*a7 - 3*a7^2);
    AA5 = (a1*a2*a5*a6^3 + a1*a2*a5*a6^2*a7 - 6*a1*a2*a5*a6^2*a8 ...
        - 4*a1*a2*a5*a6*a7*a8 + 3*a1*a2*a5*a6*a8^2 ...
        + a1*a2*a5*a7*a8^2 - 2*a2*a4*a6^3*a8 ...
        - 2*a2*a4*a6^2*a7*a8 + 3*a2*a4*a6^2*a8^2 + 2*a2*a4*a6*a7*a8^2 ...
        - 2*a3*a4*a6^3*a8 - 2*a3*a4*a6^2*a7*a8 + 3*a3*a4*a6^2*a8^2 ...
        + 2*a3*a4*a6*a7*a8^2 + a1*a2*a6^3 + 2*a1*a2*a6^2*a7 ...
        - 3*a1*a2*a6^2*a8 + a1*a2*a6*a7^2 - 4*a1*a2*a6*a7*a8 ...
        - a1*a2*a7^2*a8 - a1*a5*a6^3 - 2*a1*a5*a6^2*a7 + 3*a1*a5*a6^2*a8 ...
        - a1*a5*a6*a7^2 + 4*a1*a5*a6*a7*a8 + a1*a5*a7^2*a8 + a4*a6^3*a8 ...
        + 2*a4*a6^2*a7*a8 + a4*a6*a7^2*a8 - a1*a6^3 - 3*a1*a6^2*a7 ...
        - 3*a1*a6*a7^2 - a1*a7^3 - a2*a6^3 - 2*a2*a6^2*a7 + 3*a2*a6^2*a8 ...
        - a2*a6*a7^2 + 4*a2*a6*a7*a8 + a2*a7^2*a8 - a3*a6^3 - 2*a3*a6^2*a7 ...
        + 3*a3*a6^2*a8 - a3*a6*a7^2 + 4*a3*a6*a7*a8 + a3*a7^2*a8 ...
        + a6^3 + 3*a6^2*a7 + 3*a6*a7^2 + a7^3);
    AA6 = (2*a1*a2*a5*a6^3*a8 + 2*a1*a2*a5*a6^2*a7*a8 ...
        - 3*a1*a2*a5*a6^2*a8^2 - 2*a1*a2*a5*a6*a7*a8^2 ...
        - a2*a4*a6^3*a8^2 - a2*a4*a6^2*a7*a8^2 - a3*a4*a6^3*a8^2 ...
        - a3*a4*a6^2*a7*a8^2 + a1*a2*a6^3*a8 + 2*a1*a2*a6^2*a7*a8 ...
        + a1*a2*a6*a7^2*a8 - a1*a5*a6^3*a8 - 2*a1*a5*a6^2*a7*a8 ...
        - a1*a5*a6*a7^2*a8 - a2*a6^3*a8 - 2*a2*a6^2*a7*a8 ...
        - a2*a6*a7^2*a8 - a3*a6^3*a8 - 2*a3*a6^2*a7*a8 - a3*a6*a7^2*a8);
    AA7 = (a1*a2*a5*a6^3*a8^2 + a1*a2*a5*a6^2*a7*a8^2);
    AA8 = 0;

    out = [AA1, AA2, AA3, AA4, AA5, AA6, AA7, AA8];
end