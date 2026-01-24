% =========================================================================
% SCRIPT: Critical Diffusion Computation (Hybrid Method)
% =========================================================================
% DESCRIPTION:
% This script calculates the critical diffusion coefficient (d_c) as 
% a function of a parameter (here a1) that is required to
% destabilize a limit cycle via Floquet-Turing instability (FTDDI).
%
% METHODOLOGY (HYBRID APPROACH):
%   1. Estimate d_c using the Averaged Jacobian Method (AJM). This provides 
%      a fast but approximate starting point.
%   2. Refine the estimate using exact Floquet Multiplier analysis. 
%      - Search Window: The AJM result is used to narrow the search range.
%      - Root Finding: The script employs the "Secant Method" to iteratively 
%        find the diffusion value where the maximum Floquet multiplier 
%        equals a specific threshold (here threshold =1).
%      - Dynamic Thresholding: To ensure convergence, the algorithm starts 
%        with a relaxed threshold (e.g., 1.014) and iteratively reduces it 
%        towards the exact instability boundary (1.0) using the Secant 
%        updates at each step.
%
% OUTPUTS:
%   - A text file 'all_critical_d_hybrid_refined.txt' containing:
%     [Column 1: Parameter a1, Column 2: AJM Estimate, Column 3: Floquet Result]
%
% USAGE:
%   - Adjust 'a1_vals' to set the parameter range for the scan.
%   - Adjust 'pp' structure to set fixed model parameters.
%   - Requires Parallel Computing Toolbox (parfor).
%
% AUTHOR: Victor Juma (vjuma23@gmail.com)
% DATE updated: January 23, 2026
% =========================================================================

clear; clc; close all
format long g

% Start parallel pool if not already started
if isempty(gcp('nocreate'))
    parpool; 
end

%% ========================================================================
%% 1. USER SETTINGS & SETUP
%% ========================================================================

% --- Algorithm Settings ---
initial_floq_threshold = 1.014;  % Starting Critical threshold for instability detection
threshold_step = 0.001;          % Step size to reduce threshold during refinement
max_refine_iters = 30;           % Max number of reduction steps
search_width = 5;                % Width of search window around the AJM estimate (+/-)

% --- Model fixed parameters ---
pp.a1=5;  pp.a2=4.5; pp.a3=0.1; pp.a4=2.1053;
pp.a6=7.5; pp.a8=7.5;
pp.a5=1.5; pp.a7=0.6075; pp.gam=1; 

% --- Grid Definitions ---
% Wavenumber grid for stability check
n_nva = 200;
nn_vals = linspace(0, 4, n_nva);

% Parameter scan range (a1)
a1_vals = linspace(3.965235, 7, 250); 
n_a1 = length(a1_vals);

% --- Output Setup ---
folder_name = fullfile(pwd, 'd_crit_hybrid_method');
if ~exist(folder_name, 'dir')
    mkdir(folder_name);
end

% Save axis grids for reference
writematrix(a1_vals, fullfile(folder_name, 'a1_vals.txt'));
writematrix(nn_vals, fullfile(folder_name, 'nn_vals.txt'));

fprintf('Starting parallel computation over %d a1 values...\n', n_a1);

% Pre-allocate result matrix
results_matrix = NaN(n_a1, 3);

%% ========================================================================
%% 2. MAIN PARALLEL LOOP
%% ========================================================================
parfor i = 1:n_a1
    % Local parameter structure for parallel worker
    pp_loc = pp;
    pp_loc.a1 = a1_vals(i);
    
    % ---------------------------------------------------------------------
    % Step A: Find Steady State & Check Instability
    % ---------------------------------------------------------------------
    pol = coef_pol(pp_loc);
    m_roots  = roots(pol);
    % Filter for real, positive roots within physical bounds
    m_roots  = m_roots(imag(m_roots)==0 & m_roots>0 & m_roots<=pp_loc.a6);
    r_roots  = (m_roots./(pp_loc.a8+m_roots)) .* (pp_loc.a7+pp_loc.a6-m_roots)./(pp_loc.a6-m_roots);
    my_roots = [r_roots,m_roots];
    my_roots = my_roots(my_roots(:,1)>0 & my_roots(:,1)<=pp_loc.a2,:);
    
    num_valid_roots = size(my_roots, 1);
    Ru = []; Rv = [];
    found_unstable_root = false;
    
    % Identify the unstable steady state (origin of limit cycle)
    if num_valid_roots == 1
        Ru_temp = my_roots(1,1); Rv_temp = my_roots(1,2);
        JJ = jac_matrix_kamp(Ru_temp, Rv_temp, pp_loc);
        if any(real(eig(JJ)) >= 0) 
            Ru = Ru_temp; Rv = Rv_temp;
            found_unstable_root = true;
        end
    elseif num_valid_roots > 1
        for root_idx = 1:num_valid_roots
            RR = my_roots(root_idx,1); MM = my_roots(root_idx,2);
            JJ = jac_matrix_kamp(RR, MM, pp_loc);
            % Look for trace > 0, det > 0 (Unstable Node/Spiral)
            if trace(JJ) > 0 && det(JJ) > 0
                Ru = RR; Rv = MM;
                found_unstable_root = true;
                break;
            end
        end
    end
    
    if ~found_unstable_root
        % fprintf('Skipping a1=%g (No suitable unstable root)\n', pp_loc.a1);
        continue;
    end
    
    % ---------------------------------------------------------------------
    % Step B: Compute Base Limit Cycle
    % ---------------------------------------------------------------------
    opts = odeset('RelTol',1e-10,'AbsTol',1e-10);
    x0_prep = [Ru; Rv] + 1e-2;
    
    % Transient phase
    [~, x_prep] = ode23s(@(t,x) Nondim(t, x, pp_loc), [0:0.005:500], x0_prep, opts);
    x0 = x_prep(end, :).';
    
    % Full cycle capture
    [t_full, x_full] = ode15s(@(t,x) Nondim(t, x, pp_loc), [0:0.001:200], x0, opts);
    [x_limit, t_limit, T_period, ~] = compute_limit_cycle(t_full, x_full);
    
    if isempty(x_limit)
        continue;
    end
    
    % Create splines for Floquet evaluation
    [x_fun, y_fun] = spline_function(t_limit, x_limit); % optional
    time = t_limit;
    
    % ---------------------------------------------------------------------
    % Step C: ESTIMATE Critical d using Averaged Jacobian (AJM)
    % ---------------------------------------------------------------------
    % Calculate Jacobian elements along the cycle
    len_t = length(t_limit);
    J_vals = zeros(len_t, 4); % Columns: J11, J12, J21, J22
    
    for k = 1:len_t
        J_inst = jac_matrix_kamp(x_limit(k,1), x_limit(k,2), pp_loc);
        J_vals(k,:) = [J_inst(1,1), J_inst(1,2), J_inst(2,1), J_inst(2,2)];
    end
    
    % Integrate to get averaged matrix coefficients
    f1R = trapz(t_limit, J_vals(:,1)) / T_period;
    f1M = trapz(t_limit, J_vals(:,2)) / T_period;
    f2R = trapz(t_limit, J_vals(:,3)) / T_period;
    f2M = trapz(t_limit, J_vals(:,4)) / T_period;
    
    % Solve quadratic condition for Turing instability on averaged system
    AA0 = (f1R)^2;
    AA1 = (4*f1M*f2R - 2*f1R*f2M);
    AA2 = (f2M)^2;
    d_roots = roots([AA0, AA1, AA2]);
    valid_d = d_roots(imag(d_roots)==0 & d_roots>0);
    
    d_avg = NaN;
    current_search_bounds = [5, 50]; % Default bounds if AJM fails
    
    if ~isempty(valid_d)
        d_avg = max(valid_d);
        % Set search window around AJM estimate for refinement
        d_lower = max(1, d_avg - search_width); 
        d_upper = d_avg + search_width;
        current_search_bounds = [d_lower, d_upper];
    end

    % ---------------------------------------------------------------------
    % Step D: REFINE Critical d using Exact Floquet Analysis
    % ---------------------------------------------------------------------
    current_threshold = initial_floq_threshold;
    found_d_values = []; % Store valid d's found across iterations
    
    for iter = 0:max_refine_iters
        if current_threshold <= 1.0, break; end
        
        % Use Secant Method to find d where max_floquet == current_threshold
        [d_found, success_flag] = solve_secant_for_threshold(current_threshold, ...
            current_search_bounds, nn_vals, x_fun, y_fun, time, pp_loc, opts);
        
        if success_flag
            found_d_values = [found_d_values; d_found];
            
            % Narrow the search bounds for the next, stricter threshold
            d_lower = max(1, d_found - 1);
            d_upper = d_found + 1;
            current_search_bounds = [d_lower, d_upper];
            
            % Decrease threshold to approach 1.0
            current_threshold = current_threshold - threshold_step;
        else
            break; % Stop if solver fails to converge
        end
    end
    
    % ---------------------------------------------------------------------
    % Step E: Store Results
    % ---------------------------------------------------------------------
    if ~isempty(found_d_values)
        crit_d = min(found_d_values); % The lowest valid d found
        fprintf('a1=%g: Avg=%.2f -> Final Floquet d_c=%.4f\n', pp_loc.a1, d_avg, crit_d);
        results_matrix(i, :) = [pp_loc.a1, d_avg, crit_d];
    else
        fprintf('a1=%g: No critical d found.\n', pp_loc.a1);
        results_matrix(i, :) = [pp_loc.a1, d_avg, NaN];
    end
end

%% ========================================================================
%% 3. SAVE FINAL DATA
%% ========================================================================
clean_results = results_matrix(~any(isnan(results_matrix), 2), :);
outFile = fullfile(folder_name, 'all_critical_d_hybrid_refined.txt');
writematrix(clean_results, outFile);

disp('Calculation complete.');


%% ========================================================================
%% LOCAL FUNCTIONS
%% ========================================================================

function [crit_d, success] = solve_secant_for_threshold(target_thresh, bounds, nn_vals, x_fun, y_fun, time, pp, opts)
    % Secant Method to find 'd' such that max_floquet(d) - target = 0
    success = false; crit_d = NaN;
    
    d_prev = bounds(1);
    d_curr = bounds(2);
    
    floq_prev = get_max_floquet(d_prev, nn_vals, x_fun, y_fun, time, pp, opts);
    err_prev = floq_prev - target_thresh;
    
    floq_curr = get_max_floquet(d_curr, nn_vals, x_fun, y_fun, time, pp, opts);
    err_curr = floq_curr - target_thresh;
    
    if err_prev * err_curr > 0
        %  try widening the bracket once if failed
        if err_prev > 0 && err_curr > 0
             d_prev = 5; 
             floq_prev = get_max_floquet(d_prev, nn_vals, x_fun, y_fun, time, pp, opts);
             err_prev = floq_prev - target_thresh;
        elseif err_prev < 0 && err_curr < 0
             d_curr = 80;
             floq_curr = get_max_floquet(d_curr, nn_vals, x_fun, y_fun, time, pp, opts);
             err_curr = floq_curr - target_thresh;
        end
    end
    
    if err_prev * err_curr > 0, return; end % Still failed to bracket
    
    % Secant Iteration
    for iter = 1:100
        if abs(d_curr - d_prev) < 1e-6, break; end
        
        slope = (err_curr - err_prev) / (d_curr - d_prev);
        if slope == 0
            d_next = (d_prev + d_curr)/2;
        else
            d_next = d_curr - err_curr / slope;
        end
        
        % Safety bounds
        if d_next < 1, d_next = 1; end
        if d_next > 200, d_next = 200; end
        
        floq_next = get_max_floquet(d_next, nn_vals, x_fun, y_fun, time, pp, opts);
        err_next = floq_next - target_thresh;
        
        if abs(err_next) < 1e-5 || abs(d_next - d_curr) < 1e-5
            crit_d = d_next;
            success = true;
            break;
        end
        
        % Update bracket
        if err_curr * err_next < 0
            d_prev = d_curr; err_prev = err_curr;
            d_curr = d_next; err_curr = err_next;
        else
            d_curr = d_next; err_curr = err_next;
        end
    end
end

function max_floq = get_max_floquet(d_val, nn_vals, x_fun, y_fun, time, pp, opts)
    % Compute the maximum Floquet multiplier across all wavenumbers for given d
    pp.Dv = d_val; 
    max_floq = 0;
    n_nva = length(nn_vals);
    
    for jw = 1:n_nva
        k_val = nn_vals(jw);
        k_sqr  = (k_val * pi)^2;
        Phi    = eye(2);
        
        % Integrate Monodromy Matrix
        for col = 1:2
            [~, x_sol] = ode15s(@(tt, xx) kamp_linear_rd(tt, xx, ...
                x_fun, y_fun, pp, k_sqr), ...
                time, Phi(:,col), opts);
            Phi(:,col) = x_sol(end, :)';
        end
        
        max_eig_k  = max(abs(eig(Phi)));
        if max_eig_k > max_floq
            max_floq = max_eig_k;
        end
    end
end

%% --- ODE Functions ---

function dydt=Nondim(~,y,pp)
    dydt=zeros(2,1);
    dydt(1)=pp.gam*((pp.a1*y(1)*(pp.a2-y(1)))/(((pp.a3+pp.a2-y(1))*(pp.a4*y(2)+y(1))))-y(1)/(pp.a5+y(1))); 
    dydt(2)=pp.gam*((y(1)*(pp.a6-y(2)))/(pp.a7+pp.a6-y(2))-y(2)/(pp.a8+y(2))); 
end

function Xdot = kamp_linear_rd(t, x, x_fun, y_fun, pp, k_squared)
    xx = x_fun(t);
    yy = y_fun(t);
    J_reaction = jac_matrix_kamp(xx,yy,pp);
    D_matrix = -k_squared * [1, 0; 0, pp.Dv];
    J_total = J_reaction + D_matrix;
    Xdot = J_total * x;
end

function [x_limit, t_limit, T_final,deviation] = compute_limit_cycle(t, x)
    if any(x(:,1) < 0) || any(x(:,2) < 0)
        x_limit = []; t_limit = []; T_final = NaN; deviation = NaN; return;
    end
    [~, peak_indices1] = findpeaks(x(:,1), 'MinPeakDistance', 5);
    peak_times1 = t(peak_indices1);
    [~, peak_indices2] = findpeaks(x(:,2), 'MinPeakDistance', 5);
    peak_times2 = t(peak_indices2);
    
    if length(peak_times1) < 2 || length(peak_times2) < 2
        x_limit = []; t_limit = []; T_final = NaN; deviation = NaN; return;
    end
    
    T_final1 = mean(diff(peak_times1(2:end)));
    T_final2 = mean(diff(peak_times2(2:end)));
    
    t_start1 = peak_times1(end-1); t_end1 = peak_times1(end);
    idx_start1 = find(t >= t_start1, 1, 'first'); idx_end1 = find(t <= t_end1, 1, 'last');
    x_limit1 = x(idx_start1:idx_end1, :); t_limit1 = t(idx_start1:idx_end1) - t(idx_start1);
    
    t_start2 = peak_times2(end-1); t_end2 = peak_times2(end);
    idx_start2 = find(t >= t_start2, 1, 'first'); idx_end2 = find(t <= t_end2, 1, 'last');
    x_limit2 = x(idx_start2:idx_end2, :); t_limit2 = t(idx_start2:idx_end2) - t(idx_start2);
    
    deviation1 = norm(x_limit1(1,:) - x_limit1(end,:));
    deviation2 = norm(x_limit2(1,:) - x_limit2(end,:));
    
    if deviation1 < deviation2
        x_limit = x_limit1; t_limit = t_limit1; T_final = T_final1; deviation=deviation1;
    else
        x_limit = x_limit2; t_limit = t_limit2; T_final = T_final2; deviation=deviation2;
    end
end

function [x_fun, y_fun] = spline_function(t_limit, x_limit)
    x_spline_pp = spline(t_limit, x_limit(:,1)); 
    y_spline_pp = spline(t_limit, x_limit(:,2)); 
    x_fun = @(t) ppval(x_spline_pp, t);
    y_fun = @(t) ppval(y_spline_pp, t);
end

function Jac = jac_matrix_kamp(R, M, pp)
    fu=(pp.a1*(pp.a4*M*R^2-pp.a3*R^2-2*pp.a2*pp.a4*R*M-2*pp.a3*pp.a4*R*M+pp.a2^2*pp.a4*M+pp.a2*pp.a3*pp.a4*M))/...
        ((pp.a4*M+R)^2*(pp.a3+pp.a2-R)^2)-pp.a5/(pp.a5+R)^2;
    fv=-pp.a1*pp.a4*R*(pp.a2-R)/((pp.a2+pp.a3-R)*(pp.a4*M+R)^2);
    gu=(pp.a6-M)/(pp.a6+pp.a7-M);
    gv=-pp.a7*R/((pp.a6+pp.a7-M)^2)-pp.a8/(pp.a8+M)^2;
    Jac = pp.gam*[fu,  fv; gu,   gv];
end

function out = coef_pol(pp)
    a1 = pp.a1; a2 = pp.a2; a3 = pp.a3; a4 = pp.a4; a5 = pp.a5; a6 = pp.a6; a7 = pp.a7; a8 = pp.a8;
    AA1 = a2 * a4 + a3 * a4 - a4; 
    AA2 = ( -a1*a2*a5 - 3*a2*a4*a6 - a2*a4*a7 + 2*a2*a4*a8 - 3*a3*a4*a6 - a3*a4*a7 + 2*a3*a4*a8 - a1*a2 + a1*a5 + 3*a4*a6 + 2*a4*a7 - a4*a8 + a1 + a2 + a3 - 1);
    AA3 = ( 3*a1*a2*a5*a6 + a1*a2*a5*a7 - 2*a1*a2*a5*a8 + 3*a2*a4*a6^2 + 2*a2*a4*a6*a7 - 6*a2*a4*a6*a8 - 2*a2*a4*a7*a8 + a2*a4*a8^2 + 3*a3*a4*a6^2 + 2*a3*a4*a6*a7 - 6*a3*a4*a6*a8 - 2*a3*a4*a7*a8 + a3*a4*a8^2 + 3*a1*a2*a6 + 2*a1*a2*a7 - a1*a2*a8 - 3*a1*a5*a6 - 2*a1*a5*a7 + a1*a5*a8 - 3*a4*a6^2 - 4*a4*a6*a7 + 3*a4*a6*a8 - a4*a7^2 + 2*a4*a7*a8 - 3*a1*a6 - 3*a1*a7 - 3*a2*a6 - 2*a2*a7 + a2*a8 - 3*a3*a6 - 2*a3*a7 + a3*a8 + 3*a6 + 3*a7);
    AA4 = ( -3*a1*a2*a5*a6^2 - 2*a1*a2*a5*a6*a7 + 6*a1*a2*a5*a6*a8 + 2*a1*a2*a5*a7*a8 - a1*a2*a5*a8^2 - a2*a4*a6^3 - a2*a4*a6^2*a7 + 6*a2*a4*a6^2*a8 + 4*a2*a4*a6*a7*a8 - 3*a2*a4*a6*a8^2 - a2*a4*a7*a8^2 - a3*a4*a6^3 - a3*a4*a6^2*a7 + 6*a3*a4*a6^2*a8 + 4*a3*a4*a6*a7*a8 - 3*a3*a4*a6*a8^2 - a3*a4*a7*a8^2 - 3*a1*a2*a6^2 - 4*a1*a2*a6*a7 + 3*a1*a2*a6*a8 - a1*a2*a7^2 + 2*a1*a2*a7*a8 + 3*a1*a5*a6^2 + 4*a1*a5*a6*a7 - 3*a1*a5*a6*a8 + a1*a5*a7^2 - 2*a1*a5*a7*a8 + a4*a6^3 + 2*a4*a6^2*a7 - 3*a4*a6^2*a8 + a4*a6*a7^2 - 4*a4*a6*a7*a8 - a4*a7^2*a8 + 3*a1*a6^2 + 6*a1*a6*a7 + 3*a1*a7^2 + 3*a2*a6^2 + 4*a2*a6*a7 - 3*a2*a6*a8 + a2*a7^2 - 2*a2*a7*a8 + 3*a3*a6^2 + 4*a3*a6*a7 - 3*a3*a6*a8 + a3*a7^2 - 2*a3*a7*a8 - 3*a6^2 - 6*a6*a7 - 3*a7^2);
    AA5 = ( a1*a2*a5*a6^3 + a1*a2*a5*a6^2*a7 - 6*a1*a2*a5*a6^2*a8 - 4*a1*a2*a5*a6*a7*a8 + 3*a1*a2*a5*a6*a8^2 + a1*a2*a5*a7*a8^2 - 2*a2*a4*a6^3*a8 - 2*a2*a4*a6^2*a7*a8 + 3*a2*a4*a6^2*a8^2 + 2*a2*a4*a6*a7*a8^2 - 2*a3*a4*a6^3*a8 - 2*a3*a4*a6^2*a7*a8 + 3*a3*a4*a6^2*a8^2 + 2*a3*a4*a6*a7*a8^2 + a1*a2*a6^3 + 2*a1*a2*a6^2*a7 - 3*a1*a2*a6^2*a8 + a1*a2*a6*a7^2 - 4*a1*a2*a6*a7*a8 - a1*a2*a7^2*a8 - a1*a5*a6^3 - 2*a1*a5*a6^2*a7 + 3*a1*a5*a6^2*a8 - a1*a5*a6*a7^2 + 4*a1*a5*a6*a7*a8 + a1*a5*a7^2*a8 + a4*a6^3*a8 + 2*a4*a6^2*a7*a8 + a4*a6*a7^2*a8 - a1*a6^3 - 3*a1*a6^2*a7 - 3*a1*a6*a7^2 - a1*a7^3 - a2*a6^3 - 2*a2*a6^2*a7 + 3*a2*a6^2*a8 - a2*a6*a7^2 + 4*a2*a6*a7*a8 + a2*a7^2*a8 - a3*a6^3 - 2*a3*a6^2*a7 + 3*a3*a6^2*a8 - a3*a6*a7^2 + 4*a3*a6*a7*a8 + a3*a7^2*a8 + a6^3 + 3*a6^2*a7 + 3*a6*a7^2 + a7^3);
    AA6 = ( 2*a1*a2*a5*a6^3*a8 + 2*a1*a2*a5*a6^2*a7*a8 - 3*a1*a2*a5*a6^2*a8^2 - 2*a1*a2*a5*a6*a7*a8^2 - a2*a4*a6^3*a8^2 - a2*a4*a6^2*a7*a8^2 - a3*a4*a6^3*a8^2 - a3*a4*a6^2*a7*a8^2 + a1*a2*a6^3*a8 + 2*a1*a2*a6^2*a7*a8 + a1*a2*a6*a7^2*a8 - a1*a5*a6^3*a8 - 2*a1*a5*a6^2*a7*a8 - a1*a5*a6*a7^2*a8 - a2*a6^3*a8 - 2*a2*a6^2*a7*a8 - a2*a6*a7^2*a8 - a3*a6^3*a8 - 2*a3*a6^2*a7*a8 - a3*a6*a7^2*a8);
    AA7 = ( a1*a2*a5*a6^3*a8^2 + a1*a2*a5*a6^2*a7*a8^2);
    AA8 = 0;
    out = [AA1, AA2, AA3, AA4, AA5, AA6, AA7, AA8];
end