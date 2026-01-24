% =========================================================================
% SCRIPT: FTDDI vs. Averaged Jacobian Method 
% =========================================================================
% DESCRIPTION:
% This script performs  stability analysis of limit cycles using
% two distinct methods:
%   1. Floquet-Turing Diffusion-Driven Instability (FTDDI).
%   2. Averaged Jacobian Method (AJM) - Approximate method.
%
% METHODOLOGY:
%   - The script iterates through specific instability regions (labelled 3 & 5).
%   - It fixes the kinetic parameters and performs a stability varying:
%       * Gamma (scaling parameter)
%       * Diffusion Ratio (d)
%       * Wavenumber (k)
%   - For each parameter set, it computes the limit cycle and then calculates
%     the maximum stability metric (Floquet multiplier for FTDDI, Real part 
%     of eigenvalue for AJM).
%
% OUTPUTS:
%   - Generates folders per region  containing:
%       * 'large_floq_d_gam_*.txt': Max instability metric vs Diffusion (Max over all k).
%       * 'large_floq_d_k_gam_*.txt': Instability metric for each Diffusion/Wavenumber pair.
%
% USAGE:
%   - Ensure 'region_list' contains the regions you wish to analyze [3, 5].
%   Can be done for a fixed parameter set (remove one region)
%   - Adjust 'gam_vals' and 'd_vals' for grid resolution.
%   - Requires Parallel Computing Toolbox (parfor).
%
% AUTHOR: Victor Juma (vjuma23@gmail.com)
% DATE updated: January 23, 2026
% =========================================================================

function FTDDI_and_AJM_fixed_param()
    % Main function
    clc; close all;
    format long g

    % Optional: Start parallel pool
    if isempty(gcp('nocreate'))
        parpool;  
    end

    % -------------------------------------------------------------------------
    % Master Loop: Iterate through both Stability Methods
    % -------------------------------------------------------------------------
    methods_list = {'averaged','floquet'};

    for m_idx = 1:length(methods_list)
        stability_method = methods_list{m_idx};
        fprintf('\n=================================================\n');
        fprintf('STARTING ANALYSIS USING: %s\n', upper(stability_method));
        fprintf('=================================================\n');

        % ---------------------------------------------------------------------
        % Loop through Regions (3 and 5)
        % ---------------------------------------------------------------------
        region_list = [3, 5];
        for reg = region_list
            
            % --- fixed parameters ---
            pp.a1=5;  pp.a2=4.5; pp.a3=0.1; pp.a4=2.1053;
            pp.a6=7.5; pp.a8=7.5;
            pp.a5=1.5; pp.a7=0.6075; pp.gam=1;

            % Set specific parameters for the Region
            if reg == 3
                pp.a1 = 4; pp.a5 = 1.5;
                reg_tag = 'reg_3';
                reg_prefix = 'osc_reg_3_fixed_par';
            elseif reg == 5
                pp.a1 = 5; pp.a5 = 1.5;
                reg_tag = 'reg_5';
                reg_prefix = 'osc_reg_5_fixed_par';
            else
                error('Unknown region %d', reg);
            end

            % --- Folder Setup ---
            base_folder = fullfile(pwd, sprintf('%s_a1_%g_a5_%g', reg_prefix, pp.a1, pp.a5));
            folder_name = fullfile(base_folder, stability_method);
            if ~exist(folder_name, 'dir')
                mkdir(folder_name);
            end

            
            n_gva = 151; % For Gamma
            n_nva = 130; % For Wavenumbers

            gam_vals = linspace(1, 1000, n_gva);
            
            % diffusion grid generation (dense near critical points)
            d_vals   = linspace(37, 100, 55);
            dd = linspace(1, 17.4, 25);
            dd = [dd, linspace(17.5, 36.9, 80)];
            d_vals = [d_vals, dd];
            d_vals = sort(d_vals);
            n_dva = length(d_vals);
            
            nn_vals = linspace(0, 7, n_nva);

            % Save Grids
            writematrix(gam_vals, fullfile(folder_name, 'gam_vals.txt'));
            writematrix(d_vals,   fullfile(folder_name, 'd_vals.txt'));
            writematrix(nn_vals,  fullfile(folder_name, 'nn_vals.txt'));

            % -----------------------------------------------------------------
            % Calculate The Steady State via polynomial
            % -----------------------------------------------------------------
            pol = coef_pol(pp);
            m_roots  = roots(pol);
            m_roots  = m_roots(imag(m_roots)==0 & m_roots>0 & m_roots<=pp.a6);
            r_roots  = (m_roots./(pp.a8+m_roots)) .* (pp.a7+pp.a6-m_roots)./(pp.a6-m_roots);
            my_roots = [r_roots,m_roots];
            my_roots = my_roots(my_roots(:,1)>0 & my_roots(:,1)<=pp.a2,:);
            Ru = my_roots(1,1);
            Rv = my_roots(1,2);

            % -----------------------------------------------------------------
            % Main Parallel Loop over Gamma
            % -----------------------------------------------------------------
            parfor gg = 1:n_gva
                pp_temp = pp;
                pp_temp.gam = gam_vals(gg);
                
                if mod(gg, 20) == 0
                    fprintf('[%s - %s] Gamma index %d/%d\n', stability_method, reg_tag, gg, n_gva);
                end

                % --- Step 1: ODE Integration to find LC ---
                opts = odeset('RelTol',1e-10,'AbsTol',1e-10);
                
                % Transient phase
                tspan_prep = [0 1000]; % refine as needed
                x0_prep    = [Ru; Rv] + 1e-2;
                [~, x_prep] = ode15s(@(t,x) Nondim(t, x, pp_temp), tspan_prep, x0_prep, opts);
                
                
                    tspan = [0 300];
               
                x0 = x_prep(end, :).';
                [t_full, x_full] = ode15s(@(t,x) Nondim(t, x, pp_temp), tspan, x0, opts);
                
                % Extract LC
                [x_limit, t_limit, T_final, ~] = compute_limit_cycle(t_full, x_full);
                has_limit_cycle = ~isempty(x_limit);

                % --- Step 2:  ---
                J_avg_react = zeros(2,2);
                x_fun = [];
                y_fun = [];
                time_spline = []; 

                if has_limit_cycle
                    % Create splines (Used for FTDDI and AJM)
                    [x_fun, y_fun] = spline_function(t_limit, x_limit); % optional
                    time_spline = t_limit;

                    % AJM
                    if strcmp(stability_method, 'averaged')
                        %t_fine = linspace(t_limit(1), t_limit(end), 500);
                        t_fine = t_limit;
                        J_stack = zeros(2,2,length(t_fine));
                        for kkt = 1:length(t_fine)
                            tt = t_fine(kkt);
                            xx = x_fun(tt);
                            yy = y_fun(tt);
                            J_stack(:,:,kkt) = jac_matrix_kamp(xx, yy, pp_temp);
                        end
                        J_avg_react = trapz(t_fine, J_stack, 3) / T_final;
                    end
                end

                % Initialize saving metrics
                local_large_floq_d   = zeros(n_dva, 1);
                local_large_floq_d_k = zeros(n_dva, n_nva);

                % --- Step 3: d and wave number k ---
                for ii = 1:n_dva
                    pp_loop = pp_temp;
                    pp_loop.Dv = d_vals(ii);
                    
                    largest_eig_value = -inf;

                    for jw = 1:n_nva
                        k_val = nn_vals(jw);
                        if ii > 1 && abs(k_val) < 1e-12, continue; end
                        k_sqr  = (k_val * pi)^2;

                        if ~has_limit_cycle
                            max_eig_k = 0; 
                        else
                            % Compute Stability (FTDDI or AJM)

                            [max_eig_k, ~, ~] = compute_stability_metric_spline( ...
                                stability_method, x_fun, y_fun, pp_loop, k_sqr, time_spline, T_final, opts, J_avg_react);
                        end

                        % Store results per k
                        local_large_floq_d_k(ii, jw) = max_eig_k;

                        % Update max metric across k or fixed  d
                        if max_eig_k > largest_eig_value
                            largest_eig_value = max_eig_k;
                        end
                    end
                    local_large_floq_d(ii) = largest_eig_value;
                end

                % Output Files for fixed Gamma
                outFile_floq_d   = fullfile(folder_name, sprintf('large_floq_d_gam_%g.txt', pp_temp.gam));
                outFile_floq_d_k = fullfile(folder_name, sprintf('large_floq_d_k_gam_%g.txt', pp_temp.gam));

                writematrix(local_large_floq_d,   outFile_floq_d);
                writematrix(local_large_floq_d_k, outFile_floq_d_k);
            end
        end
    end
    delete(gcp('nocreate'));
end

%% ========================================================================
%% STABILITY CALCULATION CORE
%% ========================================================================
function [max_metric, eig_vals, det_val] = compute_stability_metric_spline( ...
    method, x_fun, y_fun, pp, k_squared, time_vec, ~, opts, J_avg_react_precalc)
    
    switch method
        case 'floquet'
            % Exact FTDDI: Integrates linearized system over period using splines
            n_dim  = 2;
            Phi_0  = eye(n_dim);
            Phi    = Phi_0;
            t_span = time_vec;
            
            for col = 1:n_dim
                %  spline handles x_fun, y_fun inside ODE
                [~, x_sol] = ode15s(@(tt, xx) kamp_linear_rd(tt, xx, ...
                    x_fun, y_fun, pp, k_squared), ...
                    t_span, Phi(:,col), opts);
                Phi(:,col) = x_sol(end, :)';
            end
            
            BB = Phi_0 \ Phi;
            eigs_raw = eig(BB);
            % Multiplier
            max_metric = max(abs(eigs_raw));
            eig_vals   = eigs_raw(:);
            det_val    = det(BB);

        case 'averaged'
            % AJM: 
            D_matrix = [1, 0; 0, pp.Dv];
            D_term   = k_squared * D_matrix;
            
            J_total_avg = J_avg_react_precalc - D_term;
            eigs_raw = eig(J_total_avg);
            
            %  Real part of eigenvalue 
            max_metric = max(real(eigs_raw));
            eig_vals   = eigs_raw(:);
            det_val    = det(J_total_avg);
    end
    
    if numel(eig_vals) ~= 2
        eig_vals = [NaN; NaN];
    end
end

%% ========================================================================
%% LOCAL FUNCTIONS
%% ========================================================================

function dydt=Nondim(~,y,pp)
    dydt=zeros(2,1);
    dydt(1)=pp.gam*((pp.a1*y(1)*(pp.a2-y(1)))/(((pp.a3+pp.a2-y(1))*(pp.a4*y(2)+y(1))))-y(1)/(pp.a5+y(1)));
    dydt(2)=pp.gam*((y(1)*(pp.a6-y(2)))/(pp.a7+pp.a6-y(2))-y(2)/(pp.a8+y(2)));
end

function Xdot = kamp_linear_rd(t, x, x_fun, y_fun, pp, k_squared)
    % Evaluates Jacobian 
    xx = x_fun(t);
    yy = y_fun(t);
    J_reaction = jac_matrix_kamp(xx,yy,pp);
    D_matrix = -k_squared * [1, 0; 0, pp.Dv];
    J_total = J_reaction + D_matrix;
    Xdot = J_total * x;
end

function Jac = jac_matrix_kamp(R, M, pp)
    fu=(pp.a1*(pp.a4*M*R^2-pp.a3*R^2-2*pp.a2*pp.a4*R*M-2*pp.a3*pp.a4*R*M+pp.a2^2*pp.a4*M+pp.a2*pp.a3*pp.a4*M))/...
        ((pp.a4*M+R)^2*(pp.a3+pp.a2-R)^2)-pp.a5/(pp.a5+R)^2;
    fv=-pp.a1*pp.a4*R*(pp.a2-R)/((pp.a2+pp.a3-R)*(pp.a4*M+R)^2);
    gu=(pp.a6-M)/(pp.a6+pp.a7-M);
    gv=-pp.a7*R/((pp.a6+pp.a7-M)^2)-pp.a8/(pp.a8+M)^2;
    Jac = pp.gam*[fu,  fv; gu,   gv];
end

function [x_fun, y_fun] = spline_function(t_limit, x_limit)
    if size(x_limit, 2) ~= 2
        error('x_limit must be an Nx2 matrix with x and y coordinates.');
    end
    x_spline_pp = spline(t_limit, x_limit(:,1)); 
    y_spline_pp = spline(t_limit, x_limit(:,2)); 
    x_fun = @(t) ppval(x_spline_pp, t);
    y_fun = @(t) ppval(y_spline_pp, t);
end

function [x_limit, t_limit, T_final, deviation] = compute_limit_cycle(t, x)
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
    t_start1 = peak_times1(end-1); t_end1 = peak_times1(end);
    idx_start1 = find(t >= t_start1, 1, 'first'); idx_end1 = find(t <= t_end1, 1, 'last');
    x_limit1 = x(idx_start1:idx_end1, :); t_limit1 = t(idx_start1:idx_end1) - t(idx_start1);
    deviation1 = norm(x_limit1(1,:) - x_limit1(end,:));

    T_final2 = mean(diff(peak_times2(2:end)));
    t_start2 = peak_times2(end-1); t_end2 = peak_times2(end);
    idx_start2 = find(t >= t_start2, 1, 'first'); idx_end2 = find(t <= t_end2, 1, 'last');
    x_limit2 = x(idx_start2:idx_end2, :); t_limit2 = t(idx_start2:idx_end2) - t(idx_start2);
    deviation2 = norm(x_limit2(1,:) - x_limit2(end,:));

    if deviation1 < deviation2
        x_limit = x_limit1; t_limit = t_limit1; T_final = T_final1; deviation = deviation1;
    else
        x_limit = x_limit2; t_limit = t_limit2; T_final = T_final2; deviation = deviation2;
    end
end

function out = coef_pol(pp)
    a1 = pp.a1; a2 = pp.a2; a3 = pp.a3; a4 = pp.a4; a5 = pp.a5; a6 = pp.a6; a7 = pp.a7; a8 = pp.a8;
    AA1 = a2 * a4 + a3 * a4 - a4;
    AA2 = (-a1*a2*a5 - 3*a2*a4*a6 - a2*a4*a7 + 2*a2*a4*a8 - 3*a3*a4*a6 -...
        a3*a4*a7 + 2*a3*a4*a8 - a1*a2 + a1*a5 + 3*a4*a6 + 2*a4*a7 - ...
        a4*a8 + a1 + a2 + a3 - 1);
    AA3 = (3*a1*a2*a5*a6 + a1*a2*a5*a7 - 2*a1*a2*a5*a8 + 3*a2*a4*a6^2 + ...
        2*a2*a4*a6*a7 - 6*a2*a4*a6*a8 - 2*a2*a4*a7*a8 + a2*a4*a8^2 + ...
        3*a3*a4*a6^2 + 2*a3*a4*a6*a7 - 6*a3*a4*a6*a8 - 2*a3*a4*a7*a8 + ...
        a3*a4*a8^2 + 3*a1*a2*a6 + 2*a1*a2*a7 - a1*a2*a8 - 3*a1*a5*a6 - ...
        2*a1*a5*a7 + a1*a5*a8 - 3*a4*a6^2 - 4*a4*a6*a7 + 3*a4*a6*a8 - ...
        a4*a7^2 + 2*a4*a7*a8 - 3*a1*a6 - 3*a1*a7 - 3*a2*a6 - 2*a2*a7 + ...
        a2*a8 - 3*a3*a6 - 2*a3*a7 + a3*a8 + 3*a6 + 3*a7);

    AA4 = (-3*a1*a2*a5*a6^2 - 2*a1*a2*a5*a6*a7 + 6*a1*a2*a5*a6*a8 + ...
        2*a1*a2*a5*a7*a8 - a1*a2*a5*a8^2 - a2*a4*a6^3 - a2*a4*a6^2*a7 + ...
        6*a2*a4*a6^2*a8 + 4*a2*a4*a6*a7*a8 - 3*a2*a4*a6*a8^2 - ...
        a2*a4*a7*a8^2 - a3*a4*a6^3 - a3*a4*a6^2*a7 + 6*a3*a4*a6^2*a8 + ...
        4*a3*a4*a6*a7*a8 - 3*a3*a4*a6*a8^2 - a3*a4*a7*a8^2 - 3*a1*a2*a6^2 - ...
        4*a1*a2*a6*a7 + 3*a1*a2*a6*a8 - a1*a2*a7^2 + 2*a1*a2*a7*a8 + ...
        3*a1*a5*a6^2 + 4*a1*a5*a6*a7 - 3*a1*a5*a6*a8 + a1*a5*a7^2 - ...
        2*a1*a5*a7*a8 + a4*a6^3 + 2*a4*a6^2*a7 - 3*a4*a6^2*a8 + ...
        a4*a6*a7^2 - 4*a4*a6*a7*a8 - a4*a7^2*a8 + 3*a1*a6^2 + 6*a1*a6*a7 + ...
        3*a1*a7^2 + 3*a2*a6^2 + 4*a2*a6*a7 - 3*a2*a6*a8 + a2*a7^2 - ...
        2*a2*a7*a8 + 3*a3*a6^2 + 4*a3*a6*a7 - 3*a3*a6*a8 + a3*a7^2 - ...
        2*a3*a7*a8 - 3*a6^2 - 6*a6*a7 - 3*a7^2);

    AA5 = (a1*a2*a5*a6^3 + a1*a2*a5*a6^2*a7 - 6*a1*a2*a5*a6^2*a8 - ...
        4*a1*a2*a5*a6*a7*a8 + 3*a1*a2*a5*a6*a8^2 + a1*a2*a5*a7*a8^2 - ...
        2*a2*a4*a6^3*a8 - 2*a2*a4*a6^2*a7*a8 + 3*a2*a4*a6^2*a8^2 + ...
        2*a2*a4*a6*a7*a8^2 - 2*a3*a4*a6^3*a8 - 2*a3*a4*a6^2*a7*a8 + ...
        3*a3*a4*a6^2*a8^2 + 2*a3*a4*a6*a7*a8^2 + a1*a2*a6^3 + ...
        2*a1*a2*a6^2*a7 - 3*a1*a2*a6^2*a8 + a1*a2*a6*a7^2 - ...
        4*a1*a2*a6*a7*a8 - a1*a2*a7^2*a8 - a1*a5*a6^3 - 2*a1*a5*a6^2*a7 + ...
        3*a1*a5*a6^2*a8 - a1*a5*a6*a7^2 + 4*a1*a5*a6*a7*a8 + ...
        a1*a5*a7^2*a8 + a4*a6^3*a8 + 2*a4*a6^2*a7*a8 + a4*a6*a7^2*a8 - ...
        a1*a6^3 - 3*a1*a6^2*a7 - 3*a1*a6*a7^2 - a1*a7^3 - a2*a6^3 - ...
        2*a2*a6^2*a7 + 3*a2*a6^2*a8 - a2*a6*a7^2 + 4*a2*a6*a7*a8 + ...
        a2*a7^2*a8 - a3*a6^3 - 2*a3*a6^2*a7 + 3*a3*a6^2*a8 - a3*a6*a7^2 + ...
        4*a3*a6*a7*a8 + a3*a7^2*a8 + a6^3 + 3*a6^2*a7 + 3*a6*a7^2 + a7^3);
    
    AA6 = (2*a1*a2*a5*a6^3*a8 + 2*a1*a2*a5*a6^2*a7*a8 - ...
        3*a1*a2*a5*a6^2*a8^2 - 2*a1*a2*a5*a6*a7*a8^2 - a2*a4*a6^3*a8^2 - ...
        a2*a4*a6^2*a7*a8^2 - a3*a4*a6^3*a8^2 - a3*a4*a6^2*a7*a8^2 + ...
        a1*a2*a6^3*a8 + 2*a1*a2*a6^2*a7*a8 + a1*a2*a6*a7^2*a8 - ...
        a1*a5*a6^3*a8 - 2*a1*a5*a6^2*a7*a8 - a1*a5*a6*a7^2*a8 - ...
        a2*a6^3*a8 - 2*a2*a6^2*a7*a8 - a2*a6*a7^2*a8 - a3*a6^3*a8 - ...
        2*a3*a6^2*a7*a8 - a3*a6*a7^2*a8);
    
    AA7 = (a1*a2*a5*a6^3*a8^2 + a1*a2*a5*a6^2*a7*a8^2);
    
    AA8 = 0;
    out = [AA1, AA2, AA3, AA4, AA5, AA6, AA7, AA8];
end