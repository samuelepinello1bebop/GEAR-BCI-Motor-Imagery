%% ========================================================================
%  SEVEN-MODEL BCI COMPARISON (RWOS-ELM variants)
%
%  Training:     first 70% (chronological) of Ses-1
%  Offline test: last 30% of Ses-1
%  Online test:  Ses-4 and Ses-5, simulated trial by trial
%
%  Features: one covariance matrix per trial, computed over the whole 5 s
%  window from cue onset, for 4 frequency bands and 5 channels, assembled
%  into a 20x20 block-diagonal SPD matrix. One prediction per trial.
%  The first N_calib trials of each online session are used for calibration;
%  the remaining trials are evaluated.
%
%  Models:
%    M1 static | M2 blind pseudo-labels | M3 Euclidean gate |
%    M4 Riemannian gate, no rebias | M5 Riemannian gate + static rebias |
%    M6 Ledoit-Wolf + static rebias | M7 GEAR (adaptive rebias)
%
%  Requires: MATLAB R2019b or later, EEGLAB, Statistics and Machine
%  Learning Toolbox. Edit base_path and out_dir below before running.
% ========================================================================
clear; clc; close all;
rng(42);

out_dir = 'C:\Dati Tesi\output_analisi';   % output folder (edit)
if ~exist(out_dir, 'dir'), mkdir(out_dir); end
diary_file = fullfile(out_dir, 'console_output.txt');
if exist(diary_file, 'file'), delete(diary_file); end
diary(diary_file); diary on;

% =========================================================================
% 1. GENERAL PARAMETERS AND HYPERPARAMETERS
% =========================================================================
base_path = 'C:\Dati Tesi\source_data__';  % data folder (edit)
subjects  = [1, 2, 4, 5, 7, 9, 10, 12, 13, 14, 15, 25];
runs      = 1:4;
ch_idx    = [10, 17, 18, 19, 40];
t_window  = [0 5];                         % epoch window from cue onset (s)
bands_fbts  = [8 12; 12 16; 16 22; 22 30]; % filter bank (Hz)

C_sel = 0.1;                               % ELM regularization
L_sel = 400;                               % ELM hidden neurons
normalize_weights = true;

N_calib = 20;                              % calibration trials per online session
lambda_forget = 0.98;
w_main = 0.6;

test_sessions = [4, 5];
test_session_names = {'post', 'follow'};

model_names = {
    '1. Static RWOS', ...
    '2. RWOS Pseudo-Label', ...
    '3. RWOS-ED', ...
    '4. RWOS-Riemann NO Rebias', ...
    '5. RWOS-Riemann Base', ...
    '6. RWOS-LW-Rebias', ...
    '7. GEAR'
};
mshort = {'M1','M2','M3','M4','M5','M6','M7'};
n_models = length(model_names);
num_subj = length(subjects);
T = length(test_sessions);
subj_labels = arrayfun(@(x) sprintf('Subj %02d', x), subjects, 'UniformOutput', false);

% --- Matrix initialization ---
acc_ses1_test  = nan(num_subj, 1);          % offline accuracy, M1 pipeline
acc_ses1_model = nan(num_subj, n_models);   % offline accuracy per model (M6 uses LW features)
n_tr_ses1 = nan(num_subj, 1);
n_te_ses1 = nan(num_subj, 1);
acc_global     = nan(num_subj, n_models, T);
bal_acc_global = nan(num_subj, n_models, T);
f1_global      = nan(num_subj, n_models, T);
auroc_global   = nan(num_subj, n_models, T);
spec_global    = nan(num_subj, n_models, T);
sens_global    = nan(num_subj, n_models, T);
pct_rest_g     = nan(num_subj, n_models, T);   % % of trials predicted as Rest
TP_g = nan(num_subj, n_models, T); TN_g = TP_g; FP_g = TP_g; FN_g = TP_g;
n_upd    = nan(num_subj, n_models, T);         % updates performed
n_upd_ok = nan(num_subj, n_models, T);         % updates with correct pseudo-label
n_trials_tot = nan(num_subj, T);               % trials loaded per session
n_eval       = nan(num_subj, T);               % trials evaluated (after the calibration trials)
n_eval_MI    = nan(num_subj, T);
n_eval_Rest  = nan(num_subj, T);
gamma_LW     = nan(num_subj, T);               % mean Ledoit-Wolf shrinkage
drift_curve  = cell(num_subj, T);
drift_d0     = nan(num_subj, T);
raw_results  = cell(num_subj, T);

fprintf('==============================================================\n');
fprintf(' FINAL SEVEN-MODEL SIMULATION (%d subjects)\n', num_subj);
fprintf('==============================================================\n');

% =========================================================================
% 2. SUBJECT LOOP
% =========================================================================
for s_idx = 1:num_subj
    subj = subjects(s_idx);
    fprintf('---> Processing subject %02d (%d/%d)\n', subj, s_idx, num_subj);

    % --- PHASE 1: TRAINING ON SES-1 (70/30 split) ---
    try
        [EEG1_FB, lab1, ~] = load_epochs(subj, 1, 'pre', runs, base_path, t_window, bands_fbts, ch_idx);
    catch ME
        fprintf('     Error loading Ses-1: %s\n', ME.message);
        continue;
    end

    if isempty(EEG1_FB{1})
        fprintf('     Ses-1 data empty or not found, skipping.\n');
        continue;
    end

    N_total = numel(lab1);
    idx_tr = 1:round(0.7 * N_total);
    idx_te = (idx_tr(end) + 1):N_total;
    n_tr_ses1(s_idx) = numel(idx_tr);
    n_te_ses1(s_idx) = numel(idx_te);
    fprintf('     Ses-1: %d trials in total -> %d training, %d offline test\n', N_total, numel(idx_tr), numel(idx_te));

    lab_tr = lab1(idx_tr); lab_te = lab1(idx_te);
    classes = unique(lab_tr);

    Y0_tr = zeros(numel(idx_tr), 2);
    for c = 1:2, Y0_tr(lab_tr == classes(c), c) = 1; end

    C_tr_FB = create_block_cov(EEG1_FB, idx_tr);
    C_te_FB = create_block_cov(EEG1_FB, idx_te);

    C_mean_tr_FB = karcher_mean(C_tr_FB);
    C_class_FB = zeros(size(C_mean_tr_FB, 1), size(C_mean_tr_FB, 2), 2);
    C_class_FB(:,:,1) = karcher_mean(C_tr_FB(:,:, lab_tr == classes(1)));
    C_class_FB(:,:,2) = karcher_mean(C_tr_FB(:,:, lab_tr == classes(2)));

    C_tr_LW = create_ledoit_wolf_block_cov(EEG1_FB, idx_tr);
    C_te_LW = create_ledoit_wolf_block_cov(EEG1_FB, idx_te);

    C_mean_tr_LW = karcher_mean(C_tr_LW);
    C_class_LW = zeros(size(C_mean_tr_LW, 1), size(C_mean_tr_LW, 2), 2);
    C_class_LW(:,:,1) = karcher_mean(C_tr_LW(:,:, lab_tr == classes(1)));
    C_class_LW(:,:,2) = karcher_mean(C_tr_LW(:,:, lab_tr == classes(2)));

    X0_tr_FB = extract_all_features(C_tr_FB, C_mean_tr_FB);
    X0_te_FB = extract_all_features(C_te_FB, C_mean_tr_FB);

    mu_tr_FB = mean(X0_tr_FB, 1); sg_tr_FB = std(X0_tr_FB, 0, 1); sg_tr_FB(sg_tr_FB==0) = 1;
    X0_tr_FB_norm = (X0_tr_FB - mu_tr_FB) ./ sg_tr_FB;

    mu_class_ED = zeros(2, size(X0_tr_FB_norm, 2));
    mu_class_ED(1,:) = mean(X0_tr_FB_norm(lab_tr == classes(1), :), 1);
    mu_class_ED(2,:) = mean(X0_tr_FB_norm(lab_tr == classes(2), :), 1);

    X0_tr_LW = extract_all_features(C_tr_LW, C_mean_tr_LW);
    X0_te_LW = extract_all_features(C_te_LW, C_mean_tr_LW);

    [beta_FB, K_FB, Q_FB, A_FB, b_FB] = train_elm(X0_tr_FB, Y0_tr, C_sel, L_sel, normalize_weights);
    [beta_LW, K_LW, Q_LW, A_LW, b_LW] = train_elm(X0_tr_LW, Y0_tr, C_sel, L_sel, normalize_weights);

    sc_FB = elm_scores(X0_te_FB, beta_FB, A_FB, b_FB);
    [~, pr_FB] = max(sc_FB, [], 2);
    acc_ses1_test(s_idx) = mean(pr_FB == lab_te) * 100;

    % Offline accuracy of M6 with its own (Ledoit-Wolf) features
    sc_LW = elm_scores(X0_te_LW, beta_LW, A_LW, b_LW);
    [~, pr_LW] = max(sc_LW, [], 2);
    acc_ses1_model(s_idx, :) = acc_ses1_test(s_idx);
    acc_ses1_model(s_idx, 6) = mean(pr_LW == lab_te) * 100;
    fprintf('     Ses-1 offline: acc M1 = %.1f%% | acc M6 (LW) = %.1f%%\n', acc_ses1_test(s_idx), acc_ses1_model(s_idx, 6));

    % --- PHASE 2: ONLINE TEST (SES-4 and SES-5) ---
    for t_idx = 1:T
        try
            [EEGt_FB, lab_long, ~] = load_epochs(subj, test_sessions(t_idx), test_session_names{t_idx}, runs, base_path, t_window, bands_fbts, ch_idx);
        catch ME
            fprintf('     Error loading Ses-%d: %s\n', test_sessions(t_idx), ME.message);
            continue;
        end
        if isempty(EEGt_FB{1}), fprintf('     Ses-%d empty, skipping.\n', test_sessions(t_idx)); continue; end

        N_test = length(lab_long);
        if N_test <= N_calib, continue; end
        N_ev = N_test - N_calib;
        n_trials_tot(s_idx, t_idx) = N_test;
        n_eval(s_idx, t_idx) = N_ev;
        fprintf('     Ses-%d: %d trials in total, %d evaluated\n', test_sessions(t_idx), N_test, N_ev);

        C_long_FB = create_block_cov(EEGt_FB, 1:N_test);
        [C_long_LW, g_LW] = create_ledoit_wolf_block_cov(EEGt_FB, 1:N_test);
        gamma_LW(s_idx, t_idx) = mean(g_LW(:));

        X_raw_FB = extract_all_features(C_long_FB, C_mean_tr_FB);

        mu_calib_ED = mean(X_raw_FB(1:N_calib, :), 1);
        sg_calib_ED = std(X_raw_FB(1:N_calib, :), 0, 1); sg_calib_ED(sg_calib_ED==0) = 1;

        C_mean_calib_FB = karcher_mean(C_long_FB(:,:,1:N_calib));
        C_mean_calib_LW = karcher_mean(C_long_LW(:,:,1:N_calib));
        pred_m = zeros(N_ev, n_models);
        score_m = zeros(N_ev, n_models);
        upd_flag = false(N_ev, n_models);
        true_labels = lab_long(N_calib+1:end);
        n_eval_MI(s_idx, t_idx) = sum(true_labels == 1);
        n_eval_Rest(s_idx, t_idx) = sum(true_labels == 2);

        beta_m1 = beta_FB;                          % M1 never updates
        beta_m2 = beta_FB; K_m2 = K_FB; Q_m2 = Q_FB;
        beta_m3 = beta_FB; K_m3 = K_FB; Q_m3 = Q_FB;
        beta_m4 = beta_FB; K_m4 = K_FB; Q_m4 = Q_FB;
        beta_m5 = beta_FB; K_m5 = K_FB; Q_m5 = Q_FB;
        beta_m6 = beta_LW; K_m6 = K_LW; Q_m6 = Q_LW;
        beta_m7 = beta_FB; K_m7 = K_FB; Q_m7 = Q_FB;

        C_mean_online_M7 = C_mean_calib_FB;
        drift_k = nan(N_ev, 1);
        drift_d0(s_idx, t_idx) = riemann_distance(C_mean_calib_FB, C_mean_tr_FB);

        for k = 1:N_ev
            idx_t = N_calib + k;

            % M1 (static)
            x_m1 = X_raw_FB(idx_t, :);
            sc_m1 = elm_scores(x_m1, beta_m1, A_FB, b_FB);
            [~, pred_m(k, 1)] = max(sc_m1);
            score_m(k, 1) = sc_m1(1) - sc_m1(2);

            % M2 (blind pseudo-labels: always updates with its own prediction)
            x_m2 = X_raw_FB(idx_t, :);
            sc_m2 = elm_scores(x_m2, beta_m2, A_FB, b_FB);
            [~, pr_elm_m2] = max(sc_m2);
            pred_m(k, 2) = pr_elm_m2;
            score_m(k, 2) = sc_m2(1) - sc_m2(2);
            yt_m2 = zeros(1,2); yt_m2(pr_elm_m2) = 1;
            [beta_m2, K_m2, Q_m2] = update_rwos_elm(x_m2, K_m2, Q_m2, A_FB, b_FB, C_sel, yt_m2, lambda_forget, w_main);
            upd_flag(k, 2) = true;

            % M3 (Euclidean gate)
            x_m3_norm = (X_raw_FB(idx_t, :) - mu_calib_ED) ./ sg_calib_ED;
            sc_m3 = elm_scores(X_raw_FB(idx_t, :), beta_m3, A_FB, b_FB);
            [~, pr_elm_m3] = max(sc_m3);
            pred_m(k, 3) = pr_elm_m3;
            score_m(k, 3) = sc_m3(1) - sc_m3(2);
            d_ed_1 = norm(x_m3_norm - mu_class_ED(1,:));
            d_ed_2 = norm(x_m3_norm - mu_class_ED(2,:));
            pr_dist_m3 = (d_ed_1 > d_ed_2) + 1;
            if pr_elm_m3 == pr_dist_m3
                yt = zeros(1,2); yt(pr_elm_m3) = 1;
                [beta_m3, K_m3, Q_m3] = update_rwos_elm(X_raw_FB(idx_t, :), K_m3, Q_m3, A_FB, b_FB, C_sel, yt, lambda_forget, w_main);
                upd_flag(k, 3) = true;
            end

            % M4 (Riemannian gate, no rebias)
            x_m4 = X_raw_FB(idx_t, :);
            sc_m4 = elm_scores(x_m4, beta_m4, A_FB, b_FB);
            [~, pr_elm_m4] = max(sc_m4);
            pred_m(k, 4) = pr_elm_m4;
            score_m(k, 4) = sc_m4(1) - sc_m4(2);
            d_rnr_1 = riemann_distance(C_long_FB(:,:,idx_t), C_class_FB(:,:,1));
            d_rnr_2 = riemann_distance(C_long_FB(:,:,idx_t), C_class_FB(:,:,2));
            pr_dist_m4 = (d_rnr_1 > d_rnr_2) + 1;
            if pr_elm_m4 == pr_dist_m4
                yt = zeros(1,2); yt(pr_elm_m4) = 1;
                [beta_m4, K_m4, Q_m4] = update_rwos_elm(x_m4, K_m4, Q_m4, A_FB, b_FB, C_sel, yt, lambda_forget, w_main);
                upd_flag(k, 4) = true;
            end

            % M5 (static rebias computed on the calibration trials)
            C_rebias_M5 = riemann_rebias(C_long_FB(:,:,idx_t), C_mean_calib_FB, C_mean_tr_FB);
            x_m5 = extract_riemann_features(C_rebias_M5, C_mean_tr_FB);
            sc_m5 = elm_scores(x_m5, beta_m5, A_FB, b_FB);
            [~, pr_elm_m5] = max(sc_m5);
            pred_m(k, 5) = pr_elm_m5;
            score_m(k, 5) = sc_m5(1) - sc_m5(2);
            d_riem_1 = riemann_distance(C_rebias_M5, C_class_FB(:,:,1));
            d_riem_2 = riemann_distance(C_rebias_M5, C_class_FB(:,:,2));
            pr_dist_m5 = (d_riem_1 > d_riem_2) + 1;
            if pr_elm_m5 == pr_dist_m5
                yt = zeros(1,2); yt(pr_elm_m5) = 1;
                [beta_m5, K_m5, Q_m5] = update_rwos_elm(x_m5, K_m5, Q_m5, A_FB, b_FB, C_sel, yt, lambda_forget, w_main);
                upd_flag(k, 5) = true;
            end

            % M6 (Ledoit-Wolf covariance + static rebias)
            C_rebias_LW = riemann_rebias(C_long_LW(:,:,idx_t), C_mean_calib_LW, C_mean_tr_LW);
            x_m6 = extract_riemann_features(C_rebias_LW, C_mean_tr_LW);
            sc_m6 = elm_scores(x_m6, beta_m6, A_LW, b_LW);
            [~, pr_elm_m6] = max(sc_m6);
            pred_m(k, 6) = pr_elm_m6;
            score_m(k, 6) = sc_m6(1) - sc_m6(2);
            d_lw_1 = riemann_distance(C_rebias_LW, C_class_LW(:,:,1));
            d_lw_2 = riemann_distance(C_rebias_LW, C_class_LW(:,:,2));
            pr_dist_m6 = (d_lw_1 > d_lw_2) + 1;
            if pr_elm_m6 == pr_dist_m6
                yt = zeros(1,2); yt(pr_elm_m6) = 1;
                [beta_m6, K_m6, Q_m6] = update_rwos_elm(x_m6, K_m6, Q_m6, A_LW, b_LW, C_sel, yt, lambda_forget, w_main);
                upd_flag(k, 6) = true;
            end

            % M7 (GEAR: adaptive rebias with online Riemannian mean)
            i_step = N_calib + k; alpha = 1 / i_step;
            C_isq = sym_fun(C_mean_online_M7, @(x) 1./sqrt(max(x, eps)));
            C_sq  = sym_fun(C_mean_online_M7, @(x) sqrt(max(x, eps)));
            M_upd = C_isq * C_long_FB(:,:,idx_t) * C_isq;
            C_mean_online_M7 = C_sq * sym_fun(M_upd, @(x) max(x, eps).^alpha) * C_sq;
            C_mean_online_M7 = (C_mean_online_M7 + C_mean_online_M7') / 2;
            drift_k(k) = riemann_distance(C_mean_online_M7, C_mean_tr_FB);

            C_rebias_M7 = riemann_rebias(C_long_FB(:,:,idx_t), C_mean_online_M7, C_mean_tr_FB);
            x_m7 = extract_riemann_features(C_rebias_M7, C_mean_tr_FB);

            sc_m7 = elm_scores(x_m7, beta_m7, A_FB, b_FB);
            [~, pr_elm_m7] = max(sc_m7);
            pred_m(k, 7) = pr_elm_m7;
            score_m(k, 7) = sc_m7(1) - sc_m7(2);
            d_gear_1 = riemann_distance(C_rebias_M7, C_class_FB(:,:,1));
            d_gear_2 = riemann_distance(C_rebias_M7, C_class_FB(:,:,2));
            pr_dist_m7 = (d_gear_1 > d_gear_2) + 1;
            if pr_elm_m7 == pr_dist_m7
                yt = zeros(1,2); yt(pr_elm_m7) = 1;
                [beta_m7, K_m7, Q_m7] = update_rwos_elm(x_m7, K_m7, Q_m7, A_FB, b_FB, C_sel, yt, lambda_forget, w_main);
                upd_flag(k, 7) = true;
            end
        end

        drift_curve{s_idx, t_idx} = drift_k;
        raw_results{s_idx, t_idx} = struct('pred', pred_m, 'score', score_m, 'upd', upd_flag, 'true', true_labels);

        for m = 1:n_models
            preds = pred_m(:, m);
            scores = score_m(:, m);

            TP = sum(preds == 1 & true_labels == 1);
            TN = sum(preds == 2 & true_labels == 2);
            FP = sum(preds == 1 & true_labels == 2);
            FN = sum(preds == 2 & true_labels == 1);
            TP_g(s_idx, m, t_idx) = TP; TN_g(s_idx, m, t_idx) = TN;
            FP_g(s_idx, m, t_idx) = FP; FN_g(s_idx, m, t_idx) = FN;

            acc_global(s_idx, m, t_idx) = (TP + TN) / max(TP + TN + FP + FN, 1) * 100;

            TPR = TP / max(TP + FN, 1);
            TNR = TN / max(TN + FP, 1);
            sens_global(s_idx, m, t_idx) = TPR * 100;
            spec_global(s_idx, m, t_idx) = TNR * 100;
            bal_acc_global(s_idx, m, t_idx) = 0.5 * (TPR + TNR) * 100;
            pct_rest_g(s_idx, m, t_idx) = (TN + FN) / max(TP + TN + FP + FN, 1) * 100;

            precision = TP / max(TP + FP, 1);
            recall = TPR;
            f1_global(s_idx, m, t_idx) = 2 * (precision * recall) / max(precision + recall, eps) * 100;

            n_upd(s_idx, m, t_idx) = sum(upd_flag(:, m));
            n_upd_ok(s_idx, m, t_idx) = sum(upd_flag(:, m) & (preds == true_labels));

            try
                [~, ~, ~, auc] = perfcurve(true_labels, scores, 1);
                auroc_global(s_idx, m, t_idx) = auc * 100;
            catch
                auroc_global(s_idx, m, t_idx) = NaN;
            end
        end
    end
end

%% ========================================================================
%  3. NUMERICAL RESULTS
% ========================================================================
acc_eco  = mean(acc_global, 3, 'omitnan');
bal_eco  = mean(bal_acc_global, 3, 'omitnan');
spec_eco = mean(spec_global, 3, 'omitnan');
sens_eco = mean(sens_global, 3, 'omitnan');
f1_eco   = mean(f1_global, 3, 'omitnan');
auc_eco  = mean(auroc_global, 3, 'omitnan');

mean_acc = mean(acc_eco, 1, 'omitnan'); std_acc = std(acc_eco, 0, 1, 'omitnan');
mean_bal = mean(bal_eco, 1, 'omitnan'); std_bal = std(bal_eco, 0, 1, 'omitnan');
mean_spc = mean(spec_eco, 1, 'omitnan'); std_spc = std(spec_eco, 0, 1, 'omitnan');
mean_sns = mean(sens_eco, 1, 'omitnan'); std_sns = std(sens_eco, 0, 1, 'omitnan');
mean_f1  = mean(f1_eco, 1, 'omitnan');  std_f1  = std(f1_eco, 0, 1, 'omitnan');
mean_auc = mean(auc_eco, 1, 'omitnan'); std_auc = std(auc_eco, 0, 1, 'omitnan');

fprintf('\n==============================================================================================\n');
fprintf(' GLOBAL ONLINE NUMERICAL RESULTS (mean of S4 and S5) - MEAN +- STD\n');
fprintf('==============================================================================================\n');
fprintf('%-8s | %-13s | %-13s | %-13s | %-13s | %-13s | %-13s\n', 'Model', 'Accuracy', 'Bal. Acc.', 'Sensitivity', 'Specificity', 'F1-Score', 'AUROC');
fprintf('----------------------------------------------------------------------------------------------\n');
for m = 1:n_models
    fprintf('M%-7d | %5.2f +- %5.2f | %5.2f +- %5.2f | %5.2f +- %5.2f | %5.2f +- %5.2f | %5.2f +- %5.2f | %5.2f +- %5.2f\n', ...
        m, mean_acc(m), std_acc(m), mean_bal(m), std_bal(m), mean_sns(m), std_sns(m), ...
        mean_spc(m), std_spc(m), mean_f1(m), std_f1(m), mean_auc(m), std_auc(m));
end

% ---- Accuracy per session (exact values for S4 and S5) ----
fprintf('\n==============================================================================================\n');
fprintf(' MEAN ACCURACY PER SESSION (mean +- SD across subjects)\n');
fprintf('==============================================================================================\n');
fprintf('%-8s | %-15s | %-15s\n', 'Model', 'S4 (Post)', 'S5 (Follow-up)');
for m = 1:n_models
    fprintf('M%-7d | %6.2f +- %5.2f | %6.2f +- %5.2f\n', m, ...
        mean(acc_global(:,m,1), 'omitnan'), std(acc_global(:,m,1), 'omitnan'), ...
        mean(acc_global(:,m,2), 'omitnan'), std(acc_global(:,m,2), 'omitnan'));
end
fprintf('Offline S1 (M1-M5, M7): %.2f +- %.2f | M6 (LW): %.2f +- %.2f\n', ...
    mean(acc_ses1_test, 'omitnan'), std(acc_ses1_test, 'omitnan'), ...
    mean(acc_ses1_model(:,6), 'omitnan'), std(acc_ses1_model(:,6), 'omitnan'));

% ---- Per-subject tables ----
print_matrix('ONLINE ACCURACY (mean of S4 and S5) PER SUBJECT (%)', acc_eco, subj_labels, mshort, '%8.2f');
print_matrix('ONLINE SENSITIVITY (mean of S4 and S5) PER SUBJECT (%)', sens_eco, subj_labels, mshort, '%8.2f');
print_matrix('ONLINE SPECIFICITY (mean of S4 and S5) PER SUBJECT (%)', spec_eco, subj_labels, mshort, '%8.2f');
print_matrix('ONLINE AUROC (mean of S4 and S5) PER SUBJECT (%)', auc_eco, subj_labels, mshort, '%8.2f');

% ---- How often each model predicts "Rest" (pooled S4+S5) ----
sumTP = sum(TP_g, 3, 'omitnan'); sumTN = sum(TN_g, 3, 'omitnan');
sumFP = sum(FP_g, 3, 'omitnan'); sumFN = sum(FN_g, 3, 'omitnan');
n_pool = sumTP + sumTN + sumFP + sumFN;
pct_rest_pool = (sumTN + sumFN) ./ max(n_pool, 1) * 100;
print_matrix('% OF TRIALS PREDICTED AS REST (pooled S4+S5). ~100% = constant classifier', pct_rest_pool, subj_labels, mshort, '%8.1f');
fprintf('\nSubjects for whom M1-M4 predict Rest in at least 95%% of trials (or MI in at least 95%%):\n');
for m = 1:4
    idx_const = find(pct_rest_pool(:, m) >= 95 | pct_rest_pool(:, m) <= 5);
    fprintf('  %s: %s\n', mshort{m}, mat2str(subjects(idx_const)));
end

% ---- Pooled confusion counts per subject (M1 and M7) ----
fprintf('\nCONFUSION COUNTS pooled S4+S5 (TP/FN/FP/TN) per subject\n');
fprintf('%-10s | %-22s | %-22s\n', 'Subject', 'M1 (TP/FN/FP/TN)', 'M7 (TP/FN/FP/TN)');
for i = 1:num_subj
    fprintf('%-10s | %4d/%4d/%4d/%4d   | %4d/%4d/%4d/%4d\n', subj_labels{i}, ...
        sumTP(i,1), sumFN(i,1), sumFP(i,1), sumTN(i,1), sumTP(i,7), sumFN(i,7), sumFP(i,7), sumTN(i,7));
end

% ---- Number of trials used ----
fprintf('\n==============================================================================================\n');
fprintf(' TRIALS ACTUALLY USED\n');
fprintf('==============================================================================================\n');
fprintf('%-10s | %-14s | %-24s | %-24s\n', 'Subject', 'S1 train/test', 'S4 total/eval (MI/Rest)', 'S5 total/eval (MI/Rest)');
for i = 1:num_subj
    fprintf('%-10s | %5d / %-6d | %4d / %3d (%2d / %2d)     | %4d / %3d (%2d / %2d)\n', subj_labels{i}, ...
        n_tr_ses1(i), n_te_ses1(i), ...
        n_trials_tot(i,1), n_eval(i,1), n_eval_MI(i,1), n_eval_Rest(i,1), ...
        n_trials_tot(i,2), n_eval(i,2), n_eval_MI(i,2), n_eval_Rest(i,2));
end
fprintf('Total online trials evaluated: %d (MI = %d, Rest = %d)\n', ...
    sum(n_eval(:), 'omitnan'), sum(n_eval_MI(:), 'omitnan'), sum(n_eval_Rest(:), 'omitnan'));

% ---- Chance threshold per subject (one-sided binomial, alpha = 0.05) ----
n_pool_subj = sum(n_eval, 2, 'omitnan');
thr_pool = (binoinv(0.95, n_pool_subj, 0.5) + 1) ./ n_pool_subj * 100;
acc_pool = (sumTP + sumTN) ./ max(n_pool, 1) * 100;
thr_s1 = (binoinv(0.95, n_te_ses1, 0.5) + 1) ./ n_te_ses1 * 100;
fprintf('\n==============================================================================================\n');
fprintf(' SUBJECTS ABOVE CHANCE (one-sided binomial test, alpha = 0.05, per subject)\n');
fprintf('==============================================================================================\n');
fprintf('Offline S1 (M1): %d of %d subjects above threshold. Per-subject thresholds (%%): %s\n', ...
    sum(acc_ses1_test >= thr_s1, 'omitnan'), num_subj, mat2str(round(thr_s1', 1)));
for m = 1:n_models
    above = acc_pool(:, m) >= thr_pool;
    fprintf('Online %s: %d of %d above threshold (pooled S4+S5 accuracy). Subjects: %s\n', ...
        mshort{m}, sum(above), num_subj, mat2str(subjects(above)));
end
fprintf('Online per-subject thresholds (%%): %s\n', mat2str(round(thr_pool', 1)));

% ---- Gate: accepted updates and their correctness ----
upd_pct = squeeze(sum(n_upd, 3, 'omitnan')) ./ max(n_pool_subj, 1) * 100;      % [subjects x models]
ok_pct  = squeeze(sum(n_upd_ok, 3, 'omitnan')) ./ max(squeeze(sum(n_upd, 3, 'omitnan')), 1) * 100;
ok_pct(squeeze(sum(n_upd, 3, 'omitnan')) == 0) = NaN;
print_matrix('% OF TRIALS USED FOR UPDATE (accepted by the gate; M2 = 100% by construction)', upd_pct, subj_labels, mshort, '%8.1f');
print_matrix('GATE PRECISION: % of performed updates with a correct pseudo-label', ok_pct, subj_labels, mshort, '%8.1f');
fprintf('\nNote: M1 never updates (0%%). A gate precision close to 50%% means the gate does not filter better than chance.\n');

% ---- Ledoit-Wolf shrinkage ----
print_matrix('MEAN LEDOIT-WOLF GAMMA (M6) PER SESSION [S4, S5]  (values ~0 = little regularization)', gamma_LW, subj_labels, {'S4','S5'}, '%8.4f');

% ---- Offline accuracy per model ----
print_matrix('OFFLINE ACCURACY S1 (30% test) - M1 and M6', [acc_ses1_test, acc_ses1_model(:,6), n_te_ses1], subj_labels, {'M1','M6(LW)','n_test'}, '%8.2f');

% ---- Correlation between offline S1 and online accuracy ----
valid_idx = ~any(isnan(acc_eco), 2);
fprintf('\nSpearman correlation between offline S1 accuracy and mean online accuracy (exploratory):\n');
for m = [1 4 5 7]
    [rho, pv] = corr(acc_ses1_test(valid_idx), acc_eco(valid_idx, m), 'Type', 'Spearman');
    fprintf('  S1 vs %s: rho = %.2f, p = %.4f\n', mshort{m}, rho, pv);
end

%% ========================================================================
%  4. INFERENTIAL STATISTICS
% ========================================================================
pairs_plan = [1 2; 2 4; 3 4; 4 5; 5 6; 5 7];
run_stat_tests(acc_eco,  'ACCURACY (primary analysis)',  pairs_plan, true);
run_stat_tests(spec_eco, 'SPECIFICITY',                  pairs_plan, true);
run_stat_tests(sens_eco, 'SENSITIVITY (exploratory)',    pairs_plan, true);
run_stat_tests(auc_eco,  'AUROC (exploratory)',          pairs_plan, true);
run_stat_tests(auc_eco,  'AUROC - extra comparisons (exploratory, FDR over 2 tests)', [1 5; 1 7], false);
run_stat_tests(acc_eco,  'ACCURACY - extra comparisons (exploratory, FDR over 2 tests)', [1 5; 1 7], false);

%% ========================================================================
%  5. DATA EXPORT
% ========================================================================
n_rows = num_subj * T * n_models;
tab_cell = cell(n_rows, 18);
r = 0;
for s = 1:num_subj
    for t = 1:T
        for m = 1:n_models
            r = r + 1;
            tab_cell(r, :) = {subjects(s), test_sessions(t), mshort{m}, n_trials_tot(s,t), n_eval(s,t), ...
                TP_g(s,m,t), FN_g(s,m,t), FP_g(s,m,t), TN_g(s,m,t), ...
                acc_global(s,m,t), bal_acc_global(s,m,t), sens_global(s,m,t), spec_global(s,m,t), ...
                f1_global(s,m,t), auroc_global(s,m,t), pct_rest_g(s,m,t), n_upd(s,m,t), n_upd_ok(s,m,t)};
        end
    end
end
T_out = cell2table(tab_cell, 'VariableNames', {'subject','session','model','n_trials_tot','n_eval', ...
    'TP','FN','FP','TN','accuracy','balanced_acc','sensitivity','specificity','f1','auroc', ...
    'pct_pred_rest','n_updates','n_updates_correct'});
writetable(T_out, fullfile(out_dir, 'risultati_per_soggetto.csv'));

pred_rows = [];
for s = 1:num_subj
    for t = 1:T
        if isempty(raw_results{s,t}), continue; end
        res = raw_results{s,t};
        n_r = numel(res.true);
        pred_rows = [pred_rows; repmat(subjects(s), n_r, 1), repmat(test_sessions(t), n_r, 1), (1:n_r)', res.true(:), ...
            res.pred, res.score, double(res.upd)]; %#ok<AGROW>
    end
end
vn = [{'subject','session','trial','true_label'}, strcat('pred_', mshort), strcat('score_', mshort), strcat('upd_', mshort)];
writetable(array2table(pred_rows, 'VariableNames', vn), fullfile(out_dir, 'predizioni_grezze.csv'));
save(fullfile(out_dir, 'risultati_completi.mat'), 'acc_global', 'bal_acc_global', 'f1_global', 'auroc_global', ...
    'spec_global', 'sens_global', 'TP_g', 'TN_g', 'FP_g', 'FN_g', 'n_upd', 'n_upd_ok', 'n_trials_tot', 'n_eval', ...
    'acc_ses1_test', 'acc_ses1_model', 'gamma_LW', 'drift_curve', 'drift_d0', 'raw_results', 'subjects', 'model_names');
fprintf('\nFiles saved in %s\n', out_dir);

%% ========================================================================
%  6. FIGURES
% ========================================================================
palette = [
    0.60, 0.60, 0.60;  % M1
    0.85, 0.20, 0.20;  % M2
    0.00, 0.45, 0.74;  % M3
    0.93, 0.69, 0.13;  % M4
    0.85, 0.40, 0.10;  % M5
    0.49, 0.18, 0.56;  % M6
    0.15, 0.60, 0.20   % M7
];

global_mean_3ses = [mean(acc_ses1_model, 1, 'omitnan'); squeeze(mean(acc_global, 1, 'omitnan'))'];
global_std_3ses  = [std(acc_ses1_model, 0, 1, 'omitnan'); squeeze(std(acc_global, 0, 1, 'omitnan'))'];

% --- FIGURE 1: GLOBAL BAR PLOT ---
fig1 = figure('Color', 'w', 'Position', [100 100 1200 700]); hold on;
bh = bar(global_mean_3ses, 'grouped', 'EdgeColor', 'k', 'LineWidth', 1, 'BarWidth', 0.85);
for m = 1:n_models
    bh(m).FaceColor = palette(m, :);
    x_pos = bh(m).XEndPoints;
    errorbar(x_pos, global_mean_3ses(:, m), global_std_3ses(:, m), 'k', 'LineStyle', 'none', 'LineWidth', 1.5, 'CapSize', 8, 'HandleVisibility', 'off');
end
yline(50, '--r', 'Chance Level', 'LineWidth', 2, 'LabelHorizontalAlignment', 'left', 'FontSize', 12, 'HandleVisibility', 'off');
xlim([0.4 3.6]); ylim([40 105]); xticks([1 2 3]); xticklabels({'Session 1 (Baseline)', 'Session 4 (Post)', 'Session 5 (Follow-up)'});
ylabel('Mean Global Accuracy (%)', 'FontWeight', 'bold', 'FontSize', 14);
title('Longitudinal Accuracy Trend (M1-M7)', 'FontSize', 16, 'FontWeight', 'bold');
lgd1 = legend(bh, model_names, 'Location', 'southoutside', 'FontSize', 12); lgd1.NumColumns = 2; legend('boxoff');
set(gca, 'FontSize', 12, 'Box', 'on', 'YGrid', 'on', 'GridAlpha', 0.3, 'LineWidth', 1.5);

% --- FIGURE 2: MULTI-METRIC ---
metrics_matrix = [mean_acc', mean_bal', mean_f1', mean_auc'];
fig2 = figure('Color', 'w', 'Position', [150 150 1200 650]);
b_met = bar(metrics_matrix, 'grouped', 'EdgeColor', 'k', 'LineWidth', 1, 'BarWidth', 0.85);
b_met(1).FaceColor = [0.20 0.60 0.80]; b_met(2).FaceColor = [0.85 0.40 0.20];
b_met(3).FaceColor = [0.30 0.75 0.30]; b_met(4).FaceColor = [0.55 0.25 0.60];
yline(50, '--k', 'Chance Level', 'LineWidth', 1.5, 'FontSize', 12, 'HandleVisibility', 'off');
set(gca, 'XTickLabel', mshort, 'FontSize', 13, 'FontWeight', 'bold');
ylabel('Mean Score (%)', 'FontWeight', 'bold', 'FontSize', 14);
title('Online Multi-Metric Evaluation (Mean of Sessions 4 and 5)', 'FontSize', 16, 'FontWeight', 'bold');
legend(b_met, {'Accuracy', 'Balanced Accuracy', 'F1-Score', 'AUROC'}, 'Location', 'northwest', 'FontSize', 12, 'Box', 'off');
ylim([0 100]); set(gca, 'Box', 'on', 'YGrid', 'on', 'GridAlpha', 0.3, 'LineWidth', 1.5);

% --- FIGURE 3: BOXPLOT OF INTER-SUBJECT VARIABILITY ---
fig3 = figure('Color', 'w', 'Position', [200 200 1000 600]); hold on;
valid_data_for_plot = acc_eco(~any(isnan(acc_eco), 2), :);
bp = boxplot(valid_data_for_plot, 'Labels', mshort, 'Widths', 0.6);
set(bp, 'LineWidth', 1.5);
h = findobj(gca, 'Tag', 'Box');
for j = 1:length(h)
    patch(get(h(j), 'XData'), get(h(j), 'YData'), palette(n_models+1-j, :), 'FaceAlpha', 0.6, 'EdgeColor', 'k', 'LineWidth', 1.5);
end
n_valid_subj = size(valid_data_for_plot, 1);
rng(1);   % fixed seed for the scatter jitter only
for m = 1:n_models
    x_jitter = m + (rand(n_valid_subj,1)-0.5)*0.2;
    scatter(x_jitter, valid_data_for_plot(:,m), 40, 'k', 'filled', 'MarkerEdgeColor', 'w', 'LineWidth', 0.5);
end
yline(50, '--r', 'Chance Level', 'LineWidth', 2, 'FontSize', 12);
ylabel('Mean Online Accuracy (%)', 'FontWeight', 'bold', 'FontSize', 14);
title('Inter-Subject Variability and Robustness of the Adaptive Models', 'FontSize', 16, 'FontWeight', 'bold');
set(gca, 'FontSize', 13, 'Box', 'on', 'YGrid', 'on', 'GridAlpha', 0.3, 'LineWidth', 1.5);
ylim([40 100]);

% --- FIGURE 4: TEMPORAL TREND ---
fig4 = figure('Color', 'w', 'Position', [250 250 900 600]); hold on;
x_ses = [1, 2, 3];
h4 = gobjects(1, n_models);
for m = 1:n_models
    y_trend = [mean(acc_ses1_model(:,m), 'omitnan'), mean(acc_global(:,m,1), 'omitnan'), mean(acc_global(:,m,2), 'omitnan')];
    h4(m) = plot(x_ses, y_trend, '-o', 'Color', palette(m,:), 'LineWidth', 3, 'MarkerSize', 10, 'MarkerFaceColor', palette(m,:), 'MarkerEdgeColor', 'w');
end
xticks([1 2 3]); xticklabels({'S1 (Offline)', 'S4 (Online)', 'S5 (Online)'});
ylabel('Global Accuracy (%)', 'FontWeight', 'bold', 'FontSize', 14);
title('Accuracy Across Sessions', 'FontSize', 16, 'FontWeight', 'bold');
yline(50, '--k', 'LineWidth', 1.5, 'HandleVisibility', 'off');
legend(h4, model_names, 'Location', 'northeastoutside', 'FontSize', 11); legend('boxoff');
set(gca, 'FontSize', 13, 'Box', 'on', 'YGrid', 'on', 'GridAlpha', 0.3, 'LineWidth', 1.5);
ylim([45 90]); xlim([0.8 3.2]);

% --- FIGURE 5: PER-SUBJECT DETAIL ACROSS THE 3 SESSIONS (appendix) ---
fig5 = figure('Color', 'w', 'Position', [50 50 1600 900]);
t_fig5 = tiledlayout(2, ceil(num_subj/2), 'TileSpacing', 'compact', 'Padding', 'compact');
for i = 1:num_subj
    nexttile; hold on;
    data_subj = [acc_ses1_model(i, :); squeeze(acc_global(i, :, :))'];
    b_sub = bar(data_subj, 'grouped', 'EdgeColor', 'k', 'LineWidth', 0.5, 'BarWidth', 0.85);
    for m = 1:n_models, b_sub(m).FaceColor = palette(m, :); end
    yline(50, '--r', 'LineWidth', 1.5); ylim([35 100]);
    title(sprintf('Subject %02d', subjects(i)), 'FontSize', 12, 'FontWeight', 'bold');
    xticks([1 2 3]); if i > (num_subj - ceil(num_subj/2)), xticklabels({'S1', 'S4', 'S5'}); else, xticklabels({}); end
    if mod(i, ceil(num_subj/2)) == 1, ylabel('Accuracy (%)', 'FontWeight', 'bold', 'FontSize', 12); end
    set(gca, 'FontSize', 11, 'Box', 'on', 'YGrid', 'on', 'GridAlpha', 0.3);
end
lgd5 = legend(b_sub, model_names, 'FontSize', 12, 'FontWeight', 'bold');
lgd5.Layout.Tile = 'east'; legend('boxoff');
title(t_fig5, 'Subject-Specific Analysis (S1, S4, S5)', 'FontSize', 18, 'FontWeight', 'bold');

% --- FIGURE 6: MEAN ONLINE ACCURACY PER SUBJECT ---
fig6 = figure('Color', 'w', 'Position', [100 100 1400 600]); hold on;
valid_idx_plot = ~any(isnan(acc_eco), 2);
acc_eco_valid = acc_eco(valid_idx_plot, :);
subj_valid = subjects(valid_idx_plot);
b_subj = bar(acc_eco_valid, 'grouped', 'EdgeColor', 'k', 'LineWidth', 1);
for m = 1:n_models, b_subj(m).FaceColor = palette(m, :); end
yline(50, '--r', 'Chance Level', 'LineWidth', 2, 'FontSize', 12, 'HandleVisibility', 'off');
ylim([30 100]); xticks(1:length(subj_valid));
xticklabels(arrayfun(@(x) sprintf('S%02d', x), subj_valid, 'UniformOutput', false));
ylabel('Mean Online Accuracy (S4-S5) [%]', 'FontWeight', 'bold', 'FontSize', 14);
title('Longitudinal Performance Comparison per Subject', 'FontSize', 16, 'FontWeight', 'bold');
lgd6 = legend(b_subj, model_names, 'Location', 'southoutside', 'FontSize', 12); lgd6.NumColumns = 4; legend('boxoff');
set(gca, 'FontSize', 12, 'Box', 'on', 'YGrid', 'on', 'GridAlpha', 0.3, 'LineWidth', 1.5);

% --- FIGURE 7: SENSITIVITY vs FALSE-POSITIVE RATE PER SUBJECT (M1 vs M7) ---
fig7 = figure('Color', 'w', 'Position', [100 100 1000 450]);
tl7 = tiledlayout(1, 2, 'TileSpacing', 'compact');
for pp = 1:2
    nexttile; hold on;
    mm = [1 7]; mm = mm(pp);
    scatter(100 - spec_eco(:, mm), sens_eco(:, mm), 70, palette(mm, :), 'filled', 'MarkerEdgeColor', 'k');
    for i = 1:num_subj, text(100 - spec_eco(i, mm) + 1, sens_eco(i, mm), sprintf('%02d', subjects(i)), 'FontSize', 9); end
    plot([0 100], [0 100], '--k');
    xlim([0 100]); ylim([0 100]); axis square; grid on;
    xlabel('False-positive rate (100 - Specificity) [%]'); ylabel('Sensitivity [%]');
    title(model_names{mm});
end
title(tl7, 'Operating point per subject (diagonal = chance)', 'FontWeight', 'bold');

% --- FIGURE 8: ACCURACY vs AUROC SCATTER PLOT (M1 vs M7) ---
fig8 = figure('Color', 'w', 'Position', [150 150 900 600]); hold on;

% Data (mean of S4 and S5); M1 is model 1, M7 is model 7
m1 = 1; m7 = 7;
valid_idx_auc = ~isnan(acc_eco(:, m1)) & ~isnan(acc_eco(:, m7));
acc_m1 = acc_eco(valid_idx_auc, m1);
auc_m1 = auc_eco(valid_idx_auc, m1);
acc_m7 = acc_eco(valid_idx_auc, m7);
auc_m7 = auc_eco(valid_idx_auc, m7);
subj_val = subjects(valid_idx_auc);

% Gray segments joining M1 and M7 for the same subject
for i = 1:length(subj_val)
    plot([auc_m1(i), auc_m7(i)], [acc_m1(i), acc_m7(i)], '-', 'Color', [0.6 0.6 0.6], 'LineWidth', 1, 'HandleVisibility', 'off');
end

% Points for M1 (gray) and M7 (green)
scatter(auc_m1, acc_m1, 80, palette(1,:), 'filled', 'MarkerEdgeColor', 'k', 'DisplayName', 'M1 (static)');
scatter(auc_m7, acc_m7, 80, palette(7,:), 'filled', 'MarkerEdgeColor', 'k', 'DisplayName', 'M7 (adaptive rebias)');

% Subject number labels
for i = 1:length(subj_val)
    text(auc_m1(i) + 0.8, acc_m1(i) - 0.8, sprintf('%02d', subj_val(i)), 'FontSize', 9, 'Color', [0.3 0.3 0.3]);
    text(auc_m7(i) + 0.8, acc_m7(i) - 0.8, sprintf('%02d', subj_val(i)), 'FontSize', 9, 'Color', [0.1 0.4 0.1]);
end

% Chance-level lines (50%)
xline(50, '--r', 'LineWidth', 1.5, 'HandleVisibility', 'off');
yline(50, '--r', 'LineWidth', 1.5, 'HandleVisibility', 'off');

xlabel('AUROC (%)', 'FontWeight', 'bold', 'FontSize', 13);
ylabel('Accuracy (%)', 'FontWeight', 'bold', 'FontSize', 13);
title('Accuracy vs. AUROC per Subject (M1 vs. M7)', 'FontSize', 15, 'FontWeight', 'bold');
legend('Location', 'northwest', 'FontSize', 12);
grid on; set(gca, 'FontSize', 12, 'Box', 'on', 'GridAlpha', 0.3);
xlim([45 100]); ylim([45 100]);

%% ========================================================================
%  7. EXTRA VALIDATION PLOTS (confusion matrices, heatmap, PCA, drift)
% ========================================================================
fprintf('\nGenerating extra validation plots...\n');

% --- GLOBAL CONFUSION MATRICES (M1 vs M7) ---
all_true = []; all_pred_m1 = []; all_pred_m7 = [];
for s = 1:num_subj
    for t = 1:T
        if ~isempty(raw_results{s,t})
            res = raw_results{s,t};
            all_true = [all_true; res.true]; %#ok<AGROW>
            all_pred_m1 = [all_pred_m1; res.pred(:,1)]; %#ok<AGROW>
            all_pred_m7 = [all_pred_m7; res.pred(:,7)]; %#ok<AGROW>
        end
    end
end
if ~isempty(all_true)
    fig_cm = figure('Color', 'w', 'Position', [100, 100, 1000, 450]);
    t_cm = tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
    nexttile; confusionchart(all_true, all_pred_m1, 'RowSummary','row-normalized', 'ColumnSummary','column-normalized', 'Title', 'Model 1 (Static)');
    nexttile; confusionchart(all_true, all_pred_m7, 'RowSummary','row-normalized', 'ColumnSummary','column-normalized', 'Title', 'Model 7 (GEAR)');
    title(t_cm, 'Confusion Matrices Across the Online Sessions (S4+S5)', 'FontSize', 16, 'FontWeight', 'bold');
end

% --- GEODESIC DRIFT (M7) over all subjects, normalized by the distance at the end of calibration ---
maxK = max(cellfun(@numel, drift_curve(:)));
D = nan(num_subj * T, maxK); rr = 0;
for s = 1:num_subj
    for t = 1:T
        if isempty(drift_curve{s,t}), continue; end
        rr = rr + 1;
        D(rr, 1:numel(drift_curve{s,t})) = drift_curve{s,t}' ./ drift_d0(s,t);
    end
end
D = D(1:rr, :);
if rr > 0
    mD = mean(D, 1, 'omitnan'); sD = std(D, 0, 1, 'omitnan') ./ sqrt(sum(~isnan(D), 1));
    fig_drift = figure('Color', 'w', 'Position', [250, 250, 800, 400]); hold on;
    xk = 1:maxK;
    fill([xk fliplr(xk)], [mD + sD, fliplr(mD - sD)], palette(7,:), 'FaceAlpha', 0.25, 'EdgeColor', 'none');
    plot(xk, mD, 'LineWidth', 2, 'Color', palette(7,:));
    yline(1, '--k');
    xlabel('Trials after calibration (S4 and S5, all subjects)');
    ylabel('\delta_R(online mean, train mean) / \delta_R(calib. mean, train mean)');
    title('Shift of the M7 online mean relative to the end of calibration (mean \pm SEM)', 'FontSize', 12, 'FontWeight', 'bold');
    grid on; set(gca, 'FontSize', 12, 'Box', 'on');
end

% --- HEATMAP (first representative subject) and PCA (representative subjects, colored by class) ---
subj_rep_list = [1, 10, 13];
for sr = 1:numel(subj_rep_list)
    subj_rep = subj_rep_list(sr);
    if ~ismember(subj_rep, subjects), continue; end
    try
        [EEG1_rep, lab1_rep, ~] = load_epochs(subj_rep, 1, 'pre', runs, base_path, t_window, bands_fbts, ch_idx);
        N_tot_rep = numel(lab1_rep); idx_tr_rep = 1:round(0.7*N_tot_rep);
        lab_tr_rep = lab1_rep(idx_tr_rep);

        C_tr_rep = create_block_cov(EEG1_rep, idx_tr_rep);
        C_mean_tr_rep = karcher_mean(C_tr_rep);
        C_class1_rep = karcher_mean(C_tr_rep(:,:, lab_tr_rep == 1));
        C_class2_rep = karcher_mean(C_tr_rep(:,:, lab_tr_rep == 2));

        [EEG4_rep, lab4_rep, ~] = load_epochs(subj_rep, 4, 'post', runs, base_path, t_window, bands_fbts, ch_idx);
        C_long_rep = create_block_cov(EEG4_rep, 1:length(lab4_rep));
        C_calib_rep = karcher_mean(C_long_rep(:,:,1:N_calib));

        if sr == 1
            fig_hm = figure('Color', 'w', 'Position', [150, 150, 900, 400]);
            t_hm = tiledlayout(1, 2, 'TileSpacing', 'compact');
            nexttile; imagesc(C_class1_rep); colormap('turbo'); colorbar; axis square;
            title('Class 1 Centroid (MI)'); xlabel('Channels x Bands'); ylabel('Channels x Bands');
            nexttile; imagesc(C_class2_rep); colormap('turbo'); colorbar; axis square;
            title('Class 2 Centroid (Rest)'); xlabel('Channels x Bands'); ylabel('Channels x Bands');
            title(t_hm, sprintf('Structure of the 20x20 SPD Matrix (Filter Bank) - Subject %02d', subj_rep), 'FontSize', 16, 'FontWeight', 'bold');
        end

        X_tr = extract_all_features(C_tr_rep, C_mean_tr_rep);
        X_ts_norebias = extract_all_features(C_long_rep, C_mean_tr_rep);
        X_ts_rebias = zeros(size(X_ts_norebias));
        for i = 1:size(C_long_rep,3)
            C_reb = riemann_rebias(C_long_rep(:,:,i), C_calib_rep, C_mean_tr_rep);
            X_ts_rebias(i,:) = extract_riemann_features(C_reb, C_mean_tr_rep);
        end
        [coeff, score_tr] = pca(X_tr);                       % a single PCA (training), used in both panels
        score_ts_no = (X_ts_norebias - mean(X_tr, 1)) * coeff;
        score_ts_re = (X_ts_rebias - mean(X_tr, 1)) * coeff;

        ax_all = [score_tr(:,1:2); score_ts_no(:,1:2); score_ts_re(:,1:2)];
        xl = [min(ax_all(:,1)) max(ax_all(:,1))] + [-0.2 0.2];
        yl = [min(ax_all(:,2)) max(ax_all(:,2))] + [-0.2 0.2];
        lo = lab4_rep(:); mk = {'o', 's'}; cls_name = {'MI', 'Rest'};

        fig_pca = figure('Color', 'w', 'Position', [200, 200, 1100, 500]);
        t_pca = tiledlayout(1, 2, 'TileSpacing', 'compact');
        for pnl = 1:2
            nexttile; hold on;
            if pnl == 1, sc_on = score_ts_no; col_on = palette(4,:); ttl = 'No Rebias (Model 4)';
            else,        sc_on = score_ts_re; col_on = palette(5,:); ttl = 'Static Rebias (Model 5)'; end
            for c = 1:2
                scatter(score_tr(lab_tr_rep == c, 1), score_tr(lab_tr_rep == c, 2), 50, [0.7 0.7 0.7], mk{c}, 'filled', 'MarkerEdgeColor', 'k', ...
                    'DisplayName', ['S1 - ' cls_name{c}]);
            end
            for c = 1:2
                scatter(sc_on(lo == c, 1), sc_on(lo == c, 2), 50, col_on, mk{c}, 'filled', 'MarkerEdgeColor', 'k', ...
                    'DisplayName', ['S4 - ' cls_name{c}]);
            end
            title(ttl); xlabel('PC1'); ylabel('PC2'); xlim(xl); ylim(yl); grid on; axis square;
            legend('Location', 'best');
        end
        title(t_pca, sprintf('PCA Projection in the Tangent Space - Subject %02d (circle = MI, square = Rest)', subj_rep), 'FontSize', 14, 'FontWeight', 'bold');
    catch ME
        fprintf('Could not generate extra plots for Subj %02d: %s\n', subj_rep, ME.message);
    end
end

diary off;
fprintf('\nDONE. Output files are in %s\n', out_dir);

%% ================= HELPER FUNCTIONS =================
function print_matrix(title_str, M, row_labels, col_labels, fmt)
    fprintf('\n==============================================================================================\n');
    fprintf(' %s\n', title_str);
    fprintf('==============================================================================================\n');
    fprintf('%-10s', 'Subject');
    for j = 1:numel(col_labels), fprintf(' | %-8s', col_labels{j}); end
    fprintf('\n');
    fprintf('%s\n', repmat('-', 1, 10 + 11 * numel(col_labels)));
    for i = 1:size(M, 1)
        fprintf('%-10s', row_labels{i});
        for j = 1:size(M, 2)
            if isnan(M(i,j)), fprintf(' | %-8s', 'NaN'); else, fprintf([' | ' fmt], M(i,j)); end
        end
        fprintf('\n');
    end
    fprintf('%-10s', 'Mean');
    mm = mean(M, 1, 'omitnan');
    for j = 1:numel(mm), fprintf([' | ' fmt], mm(j)); end
    fprintf('\n');
end

function s = fmt_p(p)
    if isnan(p), s = 'NaN';
    elseif p <= 0.001, s = '<=0.001';
    else, s = sprintf('%.4f', p); end
end

function run_stat_tests(X, metric_name, pairs, do_global)
    mnames = {'M1','M2','M3','M4','M5','M6','M7'};
    X = X(~any(isnan(X), 2), :);
    [N, k] = size(X);
    fprintf('\n##############################################################\n');
    fprintf(' STATISTICS: %s (N = %d subjects)\n', metric_name, N);
    fprintf('##############################################################\n');
    run_posthoc = true;
    if do_global
        ws = warning('off', 'all');
        p_norm = nan(1, k);
        for m = 1:k, [~, p_norm(m)] = lillietest(X(:, m)); end
        warning(ws);
        fprintf('Lilliefors (p<=0.001 = outside table): ');
        for m = 1:k, fprintf('%s: %s | ', mnames{m}, fmt_p(p_norm(m))); end
        fprintf('\n');
        [p_f, tbl, st] = friedman(X, 1, 'off');
        chi2 = tbl{2, 5}; dfv = tbl{2, 3};
        fprintf('Friedman: chi2(%d) = %.2f, p = %s, Kendall W = %.3f\n', dfv, chi2, fmt_p(p_f), chi2 / (N * (k - 1)));
        tmp = [mnames(1:k); num2cell(st.meanranks)];
        fprintf('Mean ranks: '); fprintf('%s = %.2f  ', tmp{:}); fprintf('\n');
        run_posthoc = (p_f < 0.05);
    end
    if ~run_posthoc
        fprintf('Friedman not significant: post-hoc tests not run.\n');
        return;
    end
    np = size(pairs, 1);
    p_raw = nan(np, 1); Wv = nan(np, 1); nsr = nan(np, 1); rb = nan(np, 1); dmean = nan(np, 1); dmed = nan(np, 1);
    for i = 1:np
        a = X(:, pairs(i, 1)); b = X(:, pairs(i, 2));
        d = b - a;                       % positive = the second model is better
        dmean(i) = mean(d); dmed(i) = median(d);
        nz = abs(d) > 1e-9;
        nsr(i) = sum(nz);
        if nsr(i) > 0
            rk = tiedrank(abs(d(nz)));
            Wv(i) = sum(sign(d(nz)) .* rk);
            rb(i) = Wv(i) / (nsr(i) * (nsr(i) + 1) / 2);
            try
                p_raw(i) = signrank(a, b, 'method', 'exact');
            catch
                p_raw(i) = signrank(a, b);
            end
        else
            p_raw(i) = 1;
        end
    end
    [p_sorted, sort_idx] = sort(p_raw);
    q = p_sorted .* (np ./ (1:np)');
    for i = np-1:-1:1, q(i) = min(q(i), q(i+1)); end
    q = min(q, 1);
    p_fdr = zeros(np, 1); p_fdr(sort_idx) = q;
    fprintf('Wilcoxon signed-rank, two-sided, exact p; Delta = second minus first model; Benjamini-Hochberg FDR over m = %d\n', np);
    fprintf('%-10s | %9s | %13s | %4s | %8s | %8s | %-9s | %-9s | %-6s\n', 'Pair', 'Mean D', 'Median D', 'n', 'W', 'r (RB)', 'p raw', 'p FDR', 'Sig.');
    for i = 1:np
        sig = 'NO'; if p_fdr(i) < 0.05, sig = 'YES (*)'; end
        fprintf('%-10s | %9.2f | %13.2f | %4d | %8.1f | %8.2f | %-9s | %-9s | %-6s\n', ...
            sprintf('M%d vs M%d', pairs(i,1), pairs(i,2)), dmean(i), dmed(i), nsr(i), Wv(i), rb(i), fmt_p(p_raw(i)), fmt_p(p_fdr(i)), sig);
    end
end

function [beta, K, Q, A, b] = train_elm(X, Y, C, L, nw)
    stream = RandStream('mt19937ar', 'Seed', 1); A = rand(stream, L, size(X,2)) * 2 - 1; b = rand(stream, 1, L) * 2 - 1;
    H = 1 ./ (1 + exp(-(X * A' + repmat(b, size(X,1), 1)))); w = ones(size(X,1), 1);
    if nw, for c=1:size(Y,2), w(Y(:,c)==1) = size(X,1)/(2*sum(Y(:,c)==1)); end; else, w = w / size(X,1); end
    K = H' * (w .* H); Q = H' * (w .* Y); beta = (K + eye(L) / C) \ Q;
end

function C_blk = create_block_cov(EEGb, idx)
    nb = length(EEGb); N_ch = size(EEGb{1}, 1); N_t = length(idx); C_blk = zeros(N_ch * nb, N_ch * nb, N_t);
    for i = 1:N_t
        C_temp = cell(1, nb); for b = 1:nb, C = cov(double(EEGb{b}(:,:,idx(i)))'); C_temp{b} = C + eye(N_ch) * 1e-4 * trace(C); end
        C_blk(:,:,i) = blkdiag(C_temp{:});
    end
end

function [C_blk, gam] = create_ledoit_wolf_block_cov(EEGb, idx)
    nb = length(EEGb); N_ch = size(EEGb{1}, 1); N_t = length(idx);
    C_blk = zeros(N_ch * nb, N_ch * nb, N_t); gam = zeros(N_t, nb);
    for i = 1:N_t
        C_temp = cell(1, nb);
        for b = 1:nb
            X = double(EEGb{b}(:,:,idx(i)))'; [N, P] = size(X); S_emp = cov(X); mu = trace(S_emp) / P; Target = mu * eye(P);
            X_centered = X - mean(X, 1); var_S = 0;
            for j = 1:N, obs_cov = X_centered(j,:)' * X_centered(j,:); var_S = var_S + norm(obs_cov - S_emp, 'fro')^2; end
            var_S = var_S / (N^2); dist_T_S = norm(S_emp - Target, 'fro')^2;
            gamma = max(0, min(1, var_S / max(dist_T_S, eps)));
            gam(i, b) = gamma;
            S_final = (1 - gamma) * S_emp + gamma * Target; S_final = (S_final + S_final') / 2; C_temp{b} = S_final + eye(P) * 1e-6;
        end
        C_blk(:,:,i) = blkdiag(C_temp{:});
    end
end

function C_m = karcher_mean(C, tol, max_it)
    if nargin<2, tol=1e-6; end; if nargin<3, max_it=50; end
    N = size(C, 3); C_m = mean(C, 3);
    for iter = 1:max_it
        C_isq = sym_fun(C_m, @(x) 1./sqrt(max(x, eps))); C_sq = sym_fun(C_m, @(x) sqrt(max(x, eps))); L = zeros(size(C_m));
        for i=1:N, L = L + sym_fun(C_isq * C(:,:,i) * C_isq, @(x) log(max(x, eps))); end
        L = L / N; C_m = C_sq * sym_fun(L, @exp) * C_sq; C_m = (C_m + C_m')/2; if norm(L, 'fro') < tol, break; end
    end
end

function C_al = riemann_rebias(C, C_cal, C_tr)
    C_c_isq = sym_fun(C_cal, @(x) 1./sqrt(max(x, eps))); C_tr_sq = sym_fun(C_tr, @(x) sqrt(max(x, eps)));
    C_al = C_tr_sq * (C_c_isq * C * C_c_isq) * C_tr_sq; C_al = (C_al + C_al')/2;
end

function d = riemann_distance(C1, C2)
    C1_inv_sqrt = sym_fun(C1, @(x) 1./sqrt(max(x, eps))); M = C1_inv_sqrt * C2 * C1_inv_sqrt; d = norm(sym_fun(M, @(x) log(max(x, eps))), 'fro');
end

function X = extract_all_features(C_all, C_ref)
    N = size(C_all, 3); X = zeros(N, size(C_all,1)*(size(C_all,1)+1)/2); for i=1:N, X(i,:) = extract_riemann_features(C_all(:,:,i), C_ref); end
end

function f = extract_riemann_features(C, C_ref)
    M = sym_fun(C_ref, @(x) 1./sqrt(max(x, eps))) * C * sym_fun(C_ref, @(x) 1./sqrt(max(x, eps)));
    Log_M = sym_fun(M, @(x) log(max(x, eps))); n = size(Log_M, 1); S = sqrt(2) * Log_M; S(1:n+1:end) = diag(Log_M); f = S(triu(true(n)))';
end

function R = sym_fun(M, f)
    M = (M + M') / 2; [V, D] = eig(M); d = real(diag(D)); R = V * diag(f(max(d, eps))) * V'; R = (R + R') / 2;
end

function S = elm_scores(X, beta, A, b)
    S = (1 ./ (1 + exp(-(X * A' + repmat(b, size(X, 1), 1))))) * beta;
end

function [beta_n, K_n, Q_n] = update_rwos_elm(x, K_o, Q_o, A, b, C, yt, lam, w)
    h = 1 ./ (1 + exp(-(x * A' + repmat(b, size(x, 1), 1)))); K_n = lam * K_o + w * (h' * h); Q_n = lam * Q_o + w * (h' * yt); beta_n = (K_n + eye(size(A, 1)) / C) \ Q_n;
end

function [EEGb, labels, runid] = load_epochs(subj, ses, ses_name, runs, base_path, t_window, bands, ch_idx)
    % Loads every run of a session, extracts one epoch per trigger (MI = 3, Rest = 9),
    % and band-pass filters the data for each band of the filter bank.
    % For each run it prints the number of epochs (MI/Rest) or the error.
    nb = size(bands, 1); cells = repmat({{}}, 1, nb); lab_cells = {}; run_cells = {};
    for r = runs
        filename = sprintf('sub-%02d_%s_run-%d_eeg.set', subj, ses_name, r);
        filepath = fullfile(base_path, sprintf('sub-%02d', subj), sprintf('ses-%d', ses), 'eeg');
        try
            EEG0 = pop_loadset('filename', filename, 'filepath', filepath); EEG0 = pop_select(EEG0, 'channel', ch_idx);
            fs_local = EEG0.srate; lat = []; lab = [];
            for i = 1:numel(EEG0.event)
                l = 0; ev = EEG0.event(i).type; if isnumeric(ev), ev=num2str(ev); else, ev=char(ev); end; ev=strtrim(ev);
                if strcmp(ev, '3') || strcmpi(ev, 'Target_MI'), l = 1; elseif strcmp(ev, '9') || strcmpi(ev, 'Target_Rest'), l = 2; end
                if l>0, lat(end+1,1)=EEG0.event(i).latency; lab(end+1,1)=l; end
            end
            if isempty(lat)
                fprintf('       [WARNING] sub-%02d ses-%d run-%d: no trigger 3/9 found, run not used\n', subj, ses, r);
                continue;
            end
            [lat, order] = sort(lat, 'ascend'); lab = lab(order); c = round(lat);
            vm = (c >= 1) & (c + round(t_window(2)*fs_local) <= size(EEG0.data, 2)); n_drop = sum(~vm);
            c = c(vm); lab = lab(vm);
            fprintf('       run-%d: %d epochs (MI = %d, Rest = %d)', r, numel(c), sum(lab == 1), sum(lab == 2));
            if n_drop > 0, fprintf(' | %d dropped (window beyond end of recording)', n_drop); end
            fprintf('\n');
            for b = 1:nb
                Eb = pop_eegfiltnew(EEG0, bands(b, 1), bands(b, 2)); data = double(Eb.data); ep = cell(1, numel(c));
                for e = 1:numel(c), ep{e} = data(:, c(e) : c(e) + round(t_window(2)*fs_local)); end; cells{b} = [cells{b}, ep];
            end
            lab_cells{end+1} = lab; run_cells{end+1} = r * ones(numel(c), 1);
        catch ME
            fprintf('       [WARNING] sub-%02d ses-%d run-%d NOT used: %s\n', subj, ses, r, ME.message);
        end
    end
    EEGb = cell(1, nb); if isempty(lab_cells), for b=1:nb, EEGb{b}=[]; end; labels=[]; runid=[]; return; end
    for b = 1:nb, EEGb{b} = cat(3, cells{b}{:}); end; labels = vertcat(lab_cells{:}); runid = vertcat(run_cells{:});
end