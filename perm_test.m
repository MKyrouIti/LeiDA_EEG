function out = perm_test(GroupsTS, labels, K, nPerm, includeBaseline, GroupsTS_Bas)
% PERM_TEST — Liked vs Disliked (and optional Baseline) 
%   out.POC.all.(Bas/Dis/Lik/Stim)          [N x K]
%   out.TPM.row.all.(Bas/Dis/Lik/Stim)      [N x K x K]
%   out.TPM.glob.all.(Bas/Dis/Lik/Stim)     [N x K x K]
% Comparisons:
%   - Lik_vs_Dis
%   - If includeBaseline=true:
%       Bas_vs_Dis, Bas_vs_Lik, Bas_vs_Stim

if nargin < 5 || isempty(includeBaseline), includeBaseline = false; end
Ntrials = size(GroupsTS,1);

rng(42);

% compute per-trial state counts and transition counts
% -------------------------------------------------------------------------
    function [StateCounts, TransCounts] = compute_counts(GTS)
        N = size(GTS,1);
        StateCounts = zeros(N,K,'uint32');
        TransCounts = zeros(N,K,K,'uint32');
        for ii = 1:N
            s = GTS(ii,:);
            % counts for POC
            StateCounts(ii,:) = histcounts(s, 0.5+(0:K));
            % transitions for TPM
            if numel(s) > 1
                i = s(1:end-1);
                j = s(2:end);
                for t = 1:numel(i)
                    TransCounts(ii, i(t), j(t)) = TransCounts(ii, i(t), j(t)) + 1;
                end
            end
        end
    end


% per-trial normalized POC, TPM_row (P(j|i)), TPM_glob (P(i,j))
% -------------------------------------------------------------------------
    function [POC_trials, TPMrow_trials, TPMglob_trials] = normalize_all(SC, TC)
        N = size(SC,1);
        POC_trials     = zeros(N, K);
        TPMrow_trials  = zeros(N, K, K);
        TPMglob_trials = zeros(N, K, K);
        for n = 1:N
            % --- POC per trial ---
            sc = double(SC(n,:));
            tot_sc = sum(sc);
            if tot_sc > 0
                POC_trials(n,:) = sc / tot_sc;
            end

            % --- TPMs per trial ---
            tc = double(squeeze(TC(n,:,:)));

            % Row-normalized (conditional)
            rowSums = sum(tc,2);                 % Kx1
            nz = rowSums > 0;
            if any(nz)
                trow = zeros(K,K);
                trow(nz,:) = tc(nz,:) ./ rowSums(nz);
                TPMrow_trials(n,:,:) = trow;
            end

            % Global-normalized (joint)
            totTrans = sum(tc(:));
            if totTrans > 0
                TPMglob_trials(n,:,:) = tc / totTrans;
            end
        end
    end


    function [poc, tpm_row, tpm_glob] = aggregate(mask, SC, TC)
        sel = find(mask);
        nSel = numel(sel);

        poc      = zeros(1,K);
        tpm_row  = zeros(K,K);
        tpm_glob = zeros(K,K);

        for n = sel'
            sc = double(SC(n,:));
            tot_sc = sum(sc);
            if tot_sc > 0
                poc = poc + sc / tot_sc;
            end

            % TPMs from per-trial transitions 
            tc = double(squeeze(TC(n,:,:)));

            % Row-normalized (conditional P(j|i))
            rowSums = sum(tc,2);           % Kx1
            tcRow = zeros(K,K);
            nz = rowSums > 0;
            tcRow(nz,:) = tc(nz,:) ./ rowSums(nz);
            tpm_row = tpm_row + tcRow;

            % Global-normalized (joint P(i,j))
            totTrans = sum(tc(:));
            if totTrans > 0
                tpm_glob = tpm_glob + tc / totTrans;
            end
        end

        poc      = poc / nSel;
        tpm_row  = tpm_row / nSel;
        tpm_glob = tpm_glob / nSel;
    end


[SC, TC]   = compute_counts(GroupsTS);
idxLik     = labels == 1;
idxDis     = labels == 0;

[POC_stim_all, TPMrow_stim_all, TPMglob_stim_all] = normalize_all(SC, TC);
[pocLik, tpmLik_row, tpmLik_glob] = aggregate(idxLik, SC, TC);
[pocDis, tpmDis_row, tpmDis_glob] = aggregate(idxDis, SC, TC);

out.POC.mean.Lik          = pocLik;
out.POC.mean.Dis          = pocDis;
out.TPM.row.mean.Lik      = tpmLik_row;
out.TPM.row.mean.Dis      = tpmDis_row;
out.TPM.glob.mean.Lik     = tpmLik_glob;
out.TPM.glob.mean.Dis     = tpmDis_glob;


out.POC.all.Stim          = POC_stim_all;                      % [Ntrials x K]
out.TPM.row.all.Stim      = TPMrow_stim_all;                   % [Ntrials x K x K]
out.TPM.glob.all.Stim     = TPMglob_stim_all;                  % [Ntrials x K x K]

out.POC.all.Lik           = POC_stim_all(idxLik,:);            % [N_lik x K]
out.TPM.row.all.Lik       = TPMrow_stim_all(idxLik,:,:);       % [N_lik x K x K]
out.TPM.glob.all.Lik      = TPMglob_stim_all(idxLik,:,:);      % [N_lik x K x K]

out.POC.all.Dis           = POC_stim_all(idxDis,:);            % [N_dis x K]
out.TPM.row.all.Dis       = TPMrow_stim_all(idxDis,:,:);       % [N_dis x K x K]
out.TPM.glob.all.Dis      = TPMglob_stim_all(idxDis,:,:);      % [N_dis x K x K]

if includeBaseline
    [SC_bas, TC_bas] = compute_counts(GroupsTS_Bas);

    [POC_bas_all, TPMrow_bas_all, TPMglob_bas_all] = normalize_all(SC_bas, TC_bas);
    [pocBas, tpmBas_row, tpmBas_glob] = aggregate(true(size(SC_bas,1),1), SC_bas, TC_bas);

    out.POC.mean.Bas      = pocBas;
    out.TPM.row.mean.Bas  = tpmBas_row;
    out.TPM.glob.mean.Bas = tpmBas_glob;

    [pocStim, tpmStim_row, tpmStim_glob] = aggregate(true(size(GroupsTS,1),1), SC, TC);
    out.POC.mean.Stim      = pocStim;
    out.TPM.row.mean.Stim  = tpmStim_row;
    out.TPM.glob.mean.Stim = tpmStim_glob;
    out.POC.all.Bas       = POC_bas_all;                       % [Nbase x K]
    out.TPM.row.all.Bas   = TPMrow_bas_all;                    % [Nbase x K x K]
    out.TPM.glob.all.Bas  = TPMglob_bas_all;                   % [Nbase x K x K]
end


% Define comparisons 
% -------------------------------------------------------------------------
if includeBaseline
    comparisons = {
        'Bas_vs_Dis',  GroupsTS_Bas, GroupsTS(idxDis,:), zeros(size(GroupsTS_Bas,1),1), ones(sum(idxDis),1);
        'Bas_vs_Lik',  GroupsTS_Bas, GroupsTS(idxLik,:), zeros(size(GroupsTS_Bas,1),1), ones(sum(idxLik),1);
        'Bas_vs_Stim', GroupsTS_Bas, GroupsTS,           zeros(size(GroupsTS_Bas,1),1), ones(size(GroupsTS,1),1);
        'Lik_vs_Dis',  [],            GroupsTS,          [],                             1 - labels};
else
    comparisons = {'Lik_vs_Dis', [], GroupsTS, [], 1 - labels};
end


% Permutation tests + per-comparison, per-metric BH–FDR
for c = 1:size(comparisons,1)
    cname = comparisons{c,1};
    fprintf('\nRunning comparison: %s\n', cname);

    GTS_A = comparisons{c,2};
    GTS_B = comparisons{c,3};
    labA  = comparisons{c,4};
    labB  = comparisons{c,5};

    if ~isempty(GTS_A)
        [SC_A, TC_A] = compute_counts(GTS_A);
    else
        SC_A = []; TC_A = [];
    end
    [SC_B, TC_B] = compute_counts(GTS_B);

    SCx = cat(1, SC_A, SC_B);
    TCx = cat(1, TC_A, TC_B);
    labelsComp = [labA; labB];           % 0 for group A, 1 for group B

    % Observed group means & differences
    g1 = (labelsComp == 0);
    g2 = (labelsComp == 1);

    [poc1, tpm1_row, tpm1_glob] = aggregate(g1, SCx, TCx);
    [poc2, tpm2_row, tpm2_glob] = aggregate(g2, SCx, TCx);

    diffPOC_true      = poc1     - poc2;        % 1xK (A - B)
    diffTPM_row_true  = tpm1_row - tpm2_row;    % KxK (A - B)
    diffTPM_glob_true = tpm1_glob- tpm2_glob;   % KxK (A - B)

    % Null via label permutations
    Ncomp = numel(labelsComp);
    diffPOC_null      = zeros(nPerm, K);
    diffTPM_row_null  = zeros(nPerm, K, K);
    diffTPM_glob_null = zeros(nPerm, K, K);

    for p = 1:nPerm
        perm = randperm(Ncomp);
        permLabels = labelsComp(perm);
        [pocA, tpmA_row, tpmA_glob] = aggregate(permLabels==0, SCx, TCx);
        [pocB, tpmB_row, tpmB_glob] = aggregate(permLabels==1, SCx, TCx);

        diffPOC_null(p,:)        = pocA     - pocB;
        diffTPM_row_null(p,:,:)  = tpmA_row - tpmB_row;
        diffTPM_glob_null(p,:,:) = tpmA_glob - tpmB_glob;
    end

    % raw two-sided p-values 
    % POC: 1xK
    p_poc = mean(abs(diffPOC_null) >= abs(diffPOC_true), 1);
    true_row_vec = diffTPM_row_true(:).';                  % 1 x K^2
    null_row_vec = reshape(diffTPM_row_null, nPerm, []);   % nPerm x K^2
    p_tpm_row_vec = mean(abs(null_row_vec) >= abs(true_row_vec), 1);   % 1 x K^2
    p_tpm_row_mat = reshape(p_tpm_row_vec, K, K);

    true_glob_vec = diffTPM_glob_true(:).';                % 1 x K^2
    null_glob_vec = reshape(diffTPM_glob_null, nPerm, []); % nPerm x K^2
    p_tpm_glob_vec = mean(abs(null_glob_vec) >= abs(true_glob_vec), 1);% 1 x K^2
    p_tpm_glob_mat = reshape(p_tpm_glob_vec, K, K);

    out.stats.(cname).POC.diff       = diffPOC_true;
    out.stats.(cname).TPM.row.diff   = diffTPM_row_true;
    out.stats.(cname).TPM.glob.diff  = diffTPM_glob_true;

    out.stats.(cname).POC.p_raw      = p_poc;           % 1xK
    out.stats.(cname).TPM.row.p_raw  = p_tpm_row_mat;   % KxK
    out.stats.(cname).TPM.glob.p_raw = p_tpm_glob_mat;  % KxK

    % BH–FDR
    q_poc       = mafdr(p_poc(:),        'BHFDR', true).';     % 1xK
    q_row_mat   = reshape(mafdr(p_tpm_row_vec(:),  'BHFDR', true),  K, K);
    q_glob_mat  = reshape(mafdr(p_tpm_glob_vec(:), 'BHFDR', true),  K, K);

    out.stats.(cname).POC.p_fdr      = q_poc;
    out.stats.(cname).TPM.row.p_fdr  = q_row_mat;   
    out.stats.(cname).TPM.glob.p_fdr = q_glob_mat;   
end

% Plotting
fprintf('\nGenerating plots...\n');
comp_names = fieldnames(out.stats);

for c = 1:numel(comp_names)
    cname = comp_names{c};
    fprintf('Plotting %s ...\n', cname);

    conds = split(cname,'_vs_');
    condA = conds{1}; condB = conds{2};

    pocA = out.POC.mean.(condA);   pocB = out.POC.mean.(condB);
    tpmA_row = out.TPM.row.mean.(condA); tpmB_row = out.TPM.row.mean.(condB);
    tpmA_glob= out.TPM.glob.mean.(condA); tpmB_glob= out.TPM.glob.mean.(condB);

    names = struct('Bas','Baseline','Dis','Disliked','Lik','Liked','Stim','Stimulus');
    condA_full = names.(condA);
    condB_full = names.(condB);

    %  POC 
    plot_POC_per_state(pocA, pocB, out.stats.(cname).POC.p_fdr, condA_full, condB_full);
    sgtitle(sprintf('%s vs %s', condA_full, condB_full));

    % TPM Row 
    plot_TPM_pair(tpmA_row, tpmB_row, out.stats.(cname).TPM.row.p_fdr, ...
        sprintf('%s (Conditional probabilites)', condA_full), sprintf('%s (Conditional probabilites)', condB_full));

    % TPM Global
    plot_TPM_pair(tpmA_glob, tpmB_glob, out.stats.(cname).TPM.glob.p_fdr, ...
        sprintf('%s (Joint probabilites)', condA_full), sprintf('%s (Joint probabilites)', condB_full));

    plot_TPM_diff(out.stats.(cname).TPM.row.diff,  out.stats.(cname).TPM.row.p_fdr, ...
        sprintf('Difference in switching conditional probabilites (%s vs %s):', condA_full, condB_full));
    plot_TPM_diff(out.stats.(cname).TPM.glob.diff, out.stats.(cname).TPM.glob.p_fdr, ...
        sprintf('Difference in switching joint probabilites (%s vs %s):', condA_full, condB_full));
end
fprintf('All comparisons plotted.\n');

end 


%  PLOTTING HELPERS
function plot_POC_per_state(pocA, pocB, q_poc, condA_name, condB_name)
K = numel(pocA);
cols = ceil(sqrt(K));
rows = ceil(K/cols);

figure('Name','POC Comparison','Color','w');
tiledlayout(rows, cols, 'TileSpacing','compact','Padding','compact');
for k = 1:K
    nexttile;
    vals = [pocA(k), pocB(k)];
    bar([1 2], vals, 0.6, 'FaceColor',[0.5 0.5 0.5], 'EdgeColor','none'); hold on;
    set(gca, 'XTick', [1 2], 'XTickLabel', {condA_name, condB_name}, ...
        'TickDir', 'out', 'Box', 'off', 'FontSize', 8);
    ylabel('Probability of Occurrence', 'FontSize', 8);
    title(sprintf('State %d', k), 'FontWeight', 'normal', 'FontSize', 9);
    ylim([0, max(vals)*1.3 + eps]);
    p = q_poc(k); % FDR-adjusted
    if ~isnan(p)
        if p < 0.05, stars = '*';
        elseif p < 0.10, stars = '.';
        else, stars = ''; end
        if ~isempty(stars)
            yl = ylim;
            ybar = max(vals)*1.1;
            line([1 2], [ybar ybar], 'Color', 'k', 'LineWidth', 1);
            text(1.5, ybar + 0.015*range(yl), stars, ...
                'HorizontalAlignment','center','FontWeight','bold','FontSize',10);
        end
    end
    hold off;
end
end

function plot_TPM_pair(T1, T2, q_mat, condA_name, condB_name)
K = size(T1,1);
vmin = 0; vmax = max([T1(:); T2(:)]);
if ~isfinite(vmax) || vmax <= 0, vmax = 1; end

figure('Name','TPM Comparison','Color','w');
tiledlayout(1,2,'TileSpacing','compact','Padding','compact');

nexttile;
imagesc(T1,[vmin vmax]); set(gca,'Color','w');
axis image; xlabel('To state'); ylabel('From state');
title(condA_name); cb1 = colorbar; ylabel(cb1,'Probability');
hold on; render_bins_and_stars(T1, q_mat); hold off;

nexttile;
imagesc(T2,[vmin vmax]); set(gca,'Color','w');
axis image; xlabel('To state'); ylabel('From state');
title(condB_name); cb2 = colorbar; ylabel(cb2,'Probability');
hold on; render_bins_and_stars(T2, q_mat); hold off;

colormap(parula);
end

function plot_TPM_diff(D, q_mat, ttl)
K = size(D,1);
maxabs = max(abs(D(:)));
if ~isfinite(maxabs) || maxabs == 0, maxabs = 1; end

figure('Name',[ttl ' (difference)'],'Color','w');
imagesc(D, [-maxabs, maxabs]); set(gca,'Color','w');
axis image; xticks(1:K); yticks(1:K);
xlabel('To state'); ylabel('From state');
title(ttl);
cb = colorbar; ylabel(cb, 'Difference in Probability ');

% Numbers + stars
hold on; render_bins_and_stars(D, q_mat); hold off;
colormap(parula); 
end

function render_bins_and_stars(M, q_mat)
K = size(M,1);
for i = 1:K
    for j = 1:K
        val = M(i,j);
        text(j, i, sprintf('%.3f', val), ...
            'HorizontalAlignment','center','VerticalAlignment','middle', ...
            'FontWeight','bold','Color','k','FontSize',9);

        q = q_mat(i,j);
        if isnan(q), continue; end
        if q < 0.05, stars = '*';
        elseif q < 0.10, stars = '.';
        else, stars = ''; end
        if ~isempty(stars)
            text(j, i-0.25, stars, 'HorizontalAlignment','center', ...
                'VerticalAlignment','middle', 'FontWeight','bold','Color','k','FontSize',11);
        end
    end
end
end