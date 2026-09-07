% Export unmodified upstream dynamics/clustering for Python parity checks.
% Docker: mount the repository at /work and set SCRIPT to this file.
% Requires GNU Octave's statistics package (pdist2 and kmeans).
root = fileparts(fileparts(mfilename('fullpath')));
upstream = fullfile(root, 'matlab', 'liboniEA2025image');
addpath(fullfile(upstream, 'graphs'));
addpath(fullfile(upstream, 'image_segmentation'));
if exist('OCTAVE_VERSION', 'builtin')
    pkg load statistics;
    if exist('rng', 'file') == 0
        addpath(fullfile(root, 'matlab', 'octave_compat'));
    end
end
source_files = {
    'graphs/gaussian_sheet.m', ...
    'image_segmentation/run_2layer.m', ...
    'image_segmentation/spatiotemporal_segmentation.m'};
source_text = cellfun(@(path) fileread(fullfile(upstream, path)), ...
    source_files, 'UniformOutput', false);
alpha = [0.5, 0.5];
sigma = [0.9, 0.0313];
nt = [60, 200];
window_size = 40;
window_step = 40;
dim = [1, 2, 3];
output_dir = fullfile(root, 'datasets');
names = {'2shapes', '3shapes', 'natural'};
files = {'2shapes.mat', '3shapes.mat', 'natural_image.mat'};
seeds = [1, 9, 1];
cluster_counts = [2, 3, 2];
for example = 1:numel(names)
    data = load(fullfile(output_dir, files{example}));
    if strcmp(names{example}, 'natural')
        im = data.im;
    else
        im = data.images(:, :, 1);
    end
    seed = seeds(example);
    n_clusters = cluster_counts(example);
    fprintf('Running %s, seed %d\n', names{example}, seed);
    [save_x, mask] = run_2layer(im, alpha, sigma, nt, seed);
    initial_state = save_x(:, 1); % Share actual draws, not cross-runtime seeds.
    phase_states = exp(1i * angle(save_x(~mask, :)));
    [rho, V, D, projection] = spatiotemporal_segmentation( ...
        phase_states, dim, nt, window_size, window_step);
    rho_final = rho(:, :, end);
    v_final = V(:, :, end);
    % Canonicalize each eigenvector's arbitrary global phase (rotate so its
    % largest-magnitude entry is real and positive) so the projection below
    % matches Python's backend-independent convention, not this LAPACK
    % binding's raw phase. Upstream spatiotemporal_segmentation.m itself is
    % untouched; this only post-processes its returned V.
    [~, pivot_rows] = max(abs(v_final), [], 1);
    pivots = v_final(sub2ind(size(v_final), pivot_rows, 1:size(v_final, 2)));
    v_final = v_final .* (conj(pivots) ./ abs(pivots));
    projection_final = real(rho_final) * real(v_final(:, dim));
    rng(seed);
    labels = kmeans(projection_final, n_clusters);
    cluster_map = -ones(size(im));
    cluster_map(~reshape(mask, size(im))) = labels - 1;
    reference_runtime = version;
    schema_version = 2; % v2: canonicalized eigenvector phase (see above).
    destination = fullfile(output_dir, [names{example}, '_ref.mat']);
    save('-v7', destination, 'source_files', 'source_text', ...
        'schema_version', 'reference_runtime', ...
        'im', 'initial_state', 'alpha', 'sigma', 'nt', 'seed', ...
        'n_clusters', 'window_size', 'window_step', 'save_x', 'mask', ...
        'cluster_map', 'rho_final', 'projection_final');
    fprintf('Saved %s (%d foreground pixels)\n', destination, sum(~mask));
    clear save_x rho V D projection phase_states rho_final projection_final v_final;
end
