clear;
clc;

projectRoot = fileparts(mfilename('fullpath'));
dataDir = fullfile(projectRoot, 'data');

params = build_assignment_parameters();
stateInfo = state_definition();

eventFiles = discover_comtrade_pairs(dataDir);
fprintf('Discovered %d COMTRADE event(s) in %s\n', numel(eventFiles), dataDir);

results = struct([]);

for idx = 1:numel(eventFiles)
    cfgPath = eventFiles(idx).cfg;
    datPath = eventFiles(idx).dat;
    data = read_comtrade(cfgPath, datPath);

    params.time_step = estimate_time_step(data.time, data.sample_rate_hz);
    params.gs1 = 0.5 * params.time_step / (2 * params.L1);
    params.gs2 = 0.5 * params.time_step / (2 * params.L2);
    params.gs3 = 0.5 * params.time_step / (2 * params.L3);

    data.baseline_current = (data.v_out ./ params.Rb) .* params.n;

    sampleIndex = min(max(params.wls_sample_index, 2), numel(data.v_out));
    measurement = build_single_step_measurement(data, sampleIndex, params);
    xPrev = build_initial_guess(data, sampleIndex - 1, params);
    x0 = build_initial_guess(data, sampleIndex, params);

    weights = diag(1 ./ (measurement.sigma .^ 2));
    wlsResult = solve_wls_step(measurement.z, x0, xPrev, params, weights, params.wls_options);
    trajectory = run_time_series_estimation(data, params, stateInfo);

    data.estimated_current = trajectory.state(:, stateInfo.index.ip);
    data.estimated_burden_current = trajectory.state(:, stateInfo.index.ibm);
    data.estimated_magnetizing_current = trajectory.state(:, stateInfo.index.im);
    data.estimated_flux = trajectory.state(:, stateInfo.index.lambda);

    plotSpec = struct();
    plotSpec.showBaseline = true;
    plotSpec.showEstimated = true;
    plotSpec.showDiagnostics = true;
    plotSpec.figureName = sprintf('%s Overview', data.event_name);
    plotSpec.currentLabel = sprintf('Current Estimates, R_b = %.3f ohm, n = %.1f', ...
        params.Rb, params.n);
    plotSpec.diagnosticTime = trajectory.time;
    plotSpec.residualNorm = trajectory.final_residual_norm;
    plotSpec.chiSquare = trajectory.chi_square;
    plotSpec.showResidualBreakdown = true;
    plotSpec.residualMatrix = trajectory.residual;
    plotSpec.residualLabels = measurement.labels;
    plot_comtrade(data, plotSpec);

    fprintf('\nEvent: %s\n', data.event_name);
    fprintf('  Samples: %d\n', numel(data.time));
    fprintf('  Time step: %.9f s\n', params.time_step);
    fprintf('  WLS sample index: %d\n', sampleIndex);
    fprintf('  Initial residual norm: %.6e\n', wlsResult.initial_residual_norm);
    fprintf('  Final residual norm:   %.6e\n', wlsResult.final_residual_norm);
    fprintf('  Chi-square:            %.6e\n', wlsResult.chi_square);
    fprintf('  Converged:             %d\n', wlsResult.converged);
    fprintf('  Time-series converged: %d of %d samples\n', ...
        sum(trajectory.converged), numel(trajectory.converged));

    results(idx).data = data; %#ok<SAGROW>
    results(idx).state_info = stateInfo; %#ok<SAGROW>
    results(idx).measurement = measurement; %#ok<SAGROW>
    results(idx).wls = wlsResult; %#ok<SAGROW>
    results(idx).trajectory = trajectory; %#ok<SAGROW>
    results(idx).params = params; %#ok<SAGROW>
end

assignin('base', 'ct_project_results', results);

% All Helper Functions
function params = build_assignment_parameters()
    params = struct();
    params.n = 400.0;
    params.gm = 0.001;
    params.L1 = 26.526e-6;
    params.L2 = 348.0e-6;
    params.L3 = 348.0e-6;
    params.M23 = 287.0e-6;
    params.r1 = 0.005;
    params.r2 = 0.4469;
    params.r3 = 0.4469;
    params.Rb = 0.1;
    params.gb = 1.0 / params.Rb;
    params.lambda0 = 0.1876;
    params.i0 = 6.09109;
    params.L0 = 2.36;
    params.time_step = NaN;
    params.gs1 = NaN;
    params.gs2 = NaN;
    params.gs3 = NaN;
    params.wls_sample_index = 400;
    params.time_series_stride = 1;
    params.measurement_sigma = [ ...
        0.005; ...
        0.005; 0.005; 0.005; ...
        0.0005; 0.0005; 0.0005; 0.005; 0.005; 0.0005; ...
        0.00005; 0.00005; 0.00005; 0.00005; ...
        0.0003; ...
        0.05; 0.05; 0.05; 0.05; 0.05; ...
        1.0];
    params.wls_options = struct('max_iterations', 20, 'step_tolerance', 1e-8, ...
        'residual_tolerance', 1e-8, 'damping', 1e-8);
end

function eventFiles = discover_comtrade_pairs(dataDir)
    files = dir(dataDir);
    files = files(~[files.isdir]);

    pairs = struct();
    keys = {};

    for idx = 1:numel(files)
        [~, baseName, ext] = fileparts(files(idx).name);
        extLower = lower(ext);
        if ~ismember(extLower, {'.cfg', '.dat'})
            continue;
        end

        key = matlab.lang.makeValidName(baseName);
        if ~isfield(pairs, key)
            pairs.(key) = struct('baseName', baseName, 'cfg', '', 'dat', '');
            keys{end + 1} = key; %#ok<AGROW>
        end

        fullPath = fullfile(dataDir, files(idx).name);
        pairs.(key).(extLower(2:end)) = fullPath;
    end

    eventFiles = struct('baseName', {}, 'cfg', {}, 'dat', {});
    for idx = 1:numel(keys)
        pair = pairs.(keys{idx});
        if isempty(pair.cfg) || isempty(pair.dat)
            error('main:MissingPair', ...
                'Missing COMTRADE pair for base name "%s" in %s.', pair.baseName, dataDir);
        end
        eventFiles(end + 1) = pair; %#ok<AGROW>
    end
end

function timeStep = estimate_time_step(timeVector, sampleRateHz)
    if numel(timeVector) > 1
        diffs = diff(timeVector);
        timeStep = median(diffs);
    else
        timeStep = 1.0 / sampleRateHz;
    end

    if ~isfinite(timeStep) || timeStep <= 0
        timeStep = 1.0 / sampleRateHz;
    end
end

function measurement = build_single_step_measurement(data, sampleIndex, params)
    measurement = struct();
    measurement.sample_index = sampleIndex;
    measurement.time = data.time(sampleIndex);
    measurement.z = [data.v_out(sampleIndex); zeros(20, 1)];
    measurement.labels = { ...
        'v_out', ...
        'kcl_node0', 'kcl_node1', 'kcl_node2', ...
        'kvl_loop_transformer', 'kvl_loop_upper', 'kvl_loop_lower', ...
        'kcl_node3', 'kcl_node4', 'flux_derivative', ...
        'y1_relation', 'y2_relation', 'y3_relation', 'y4_relation', ...
        'magnetization', ...
        'ibm_burden', 'ibm_l1', 'ibm_l2', 'ibm_l3', 'ibm_transformer', ...
        'v4_reference'};
    measurement.sigma = params.measurement_sigma(:);
end

function x0 = build_initial_guess(data, sampleIndex, params)
    vOut = data.v_out(sampleIndex);
    gbv = params.gb * vOut;

    x0 = zeros(16, 1);
    x0(1) = vOut;
    x0(2) = 0.0;
    x0(3) = vOut;
    x0(4) = 0.0;
    x0(16) = -gbv;
    x0(6) = x0(16);
    x0(7) = x0(16);
    x0(8) = -x0(16);
    x0(9) = 0.0;
    x0(11) = 0.0;
    x0(5) = params.n * (params.gm * x0(11) + x0(9) - x0(16));
    x0(10) = params.L0 * x0(9);
    x0(12) = (x0(10) / params.lambda0) ^ 2;
    x0(13) = x0(12) ^ 2;
    x0(14) = x0(13) ^ 2;
    x0(15) = x0(14) * x0(12);
end

function trajectory = run_time_series_estimation(data, params, stateInfo)
    stride = max(1, round(params.time_series_stride));
    sampleIndices = 1:stride:numel(data.v_out);
    sampleCount = numel(sampleIndices);

    trajectory = struct();
    trajectory.sample_index = sampleIndices(:);
    trajectory.time = data.time(sampleIndices);
    trajectory.state = zeros(sampleCount, stateInfo.num_states);
    trajectory.initial_guess = zeros(sampleCount, stateInfo.num_states);
    trajectory.predicted_measurement = zeros(sampleCount, 21);
    trajectory.residual = zeros(sampleCount, 21);
    trajectory.initial_residual_norm = zeros(sampleCount, 1);
    trajectory.final_residual_norm = zeros(sampleCount, 1);
    trajectory.chi_square = zeros(sampleCount, 1);
    trajectory.iterations = zeros(sampleCount, 1);
    trajectory.converged = false(sampleCount, 1);
    trajectory.status = cell(sampleCount, 1);

    previousState = build_initial_guess(data, sampleIndices(1), params);

    for sampleIdx = 1:sampleCount
        k = sampleIndices(sampleIdx);
        measurement = build_single_step_measurement(data, k, params);
        weights = diag(1 ./ (measurement.sigma .^ 2));

        if sampleIdx == 1
            xPrev = previousState;
            x0 = previousState;
        else
            xPrev = previousState;
            x0 = previousState;
            x0(stateInfo.index.v1) = data.v_out(k);
            x0(stateInfo.index.v3) = data.v_out(k);
            x0(stateInfo.index.v2) = 0.0;
            x0(stateInfo.index.v4) = 0.0;
        end

        wlsResult = solve_wls_step(measurement.z, x0, xPrev, params, weights, params.wls_options);
        previousState = wlsResult.x;

        trajectory.initial_guess(sampleIdx, :) = x0(:).';
        trajectory.state(sampleIdx, :) = wlsResult.x(:).';
        trajectory.predicted_measurement(sampleIdx, :) = wlsResult.predicted_measurement(:).';
        trajectory.residual(sampleIdx, :) = wlsResult.residual(:).';
        trajectory.initial_residual_norm(sampleIdx) = wlsResult.initial_residual_norm;
        trajectory.final_residual_norm(sampleIdx) = wlsResult.final_residual_norm;
        trajectory.chi_square(sampleIdx) = wlsResult.chi_square;
        trajectory.iterations(sampleIdx) = wlsResult.iterations;
        trajectory.converged(sampleIdx) = wlsResult.converged;
        trajectory.status{sampleIdx} = wlsResult.status;
    end
end
