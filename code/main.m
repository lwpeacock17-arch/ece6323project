clear;
clc;

projectRoot = fileparts(mfilename('fullpath'));
dataDir = fullfile(projectRoot, 'data');

params = struct();
params.burden_resistance = 1.0;
params.turns_ratio = 200.0;
params.magnetizing_inductance = 0.05;
params.pseudo_measurement_sigma = [1e-3; 1e-3; 5e-3; 5e-3];
params.wls_sample_index = 400;
params.time_series_stride = 1;
params.wls_options = struct('max_iterations', 15, 'step_tolerance', 1e-9, ...
    'residual_tolerance', 1e-9, 'damping', 1e-9);

eventFiles = discover_comtrade_pairs(dataDir);
fprintf('Discovered %d COMTRADE event(s) in %s\n', numel(eventFiles), dataDir);

results = struct([]);

for idx = 1:numel(eventFiles)
    cfgPath = eventFiles(idx).cfg;
    datPath = eventFiles(idx).dat;
    data = read_comtrade(cfgPath, datPath);

    data.baseline_current = (data.v_out ./ params.burden_resistance) .* params.turns_ratio;

    stateInfo = state_definition();
    sampleIndex = min(max(params.wls_sample_index, 1), numel(data.v_out));
    measurement = build_single_step_measurement(data, sampleIndex, params);
    x0 = build_initial_guess(data, sampleIndex, params);

    weights = diag(1 ./ (measurement.sigma .^ 2));
    wlsResult = solve_wls_step(measurement.z, x0, params, weights, params.wls_options);
    trajectory = run_time_series_estimation(data, params);

    data.estimated_current = trajectory.state(:, 3);
    data.estimated_magnetizing_current = trajectory.state(:, 4);
    data.estimated_flux = trajectory.state(:, 5);

    plotSpec = struct();
    plotSpec.showBaseline = true;
    plotSpec.showEstimated = true;
    plotSpec.showDiagnostics = true;
    plotSpec.figureName = sprintf('%s Overview', data.event_name);
    plotSpec.currentLabel = sprintf('Current Estimates, R_b = %.3f ohm, n = %.1f', ...
        params.burden_resistance, params.turns_ratio);
    plotSpec.diagnosticTime = trajectory.time;
    plotSpec.residualNorm = trajectory.final_residual_norm;
    plotSpec.chiSquare = trajectory.chi_square;
    plot_comtrade(data, plotSpec);

    fprintf('\nEvent: %s\n', data.event_name);
    fprintf('  Samples: %d\n', numel(data.time));
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
end

assignin('base', 'ct_project_results', results);

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

        if ~isfield(pairs, matlab.lang.makeValidName(baseName))
            key = matlab.lang.makeValidName(baseName);
            pairs.(key) = struct('baseName', baseName, 'cfg', '', 'dat', '');
            keys{end + 1} = key; %#ok<AGROW>
        else
            key = matlab.lang.makeValidName(baseName);
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

function measurement = build_single_step_measurement(data, sampleIndex, params)
    measurement = struct();
    measurement.sample_index = sampleIndex;
    measurement.time = data.time(sampleIndex);
    measurement.z = [data.v_out(sampleIndex); 0; 0; 0; 0];
    measurement.labels = {'v_out', 'burden_consistency', 'flux_consistency', ...
        'current_split', 'common_mode_reference'};
    measurement.sigma = [5e-3; params.pseudo_measurement_sigma(:)];
end

function x0 = build_initial_guess(data, sampleIndex, params)
    vOut = data.v_out(sampleIndex);
    iPrimary = (vOut / params.burden_resistance) * params.turns_ratio;
    iMag = 0.05 * iPrimary;
    lambda = params.magnetizing_inductance * iMag;

    x0 = [vOut / 2; -vOut / 2; iPrimary; iMag; lambda];
end

function trajectory = run_time_series_estimation(data, params)
    stride = max(1, round(params.time_series_stride));
    sampleIndices = 1:stride:numel(data.v_out);
    sampleCount = numel(sampleIndices);

    trajectory = struct();
    trajectory.sample_index = sampleIndices(:);
    trajectory.time = data.time(sampleIndices);
    trajectory.state = zeros(sampleCount, 5);
    trajectory.initial_guess = zeros(sampleCount, 5);
    trajectory.predicted_measurement = zeros(sampleCount, 5);
    trajectory.residual = zeros(sampleCount, 5);
    trajectory.initial_residual_norm = zeros(sampleCount, 1);
    trajectory.final_residual_norm = zeros(sampleCount, 1);
    trajectory.chi_square = zeros(sampleCount, 1);
    trajectory.iterations = zeros(sampleCount, 1);
    trajectory.converged = false(sampleCount, 1);
    trajectory.status = cell(sampleCount, 1);

    previousState = [];

    for sampleIdx = 1:sampleCount
        k = sampleIndices(sampleIdx);
        measurement = build_single_step_measurement(data, k, params);
        weights = diag(1 ./ (measurement.sigma .^ 2));

        if isempty(previousState)
            x0 = build_initial_guess(data, k, params);
        else
            x0 = previousState;
            x0(1) = data.v_out(k) / 2;
            x0(2) = -data.v_out(k) / 2;
        end

        wlsResult = solve_wls_step(measurement.z, x0, params, weights, params.wls_options);
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
