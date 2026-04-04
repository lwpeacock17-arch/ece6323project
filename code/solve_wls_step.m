function result = solve_wls_step(z, x0, params, weights, options)
    if nargin < 5
        options = struct();
    end
    options = apply_solver_defaults(options);

    if size(weights, 1) ~= numel(z) || size(weights, 2) ~= numel(z)
        error('solve_wls_step:InvalidWeights', ...
            'Weights must be a square matrix matching the measurement length.');
    end

    x = x0;
    residual = residual_function(z, x, params);
    initialResidualNorm = norm(residual);

    result = struct();
    result.iterations = 0;
    result.converged = false;
    result.status = 'max_iterations_reached';
    result.history = struct('iteration', {}, 'residual_norm', {}, 'step_norm', {}, ...
        'chi_square', {});

    for iter = 1:options.max_iterations
        residual = residual_function(z, x, params);
        J = jacobian_function(x, params);

        normalMatrix = J.' * weights * J;
        gradient = J.' * weights * residual;
        regularizedMatrix = normalMatrix + options.damping * eye(size(normalMatrix));

        if rcond(regularizedMatrix) < 1e-12
            result.status = 'ill_conditioned_normal_matrix';
            break;
        end

        dx = regularizedMatrix \ gradient;
        xCandidate = x + dx;
        residualCandidate = residual_function(z, xCandidate, params);

        result.history(end + 1) = struct( ... %#ok<AGROW>
            'iteration', iter, ...
            'residual_norm', norm(residualCandidate), ...
            'step_norm', norm(dx), ...
            'chi_square', residualCandidate.' * weights * residualCandidate);

        x = xCandidate;
        result.iterations = iter;

        if norm(dx) <= options.step_tolerance
            result.converged = true;
            result.status = 'step_tolerance_reached';
            break;
        end

        if norm(residualCandidate) <= options.residual_tolerance
            result.converged = true;
            result.status = 'residual_tolerance_reached';
            break;
        end
    end

    finalResidual = residual_function(z, x, params);
    result.x = x;
    result.residual = finalResidual;
    result.predicted_measurement = measurement_function(x, params);
    result.initial_residual_norm = initialResidualNorm;
    result.final_residual_norm = norm(finalResidual);
    result.chi_square = finalResidual.' * weights * finalResidual;
    result.weights = weights;
end

function options = apply_solver_defaults(options)
    if ~isfield(options, 'max_iterations')
        options.max_iterations = 15;
    end
    if ~isfield(options, 'step_tolerance')
        options.step_tolerance = 1e-9;
    end
    if ~isfield(options, 'residual_tolerance')
        options.residual_tolerance = 1e-9;
    end
    if ~isfield(options, 'damping')
        options.damping = 1e-9;
    end
end
