function h = measurement_function(x, params)
    if nargin ~= 2
        error('measurement_function:InvalidInputCount', ...
            'measurement_function expects exactly two inputs.');
    end
    validate_state_vector(x);

    v3 = x(1);
    v4 = x(2);
    ip = x(3);
    im = x(4);
    lambda = x(5);

    burdenVoltage = v3 - v4;
    burdenConsistency = ip - ((params.turns_ratio / params.burden_resistance) * burdenVoltage);
    fluxConsistency = lambda - params.magnetizing_inductance * im;
    currentSplit = ip - im - (burdenVoltage / params.burden_resistance);
    commonModeReference = v3 + v4;

    h = [burdenVoltage; burdenConsistency; fluxConsistency; currentSplit; commonModeReference];
end

function validate_state_vector(x)
    if numel(x) ~= 5
        error('measurement_function:InvalidStateVector', ...
            'Expected a 5-element active state vector, got %d elements.', numel(x));
    end
end
