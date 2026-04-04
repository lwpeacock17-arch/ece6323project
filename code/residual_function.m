function residual = residual_function(z, xk, xkm1, params)
    if nargin ~= 4
        error('residual_function:InvalidInputCount', ...
            'residual_function expects measurement vector, current state, previous state, and params.');
    end

    h = measurement_function(xk, xkm1, params);
    if numel(z) ~= numel(h)
        error('residual_function:DimensionMismatch', ...
            'Measurement vector length %d does not match model output length %d.', ...
            numel(z), numel(h));
    end

    residual = z - h;
end
