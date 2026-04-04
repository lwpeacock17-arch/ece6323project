function residual = residual_function(z, x, params)
    if nargin ~= 3
        error('residual_function:InvalidInputCount', ...
            'residual_function expects exactly three inputs.');
    end

    h = measurement_function(x, params);
    if numel(z) ~= numel(h)
        error('residual_function:DimensionMismatch', ...
            'Measurement vector length %d does not match model output length %d.', ...
            numel(z), numel(h));
    end

    residual = z - h;
end
