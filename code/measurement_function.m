function h = measurement_function(xk, xkm1, params)
    if nargin ~= 3
        error('measurement_function:InvalidInputCount', ...
            'measurement_function expects current state, previous state, and params.');
    end

    validate_state_vector(xk);
    validate_state_vector(xkm1);

    current = unpack_state(xk);
    previous = unpack_state(xkm1);
    deriv = compute_derivatives(current, previous, params.time_step);

    coupling23 = params.L2 * deriv.diL2_dt - params.M23 * deriv.diL3_dt;
    coupling32 = params.L3 * deriv.diL3_dt - params.M23 * deriv.diL2_dt;

    h = zeros(14, 1);
    h(1) = current.v3 - current.v4;
    h(2) = -params.gm * current.e - current.im + (current.ip / params.n) + ...
        current.iL1 + params.gs1 * params.L1 * deriv.diL1_dt;
    h(3) = params.gm * current.e + current.im - (current.ip / params.n) - ...
        current.iL2 - params.gs2 * coupling23;
    h(4) = -params.gm * current.e - current.im + (current.ip / params.n) + ...
        current.iL3 + params.gs3 * coupling32;
    h(5) = -current.v1 + current.v2 + current.e + params.L1 * deriv.diL1_dt + ...
        params.r1 * (params.gm * current.e + current.im - (current.ip / params.n));
    h(6) = -current.v3 + current.v1 + ...
        params.r2 * (current.iL2 + params.gs2 * coupling23) + coupling23;
    h(7) = -current.v2 + current.v4 + ...
        params.r3 * (current.iL3 + params.gs3 * coupling32) + coupling32;
    h(8) = current.iL2 + params.gs2 * coupling23 + params.gb * (current.v3 - current.v4);
    h(9) = -current.iL3 - params.gs3 * coupling32 + params.gb * (current.v4 - current.v3);
    h(10) = current.e - deriv.dlambda_dt;
    h(11) = current.v4;
    h(12) = current.im;
    h(13) = current.lambda;
    h(14) = current.e;
end

function validate_state_vector(x)
    if numel(x) ~= 11
        error('measurement_function:InvalidStateVector', ...
            'Expected an 11-element state vector, got %d elements.', numel(x));
    end
end

function state = unpack_state(x)
    state = struct();
    state.v1 = x(1);
    state.v2 = x(2);
    state.v3 = x(3);
    state.v4 = x(4);
    state.ip = x(5);
    state.iL1 = x(6);
    state.iL2 = x(7);
    state.iL3 = x(8);
    state.im = x(9);
    state.lambda = x(10);
    state.e = x(11);
end

function deriv = compute_derivatives(current, previous, dt)
    deriv = struct();
    deriv.diL1_dt = (current.iL1 - previous.iL1) / dt;
    deriv.diL2_dt = (current.iL2 - previous.iL2) / dt;
    deriv.diL3_dt = (current.iL3 - previous.iL3) / dt;
    deriv.dlambda_dt = (current.lambda - previous.lambda) / dt;
end
