function J = jacobian_function(xk, xkm1, params)
    if nargin ~= 3
        error('jacobian_function:InvalidInputCount', ...
            'jacobian_function expects current state, previous state, and params.');
    end

    if numel(xk) ~= 11 || numel(xkm1) ~= 11
        error('jacobian_function:InvalidStateVector', ...
            'jacobian_function expects 11-element current and previous state vectors.');
    end

    dt = params.time_step;

    d1 = params.gs1 * params.L1 / dt;
    d2 = params.gs2 * params.L2 / dt;
    dm23_2 = params.gs2 * params.M23 / dt;
    d3 = params.gs3 * params.L3 / dt;
    dm23_3 = params.gs3 * params.M23 / dt;

    J = zeros(14, 11);

    J(1, [3 4]) = [1 -1];

    J(2, 5) = 1 / params.n;
    J(2, 6) = 1 + d1;
    J(2, 9) = -1;
    J(2, 11) = -params.gm;

    J(3, 5) = -1 / params.n;
    J(3, 7) = -1 - d2;
    J(3, 8) = dm23_2;
    J(3, 9) = 1;
    J(3, 11) = params.gm;

    J(4, 5) = 1 / params.n;
    J(4, 7) = -dm23_3;
    J(4, 8) = 1 + d3;
    J(4, 9) = -1;
    J(4, 11) = -params.gm;

    J(5, 1) = -1;
    J(5, 2) = 1;
    J(5, 5) = -params.r1 / params.n;
    J(5, 6) = params.L1 / dt;
    J(5, 9) = params.r1;
    J(5, 11) = 1 + params.r1 * params.gm;

    J(6, 1) = 1;
    J(6, 3) = -1;
    J(6, 7) = params.r2 * (1 + d2) + params.L2 / dt;
    J(6, 8) = -params.r2 * dm23_2 - params.M23 / dt;

    J(7, 2) = -1;
    J(7, 4) = 1;
    J(7, 7) = -params.r3 * dm23_3 - params.M23 / dt;
    J(7, 8) = params.r3 * (1 + d3) + params.L3 / dt;

    J(8, 3) = params.gb;
    J(8, 4) = -params.gb;
    J(8, 7) = 1 + d2;
    J(8, 8) = -dm23_2;

    J(9, 3) = -params.gb;
    J(9, 4) = params.gb;
    J(9, 7) = dm23_3;
    J(9, 8) = -1 - d3;

    J(10, 10) = -1 / dt;
    J(10, 11) = 1;

    J(11, 4) = 1;
    J(12, 9) = 1;
    J(13, 10) = 1;
    J(14, 11) = 1;
end
