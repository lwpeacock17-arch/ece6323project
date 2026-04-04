function stateInfo = state_definition()
    stateInfo = struct();
    stateInfo.full_names = {'v1', 'v2', 'v3', 'v4', 'ip', 'im', 'iL1', 'iL2', 'iL3', 'lambda'};
    stateInfo.active_names = {'v3', 'v4', 'ip', 'im', 'lambda'};
    stateInfo.active_indices = [3, 4, 5, 6, 10];
    stateInfo.num_full_states = numel(stateInfo.full_names);
    stateInfo.num_active_states = numel(stateInfo.active_names);
    stateInfo.description = ...
        'Reduced first-milestone estimator state. Expand this mapping when the full CT model is added.';
end
