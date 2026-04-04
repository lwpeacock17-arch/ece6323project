function fig = plot_comtrade(dataStruct, plotSpec)
    if nargin < 2
        plotSpec = struct();
    end

    plotSpec = apply_plot_defaults(plotSpec);

    fig = figure('Name', plotSpec.figureName, 'Color', 'w');
    subplot(2, 1, 1);

    plot(dataStruct.time, dataStruct.v_out, 'LineWidth', 1.2, 'Color', [0 0.35 0.7]);
    grid on;
    xlabel('Time (s)');
    ylabel(sprintf('Voltage (%s)', dataStruct.channel_info(1).units));
    title(sprintf('%s: Burden Voltage', dataStruct.event_name), 'Interpreter', 'none');

    subplot(2, 1, 2);
    if plotSpec.showBaseline && isfield(dataStruct, 'baseline_current')
        plot(dataStruct.time, dataStruct.baseline_current, 'LineWidth', 1.2, ...
            'Color', [0.8 0.25 0.1]);
        ylabel('Current (A)');
        title(plotSpec.currentLabel, 'Interpreter', 'none');
    else
        plot(dataStruct.time, dataStruct.analog_raw(:, 1), 'LineWidth', 1.0, ...
            'Color', [0.2 0.2 0.2]);
        ylabel('Raw Counts');
        title('Raw Analog Samples', 'Interpreter', 'none');
    end
    grid on;
    xlabel('Time (s)');
end

function plotSpec = apply_plot_defaults(plotSpec)
    if ~isfield(plotSpec, 'figureName') || isempty(plotSpec.figureName)
        plotSpec.figureName = 'COMTRADE Event';
    end
    if ~isfield(plotSpec, 'showBaseline')
        plotSpec.showBaseline = false;
    end
    if ~isfield(plotSpec, 'currentLabel') || isempty(plotSpec.currentLabel)
        plotSpec.currentLabel = 'Baseline Current Estimate';
    end
end
