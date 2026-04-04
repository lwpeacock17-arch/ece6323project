function fig = plot_comtrade(dataStruct, plotSpec)
    if nargin < 2
        plotSpec = struct();
    end

    plotSpec = apply_plot_defaults(plotSpec);

    fig = figure('Name', plotSpec.figureName, 'Color', 'w');
    if plotSpec.showDiagnostics
        subplot(4, 1, 1);
    else
        subplot(2, 1, 1);
    end

    plot(dataStruct.time, dataStruct.v_out, 'LineWidth', 1.2, 'Color', [0 0.35 0.7]);
    grid on;
    xlabel('Time (s)');
    ylabel(sprintf('Voltage (%s)', dataStruct.channel_info(1).units));
    title(sprintf('%s: Burden Voltage', dataStruct.event_name), 'Interpreter', 'none');

    if plotSpec.showDiagnostics
        subplot(4, 1, 2);
    else
        subplot(2, 1, 2);
    end

    if plotSpec.showBaseline && isfield(dataStruct, 'baseline_current')
        plot(dataStruct.time, dataStruct.baseline_current, 'LineWidth', 1.2, ...
            'Color', [0.8 0.25 0.1]);
        hold on;

        if plotSpec.showEstimated && isfield(dataStruct, 'estimated_current')
            plot(dataStruct.time, dataStruct.estimated_current, 'LineWidth', 1.2, ...
                'Color', [0.15 0.55 0.15]);
            legend({'Baseline i_p', 'Estimated i_p'}, 'Location', 'best');
        end
        hold off;
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

    if plotSpec.showDiagnostics
        subplot(4, 1, 3);
        plot(plotSpec.diagnosticTime, plotSpec.residualNorm, 'LineWidth', 1.1, ...
            'Color', [0.55 0.2 0.75]);
        grid on;
        xlabel('Time (s)');
        ylabel('||r||_2');
        title('Final Residual Norm', 'Interpreter', 'none');

        subplot(4, 1, 4);
        plot(plotSpec.diagnosticTime, plotSpec.chiSquare, 'LineWidth', 1.1, ...
            'Color', [0.15 0.15 0.15]);
        grid on;
        xlabel('Time (s)');
        ylabel('\chi^2');
        title('Chi-Square Statistic', 'Interpreter', 'tex');
    end

    if plotSpec.showResidualBreakdown
        plot_residual_breakdown(dataStruct, plotSpec);
    end
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
    if ~isfield(plotSpec, 'showEstimated')
        plotSpec.showEstimated = false;
    end
    if ~isfield(plotSpec, 'showDiagnostics')
        plotSpec.showDiagnostics = false;
    end
    if ~isfield(plotSpec, 'diagnosticTime')
        plotSpec.diagnosticTime = [];
    end
    if ~isfield(plotSpec, 'residualNorm')
        plotSpec.residualNorm = [];
    end
    if ~isfield(plotSpec, 'chiSquare')
        plotSpec.chiSquare = [];
    end
    if ~isfield(plotSpec, 'showResidualBreakdown')
        plotSpec.showResidualBreakdown = false;
    end
    if ~isfield(plotSpec, 'residualMatrix')
        plotSpec.residualMatrix = [];
    end
    if ~isfield(plotSpec, 'residualLabels')
        plotSpec.residualLabels = {};
    end
end

function plot_residual_breakdown(dataStruct, plotSpec)
    figName = sprintf('%s Residual Breakdown', dataStruct.event_name);
    figure('Name', figName, 'Color', 'w');

    groups = { ...
        1, ...
        2:4, ...
        5:7, ...
        8:10, ...
        11:15, ...
        16:21};
    titles = { ...
        'Actual Measurement', ...
        'KCL Constraints', ...
        'KVL Constraints', ...
        'Node and Flux Constraints', ...
        'Nonlinear Magnetization', ...
        'Derived and Pseudo Measurements'};

    colors = lines(6);
    for idx = 1:numel(groups)
        subplot(3, 2, idx);
        cols = groups{idx};
        plot(plotSpec.diagnosticTime, plotSpec.residualMatrix(:, cols), 'LineWidth', 1.0);
        grid on;
        xlabel('Time (s)');
        ylabel('Residual');
        title(titles{idx}, 'Interpreter', 'none');

        labels = plotSpec.residualLabels(cols);
        if isempty(labels)
            labels = build_default_labels(cols);
        end
        legend(labels, 'Location', 'best', 'Interpreter', 'none');

        ax = gca;
        ax.ColorOrder = colors;
    end
end

function labels = build_default_labels(indices)
    labels = cell(size(indices));
    for idx = 1:numel(indices)
        labels{idx} = sprintf('r_%d', indices(idx));
    end
end
