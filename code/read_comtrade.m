function data = read_comtrade(cfgPath, datPath)
    if nargin ~= 2
        error('read_comtrade:InvalidInputCount', ...
            'read_comtrade expects exactly two inputs: cfgPath and datPath.');
    end

    if exist(cfgPath, 'file') ~= 2
        error('read_comtrade:MissingCfg', 'CFG file not found: %s', cfgPath);
    end
    if exist(datPath, 'file') ~= 2
        error('read_comtrade:MissingDat', 'DAT file not found: %s', datPath);
    end

    cfgLines = read_text_lines(cfgPath);
    if numel(cfgLines) < 8
        error('read_comtrade:MalformedCfg', ...
            'CFG file must contain at least 8 lines: %s', cfgPath);
    end

    data = struct();
    data.event_name = strtrim(cfgLines{1});

    [channelCounts, analogCount, digitalCount] = parse_channel_counts(cfgLines{2}, cfgPath);
    data.channel_info = parse_analog_channels(cfgLines(3:(2 + analogCount)), cfgPath);
    data.channel_counts = struct('total', channelCounts, 'analog', analogCount, ...
        'digital', digitalCount);

    cursor = 3 + analogCount + digitalCount;
    if cursor + 4 > numel(cfgLines)
        error('read_comtrade:MalformedCfg', ...
            'CFG metadata section is incomplete: %s', cfgPath);
    end

    data.system_frequency_hz = str2double(strtrim(cfgLines{cursor}));
    sampleRateCount = str2double(strtrim(cfgLines{cursor + 1}));
    cursor = cursor + 2;

    data.sample_rates = zeros(sampleRateCount, 2);
    for idx = 1:sampleRateCount
        tokens = split_csv_line(cfgLines{cursor + idx - 1});
        if numel(tokens) < 2
            error('read_comtrade:MalformedCfg', ...
                'Invalid sample rate definition in %s.', cfgPath);
        end
        data.sample_rates(idx, :) = [str2double(tokens{1}), str2double(tokens{2})];
    end
    cursor = cursor + sampleRateCount;

    data.start_time = parse_comtrade_datetime(cfgLines{cursor});
    data.trigger_time = parse_comtrade_datetime(cfgLines{cursor + 1});
    data.format = upper(strtrim(cfgLines{cursor + 2}));

    if ~strcmp(data.format, 'ASCII')
        error('read_comtrade:UnsupportedFormat', ...
            'Only ASCII COMTRADE DAT files are supported. Found %s.', data.format);
    end

    data.time_multiplier = str2double(strtrim(cfgLines{cursor + 3}));
    data.sample_rate_hz = data.sample_rates(end, 1);
    data.source = struct('cfg', cfgPath, 'dat', datPath);

    rawMatrix = read_ascii_dat(datPath, analogCount);
    data.sample_index = rawMatrix(:, 1);
    rawTime = rawMatrix(:, 2);
    data.time = rawTime .* 1e-6 .* data.time_multiplier;
    data.analog_raw = rawMatrix(:, 3:(2 + analogCount));

    data.analog_scaled = zeros(size(data.analog_raw));
    for idx = 1:analogCount
        a = data.channel_info(idx).scale_a;
        b = data.channel_info(idx).scale_b;
        data.analog_scaled(:, idx) = a .* data.analog_raw(:, idx) + b;
    end

    data.v_out = data.analog_scaled(:, 1);
    data.metadata = struct();
    data.metadata.cfg_line_count = numel(cfgLines);
    data.metadata.dat_sample_count = size(rawMatrix, 1);
    data.metadata.quantization_step = abs(data.channel_info(1).scale_a);
    data.metadata.quantization_sigma = data.metadata.quantization_step / sqrt(12);
    data.metadata.raw_min = data.channel_info(1).min_raw;
    data.metadata.raw_max = data.channel_info(1).max_raw;
end

function lines = read_text_lines(filePath)
    rawText = fileread(filePath);
    lines = regexp(rawText, '\r\n|\n|\r', 'split');
    if isempty(lines{end})
        lines(end) = [];
    end
end

function [totalChannels, analogCount, digitalCount] = parse_channel_counts(line, cfgPath)
    tokens = split_csv_line(line);
    if numel(tokens) < 3
        error('read_comtrade:MalformedCfg', ...
            'Invalid channel count line in %s.', cfgPath);
    end

    totalChannels = str2double(tokens{1});
    analogCount = parse_count_token(tokens{2}, 'A', cfgPath);
    digitalCount = parse_count_token(tokens{3}, 'D', cfgPath);
end

function count = parse_count_token(token, suffix, cfgPath)
    token = upper(strtrim(token));
    if isempty(token) || token(end) ~= suffix
        error('read_comtrade:MalformedCfg', ...
            'Expected count token ending with %s in %s.', suffix, cfgPath);
    end

    count = str2double(token(1:end-1));
end

function channels = parse_analog_channels(lines, cfgPath)
    channels = struct('index', {}, 'name', {}, 'phase', {}, 'circuit_component', {}, ...
        'units', {}, 'scale_a', {}, 'scale_b', {}, 'skew', {}, 'min_raw', {}, ...
        'max_raw', {}, 'primary', {}, 'secondary', {}, 'ps', {});

    for idx = 1:numel(lines)
        tokens = split_csv_line(lines{idx});
        if numel(tokens) < 13
            error('read_comtrade:MalformedCfg', ...
                'Analog channel line %d is malformed in %s.', idx, cfgPath);
        end

        channels(idx).index = str2double(tokens{1}); %#ok<AGROW>
        channels(idx).name = tokens{2}; %#ok<AGROW>
        channels(idx).phase = tokens{3}; %#ok<AGROW>
        channels(idx).circuit_component = tokens{4}; %#ok<AGROW>
        channels(idx).units = tokens{5}; %#ok<AGROW>
        channels(idx).scale_a = str2double(tokens{6}); %#ok<AGROW>
        channels(idx).scale_b = str2double(tokens{7}); %#ok<AGROW>
        channels(idx).skew = str2double(tokens{8}); %#ok<AGROW>
        channels(idx).min_raw = str2double(tokens{9}); %#ok<AGROW>
        channels(idx).max_raw = str2double(tokens{10}); %#ok<AGROW>
        channels(idx).primary = str2double(tokens{11}); %#ok<AGROW>
        channels(idx).secondary = str2double(tokens{12}); %#ok<AGROW>
        channels(idx).ps = tokens{13}; %#ok<AGROW>
    end
end

function dt = parse_comtrade_datetime(line)
    line = strtrim(line);
    formats = {'MM/dd/yyyy,HH:mm:ss.SSSSSS', 'MM/dd/yyyy,HH:mm:ss'};

    dt = NaT;
    for idx = 1:numel(formats)
        try
            dt = datetime(line, 'InputFormat', formats{idx});
            return;
        catch
        end
    end

    error('read_comtrade:MalformedCfg', ...
        'Unable to parse COMTRADE datetime string: %s', line);
end

function rawMatrix = read_ascii_dat(datPath, analogCount)
    rawText = fileread(datPath);
    lines = regexp(strtrim(rawText), '\r\n|\n|\r', 'split');
    expectedColumns = 2 + analogCount;
    rawMatrix = zeros(numel(lines), expectedColumns);

    for idx = 1:numel(lines)
        tokens = split_csv_line(lines{idx});
        if numel(tokens) < expectedColumns
            error('read_comtrade:MalformedDat', ...
                'DAT line %d in %s has %d columns, expected at least %d.', ...
                idx, datPath, numel(tokens), expectedColumns);
        end

        for col = 1:expectedColumns
            rawMatrix(idx, col) = str2double(tokens{col});
        end
    end
end

function tokens = split_csv_line(line)
    parts = regexp(line, ',', 'split');
    tokens = cellfun(@strtrim, parts, 'UniformOutput', false);
end
