function g = orientationGrid()
%ORIENTATIONGRID  Tilt/azimuth search grid and shared constants.
%   g = pvorient.orientationGrid() returns a struct with fields:
%     TILTS_DEG          1x16 tilt values (0:5:75)
%     AZIMUTHS_EU_DEG    1x20 azimuth values (-50:5:45), EU convention
%                        (degrees relative to south, +west / -east)
%     LAYOUTS            Nx2 [tilt, azimuth_eu] for each candidate layout
%     LAYOUT_LABELS      Nx1 cellstr "tilt,azimuth"
%     N_LAYOUTS          number of layouts (320)
%     AZIMUTHS_PVLIB     Nx1 north-clockwise azimuth (180 + azimuth_eu)
%     TILTS_FLOAT        Nx1 tilt per layout
%     DAYTIME_GHI_THRESHOLD  clearsky GHI cut-off for "daytime" (W/m^2)
%
%   Ordering matches the Python implementation: azimuth is the outer loop,
%   tilt the inner loop (so tilt varies fastest).

    g.TILTS_DEG       = 0:5:75;       % 16 values
    g.AZIMUTHS_EU_DEG = -50:5:45;     % 20 values

    nT = numel(g.TILTS_DEG);
    nA = numel(g.AZIMUTHS_EU_DEG);
    N  = nT * nA;

    layouts = zeros(N, 2);
    labels  = cell(N, 1);
    k = 0;
    for az = g.AZIMUTHS_EU_DEG          % outer loop
        for tilt = g.TILTS_DEG          % inner loop
            k = k + 1;
            layouts(k, :) = [tilt, az];
            labels{k}     = sprintf('%d,%d', tilt, az);
        end
    end

    g.LAYOUTS        = layouts;
    g.LAYOUT_LABELS  = labels;
    g.N_LAYOUTS      = N;               % 320
    g.AZIMUTHS_PVLIB = 180 + layouts(:, 2);
    g.TILTS_FLOAT    = layouts(:, 1);
    g.DAYTIME_GHI_THRESHOLD = 50.0;
end
