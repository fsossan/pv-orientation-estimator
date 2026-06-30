function [grid, tilts, azimuthsEu] = alphaToHeatmapGrid(alpha)
%ALPHATOHEATMAPGRID  Reshape alpha (N x 1) into a (tilt x azimuth) grid.
%   [grid, tilts, azs] = pvorient.alphaToHeatmapGrid(alpha)
%   grid is (numel(tilts) x numel(azs)); grid(i, j) is the capacity attributed
%   to tilts(i), azs(j).

    g  = pvorient.orientationGrid();
    nT = numel(g.TILTS_DEG);
    nA = numel(g.AZIMUTHS_EU_DEG);

    % LAYOUTS ordering: azimuth outer, tilt inner -> tilt varies fastest, which
    % is exactly MATLAB's column-major fill order, so a plain reshape gives a
    % (tilt x azimuth) grid directly.
    grid       = reshape(alpha(:), nT, nA);
    tilts      = g.TILTS_DEG;
    azimuthsEu = g.AZIMUTHS_EU_DEG;
end
