%DEMO_ROUNDTRIP  Synthetic recovery demo for the MATLAB PV tilt estimator.
%
%   Generates clearsky AC power for a known (tilt, azimuth) installation, then
%   runs the NNLS estimator and verifies the orientation and capacity are
%   recovered. This mirrors the Python test in ../python/tests/test_roundtrip.py.
%
%   Requirements: the CVX toolbox must be installed and on the path
%   (https://cvxr.com/cvx/). The +pvorient package only needs base MATLAB.
%
%   Run from the matlab/ folder:  >> demo_roundtrip

% Make the +pvorient package on this folder visible regardless of cwd.
addpath(fileparts(mfilename('fullpath')));

% --- Site + ground truth -------------------------------------------------
lat = 46.52; lon = 6.63; elev = 500;          % Lausanne, CH
trueTilt = 30; trueAz = 0; trueCapacity = 100; % deg, deg (EU), kWp

% ~7 weeks of hourly UTC timestamps so daytime samples comfortably exceed the
% 320-column grid (-> the orientation is identifiable).
times = (datetime(2023,6,1,0,0,0,'TimeZone','UTC') : hours(1) : ...
         datetime(2023,7,20,0,0,0,'TimeZone','UTC')).';

% --- Build reference matrix + synthesise "measured" power -----------------
[Ppu, ghi] = pvorient.buildReferenceMatrix(lat, lon, elev, times);

g   = pvorient.orientationGrid();
idx = find(g.LAYOUTS(:,1) == trueTilt & g.LAYOUTS(:,2) == trueAz, 1);

Pmeasured = Ppu(:, idx) * trueCapacity;
daytime   = ghi > g.DAYTIME_GHI_THRESHOLD;

fprintf('Daytime samples: %d  (grid columns: %d)\n', sum(daytime), g.N_LAYOUTS);

% --- Estimate ------------------------------------------------------------
result = pvorient.runEstimation(Ppu, Pmeasured, daytime);

fprintf('status        : %s\n',          result.status);
fprintf('best_tilt     : %d  (true %d)\n', result.best_tilt,  trueTilt);
fprintf('best_az_eu    : %d  (true %d)\n', result.best_az_eu, trueAz);
fprintf('effective_kWp : %.2f (true %.2f)\n', result.effective_kWp, trueCapacity);
fprintf('r2            : %.5f\n',         result.r2);
fprintf('rmse_kw       : %.4f\n',         result.rmse_kw);

assert(result.best_tilt  == trueTilt, 'tilt not recovered');
assert(result.best_az_eu == trueAz,   'azimuth not recovered');
assert(abs(result.effective_kWp - trueCapacity) < 1e-2*trueCapacity, 'capacity off');
assert(result.r2 > 0.999, 'poor fit');

disp('--- Top orientations ---');
disp(pvorient.formatResultsTable(result.alpha));

disp('Round-trip OK.');
