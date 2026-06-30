function result = runEstimation(Ppu, Pmeasured, daytimeMask)
%RUNESTIMATION  NNLS orientation fit via CVX.
%   result = pvorient.runEstimation(Ppu, Pmeasured, daytimeMask)
%
%   Solves, on the daytime-filtered samples:
%       minimize  || Pmeasured - Ppu*alpha ||^2     s.t.  alpha >= 0
%   using the CVX toolbox (https://cvxr.com/cvx/), which must be on the path.
%
%   Ppu         : (T x N) per-unit POA reference matrix
%   Pmeasured   : (T x 1) measured AC power [kW]
%   daytimeMask : (T x 1) logical, true for daytime samples to fit
%
%   result is a struct with fields:
%     status         solver status (CVX cvx_status)
%     alpha          (N x 1) kWp attributed to each orientation, or []
%     effective_kWp  sum(alpha)
%     best_idx       index of the dominant orientation (1-based, MATLAB)
%     best_label     'tilt,azimuth' of the dominant orientation
%     best_tilt      tilt of the dominant orientation [deg]
%     best_az_eu     azimuth of the dominant orientation [deg, EU convention]
%     r2             coefficient of determination on daytime points
%     rmse_kw        RMSE on daytime points [kW]
%
%   On solver failure only status and alpha (= []) are populated.

    g = pvorient.orientationGrid();
    N = g.N_LAYOUTS;

    Pd = Ppu(daytimeMask, :);
    yd = Pmeasured(daytimeMask);
    yd = yd(:);

    cvx_begin quiet
        variable a(N) nonnegative
        minimize( sum_square(yd - Pd*a) )
    cvx_end

    if ~any(strcmp(cvx_status, {'Solved', 'Inaccurate/Solved'})) || isempty(a)
        result.status = cvx_status;
        result.alpha  = [];
        return;
    end

    a = max(a, 0);

    Pfit  = Pd * a;
    ssRes = sum((yd - Pfit).^2);
    ssTot = sum((yd - mean(yd)).^2);
    if ssTot > 0
        r2 = 1 - ssRes/ssTot;
    else
        r2 = NaN;
    end
    rmse = sqrt(ssRes / numel(yd));

    [~, bestIdx] = max(a);

    result.status        = cvx_status;
    result.alpha         = a;
    result.effective_kWp = sum(a);
    result.best_idx      = bestIdx;            % 1-based
    result.best_label    = g.LAYOUT_LABELS{bestIdx};
    result.best_tilt     = g.LAYOUTS(bestIdx, 1);
    result.best_az_eu    = g.LAYOUTS(bestIdx, 2);
    result.r2            = r2;
    result.rmse_kw       = rmse;
end
