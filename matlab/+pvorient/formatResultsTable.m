function T = formatResultsTable(alpha)
%FORMATRESULTSTABLE  Orientations holding > 1% of the inferred capacity.
%   T = pvorient.formatResultsTable(alpha) returns a table sorted by attributed
%   capacity (descending), with columns:
%     tilt_deg, azimuth_eu, label, capacity_kwp, share_pct

    g = pvorient.orientationGrid();
    alpha = alpha(:);
    total = sum(alpha);
    if total > 0
        thr = 0.01 * total;
    else
        thr = 0;
    end

    idx          = find(alpha > thr);
    tilt_deg     = g.LAYOUTS(idx, 1);
    azimuth_eu   = g.LAYOUTS(idx, 2);
    label        = g.LAYOUT_LABELS(idx);
    capacity_kwp = round(alpha(idx), 2);
    if total > 0
        share_pct = round(100 * alpha(idx) / total, 1);
    else
        share_pct = zeros(numel(idx), 1);
    end

    T = table(tilt_deg, azimuth_eu, label, capacity_kwp, share_pct);
    T = sortrows(T, 'capacity_kwp', 'descend');
end
