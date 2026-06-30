function [Ppu, ghi] = buildReferenceMatrix(lat, lon, elev, times, varargin)
%BUILDREFERENCEMATRIX  Per-unit clearsky POA reference matrix.
%   [Ppu, ghi] = pvorient.buildReferenceMatrix(lat, lon, elev, times)
%   [...]      = pvorient.buildReferenceMatrix(..., 'LinkeTurbidity', TL, ...
%                                                  'Albedo', a)
%   [...]      = pvorient.buildReferenceMatrix(..., 'AirTemp', Tair, ...
%                                                  'Gamma', g, ...
%                                                  'TempRiseCoeff', k, ...
%                                                  'TempRef', Tref)
%
%   lat, lon : site latitude / longitude [deg]
%   elev     : site altitude [m]
%   times    : datetime vector (UTC)
%
%   Ppu : (T x 320) per-unit POA irradiance (clearsky POA / 1000 W/m^2), one
%         column per orientation in pvorient.orientationGrid().LAYOUTS
%   ghi : (T x 1) clearsky GHI [W/m^2], for building the daytime mask
%
%   Uses an isotropic sky-diffuse transposition, matching the default model of
%   pvlib.irradiance.get_total_irradiance:
%     POA = DNI*cos(AOI) + DHI*(1+cos t)/2 + GHI*albedo*(1-cos t)/2
%
%   Temperature correction (optional). If 'AirTemp' is provided (a scalar degC
%   or a T x 1 vector), each column is additionally scaled by the empirical
%   temperature factor of Sossan et al. (Eqs. 6-7):
%     Tcell  = AirTemp + TempRiseCoeff .* POA           % POA in W/m^2   (7)
%     factor = 1 + Gamma .* (Tcell - TempRef)                          % (6)
%   Defaults reproduce the paper's polycrystalline / mixed-mounting values:
%     Gamma = -0.0043 [1/degC], TempRiseCoeff = 0.038 [degC*m^2/W],
%     TempRef = 25 [degC]. With no 'AirTemp' the matrix is the plain per-unit
%     POA irradiance (no correction).

    p = inputParser;
    addParameter(p, 'LinkeTurbidity', 3.0);
    addParameter(p, 'Albedo', 0.25);
    addParameter(p, 'AirTemp', []);
    addParameter(p, 'Gamma', -0.0043);
    addParameter(p, 'TempRiseCoeff', 0.038);
    addParameter(p, 'TempRef', 25.0);
    parse(p, varargin{:});
    TL        = p.Results.LinkeTurbidity;
    albedo    = p.Results.Albedo;
    airTemp   = p.Results.AirTemp;
    gamma     = p.Results.Gamma;
    tempRise  = p.Results.TempRiseCoeff;
    tempRef   = p.Results.TempRef;

    g     = pvorient.orientationGrid();
    times = times(:);

    [zen, saz]      = pvorient.solarPosition(lat, lon, times);   % [deg]
    [ghi, dni, dhi] = pvorient.ineichenClearsky(zen, elev, TL);

    z    = deg2rad(zen);
    sazr = deg2rad(saz);
    cosz = cos(z);
    sinz = sin(z);

    T   = numel(times);
    N   = g.N_LAYOUTS;
    Ppu = zeros(T, N);

    applyTemp = ~isempty(airTemp);
    if applyTemp
        airTemp = airTemp(:);
        if isscalar(airTemp)
            airTemp = repmat(airTemp, T, 1);
        elseif numel(airTemp) ~= T
            error('buildReferenceMatrix:AirTempSize', ...
                  'AirTemp must be a scalar or have length %d, got %d.', ...
                  T, numel(airTemp));
        end
    end

    for k = 1:N
        tilt   = deg2rad(g.TILTS_FLOAT(k));
        surfAz = deg2rad(g.AZIMUTHS_PVLIB(k));

        cosAOI = cosz.*cos(tilt) + sinz.*sin(tilt).*cos(sazr - surfAz);
        cosAOI = max(cosAOI, 0);

        poaBeam    = dni .* cosAOI;
        poaSkyDiff = dhi .* (1 + cos(tilt)) / 2;             % isotropic sky
        poaGndDiff = ghi .* albedo .* (1 - cos(tilt)) / 2;   % ground reflected

        poa = max(poaBeam + poaSkyDiff + poaGndDiff, 0);     % [W/m^2]
        col = poa / 1000;
        if applyTemp
            tCell = airTemp + tempRise .* poa;               % Eq. (7)
            col   = col .* (1 + gamma .* (tCell - tempRef));  % Eq. (6)
        end
        Ppu(:, k) = col;
    end
end
