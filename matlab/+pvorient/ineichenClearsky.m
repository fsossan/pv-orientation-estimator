function [ghi, dni, dhi] = ineichenClearsky(apparentZenithDeg, altitude, linkeTurbidity)
%INEICHENCLEARSKY  Ineichen-Perez clearsky irradiance (GHI/DNI/DHI).
%   [ghi, dni, dhi] = pvorient.ineichenClearsky(zenithDeg, altitude, TL)
%
%   Independent, toolbox-free reimplementation of pvlib.clearsky.ineichen with
%   a constant Linke turbidity TL. (pvlib looks TL up from a monthly
%   climatology; here it is a scalar input — see buildReferenceMatrix, default
%   TL = 3.0, a typical mid-latitude value.)
%
%   apparentZenithDeg : apparent solar zenith angle [deg]
%   altitude          : site altitude [m]
%   linkeTurbidity    : Linke turbidity factor (dimensionless)

    z    = apparentZenithDeg(:);
    cosz = max(cosd(z), 0);

    % Kasten-Young relative airmass (apparent elevation form)
    am = 1 ./ (cosz + 0.50572*(96.07995 - z).^(-1.6364));
    night = z >= 90;
    am(night) = NaN;

    % Pressure-corrected (absolute) airmass
    pressure = 101325 * (1 - 2.25577e-5*altitude).^5.25588;
    amAbs    = am * (pressure/101325);

    fh1 = exp(-altitude/8000);
    fh2 = exp(-altitude/1250);
    cg1 = 5.09e-5*altitude + 0.868;
    cg2 = 3.92e-5*altitude + 0.0387;

    I0 = 1364;   % mean extraterrestrial normal irradiance [W/m^2]

    % Global horizontal irradiance
    ghi = cg1 .* I0 .* cosz ...
        .* exp(-cg2 .* amAbs .* (fh1 + fh2*(linkeTurbidity - 1))) ...
        .* exp(0.01 * amAbs.^1.8);
    ghi = max(ghi, 0);

    % Direct normal irradiance (Ineichen 2002; pvlib eqns 3 & 4)
    b    = 0.664 + 0.163./fh1;
    bnci = b .* I0 .* exp(-0.09 * amAbs .* (linkeTurbidity - 1));
    bnci = max(bnci, 0);
    bnci2 = (1 - (0.1 - 0.2*exp(-linkeTurbidity)) ./ (0.1 + 0.882./fh1)) ./ cosz;
    bnci2 = ghi .* min(max(bnci2, 0), 1e20);
    dni   = min(bnci, bnci2);

    % Diffuse horizontal as the closure term
    dhi = max(ghi - dni.*cosz, 0);

    % Night / undefined -> 0
    bad = night | isnan(am);
    ghi(bad) = 0; dni(bad) = 0; dhi(bad) = 0;
end
