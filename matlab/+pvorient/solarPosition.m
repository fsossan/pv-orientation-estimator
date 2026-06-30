function [apparentZenith, azimuth] = solarPosition(lat, lon, times)
%SOLARPOSITION  Apparent solar zenith and azimuth (NOAA algorithm).
%   [zen, az] = pvorient.solarPosition(lat, lon, times)
%   lat, lon : site latitude / longitude in degrees (east positive)
%   times    : datetime vector (interpreted as UTC; a tz-aware datetime is
%              converted to UTC)
%   zen      : apparent (refraction-corrected) solar zenith angle [deg]
%   az       : solar azimuth [deg], measured clockwise from North
%
%   Independent reimplementation of the NOAA solar-position equations
%   (toolbox-free). Agrees with pvlib's solar position to well within a tenth
%   of a degree, which is ample for building clearsky reference profiles.

    times = times(:);
    if ~isempty(times.TimeZone)
        times.TimeZone = 'UTC';
    end

    % Julian date from the true UTC instant (toolbox-free, tz-unambiguous):
    % JD of the Unix epoch (1970-01-01 00:00 UTC) is 2440587.5.
    jd = 2440587.5 + posixtime(times) / 86400;
    jc = (jd - 2451545.0) / 36525.0;        % Julian century from J2000

    % Geometric mean longitude / anomaly of the sun [deg]
    L0 = mod(280.46646 + jc.*(36000.76983 + jc*0.0003032), 360);
    M  = 357.52911 + jc.*(35999.05029 - 0.0001537*jc);

    % Sun's equation of centre [deg]
    C = sind(M).*(1.914602 - jc.*(0.004817 + 0.000014*jc)) ...
      + sind(2*M).*(0.019993 - 0.000101*jc) ...
      + sind(3*M)*0.000289;

    trueLong = L0 + C;

    % Apparent longitude (nutation/aberration) [deg]
    omega  = 125.04 - 1934.136*jc;
    lambda = trueLong - 0.00569 - 0.00478*sind(omega);

    % Obliquity of the ecliptic (mean + correction) [deg]
    seconds = 21.448 - jc.*(46.8150 + jc.*(0.00059 - jc*0.001813));
    eps0    = 23 + (26 + seconds/60)/60;
    epsCorr = eps0 + 0.00256*cosd(omega);

    % Solar declination [deg]
    decl = asind(sind(epsCorr).*sind(lambda));

    % Equation of time [minutes]
    ecc = 0.016708634;
    y   = tand(epsCorr/2).^2;
    L0r = deg2rad(L0);
    Mr  = deg2rad(M);
    eot = 4*rad2deg( y.*sin(2*L0r) - 2*ecc*sin(Mr) ...
        + 4*ecc.*y.*sin(Mr).*cos(2*L0r) - 0.5*y.^2.*sin(4*L0r) ...
        - 1.25*ecc^2*sin(2*Mr) );

    % True solar time [minutes] and hour angle [deg]
    minutesOfDay = hour(times)*60 + minute(times) + second(times)/60;
    tst = mod(minutesOfDay + eot + 4*lon, 1440);
    ha  = tst/4 - 180;
    ha(ha < -180) = ha(ha < -180) + 360;

    % True zenith
    cosZen = sind(lat).*sind(decl) + cosd(lat).*cosd(decl).*cosd(ha);
    cosZen = min(max(cosZen, -1), 1);
    zenith = acosd(cosZen);
    elev   = 90 - zenith;

    % Apparent elevation (atmospheric refraction)
    apparentElev   = elev + localRefraction(elev);
    apparentZenith = 90 - apparentElev;

    % Azimuth, clockwise from North
    denom = cosd(lat).*sind(zenith);
    cosAz = (sind(decl) - sind(lat).*cosd(zenith)) ./ denom;
    cosAz = min(max(cosAz, -1), 1);
    az = acosd(cosAz);                       % 0..180
    azimuth = zeros(size(az));
    pos = ha > 0;
    azimuth(pos)  = mod(az(pos) + 180, 360);
    azimuth(~pos) = mod(540 - az(~pos), 360);
    azimuth(denom == 0) = 0;                 % sun at zenith/nadir
end

function r = localRefraction(elevDeg)
%LOCALREFRACTION  NOAA atmospheric refraction approximation [degrees].
    r  = zeros(size(elevDeg));
    e  = elevDeg;
    te = tand(e);

    hi  = e > 85;
    md  = e > 5 & e <= 85;
    lo  = e > -0.575 & e <= 5;
    vlo = e <= -0.575;

    r(md)  = (58.1./te(md) - 0.07./te(md).^3 + 0.000086./te(md).^5) / 3600;
    r(lo)  = (1735 + e(lo).*(-518.2 + e(lo).*(103.4 + e(lo).*(-12.79 + e(lo)*0.711)))) / 3600;
    r(vlo) = (-20.774./te(vlo)) / 3600;
    r(hi)  = 0;
end
