from math import sin, cos, sqrt, atan2
from math import pi, radians

array_center_gps = [51.51878, 5.85821]
radius = 6371000  # Stolen from some random website: https://planetcalc.com/7721/


def equirectangular(coords):
    acg = [radians(array_center_gps[0]), radians(array_center_gps[1])]
    coords = [radians(coords[0]), radians(coords[1])]
    dlat = coords[0] - acg[0]
    dlon = coords[1] - acg[1]

    mlat = (coords[0] + acg[0]) / 2

    xg, yg = radius * dlon * cos(mlat), radius * dlat
    converted = [-xg * cos(pi / 6) - yg * sin(pi / 6), xg * sin(pi / 6) - yg * cos(pi / 6)]
    return [xg, yg]


def haversine(coords):
    phi1, phi2 = radians(array_center_gps[0]), radians(coords[0])
    dphi = phi2 - phi1
    dlambda = radians(coords[1] - array_center_gps[1])

    a = sin(dphi / 2) ** 2 + cos(phi1) * cos(phi2) * (sin(dlambda / 2)) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    d = radius * c  # Distance to array

    y = sin(dlambda) * cos(phi2)
    x = cos(phi1) * sin(phi2) - sin(phi1) * cos(phi2) * cos(dlambda)
    theta = atan2(y, x)  # Bearing to array

    reltheta = pi + (theta + pi / 6)
    x, y = d * cos(reltheta), d * sin(reltheta)  # Relative coordinates
    # print("HAV")
    # print(x,y)

    return (x, y)
