from math import sin, cos, sqrt
from math import pi, radians

array_center_gps = [51.51878, 5.85821]
radius = 6371000 # Stolen from some random website: https://planetcalc.com/7721/

def shitcum(coords):
    acg = [radians(array_center_gps[0]), radians(array_center_gps[1])]
    coords = [radians(coords[0]), radians(coords[1])]
    dlat = coords[0]-acg[0]
    dlon = coords[1]-acg[1]

    mlat = (coords[0]+acg[0])/2

    xg, yg = radius * dlon * cos(mlat), radius * dlat
    converted = [-xg * cos(pi/6) - yg * sin(pi/6), xg * sin(pi/6) - yg * cos(pi/6)]
    return [xg, yg]

def main():
    test_coords = [51.52, 5.9]
    transformed = shitcum(test_coords)
    print(sqrt(transformed[0]**2 + transformed[1]**2))

if __name__=="__main__":
    main()

