import os

import folium
import numpy as np
import pickle as pcl
import osrm
import tqdm
import matplotlib.pyplot as plt

from config import myCoordinates, locations


class Route:
    distance = -1
    duration = -1
    coordinates = []

    def __init__(self, route_string):
        route = route_string
        self.distance = route['routes'][0]['distance']
        self.duration = route['routes'][0]['duration']
        self.coordinates = route['waypoints']


def km_to_coordinates(km, current_latitude):
    # Convert distance to change in latitude
    delta_latitude = km / 111.32

    # Convert distance to change in longitude
    delta_longitude = km / (111.32 * np.cos(np.radians(current_latitude)))

    return delta_latitude, delta_longitude


km_coordinate = km_to_coordinates(1, 48.03260251661082)
print(km_coordinate)


def crete_coordination_string(coordination1, coordination2):
    return str(coordination1) + "," + str(coordination2)


# Generate a range of kilometers in the x and y direction
x = np.arange(0, 161)
y = np.arange(0, 161)

# Use numpy.meshgrid to generate a grid of x and y coordinates
xv, yv = np.meshgrid(x, y)
xv = xv - 80
yv = yv - 80

# Calculate the change in latitude and longitude for each kilometer in the x and y direction
delta_latitude, delta_longitude = km_to_coordinates(xv, myCoordinates[0])
delta_latitude2, delta_longitude2 = km_to_coordinates(yv, myCoordinates[0])

# Add the change in latitude and longitude to the current latitude and longitude
new_latitude = myCoordinates[0] + np.flip(delta_latitude2, 0)
new_longitude = myCoordinates[1] + delta_longitude

# Stack the new latitude and longitude arrays along the third dimension
locations = np.dstack((new_latitude, new_longitude, np.zeros(locations.shape[:2])))

print("preparation done")

coordinate_strings = []
client = osrm.Client(host='http://localhost:5000')
locations = locations.reshape(-1, 3)
if not os.path.exists("locations.pkl"):
    for element in tqdm.tqdm(locations):
        # Get the coordinates at the current location
        coordinates = element
        routecoor = [list(myCoordinates)[::-1], coordinates[-2::-1]]

        route = client.route(coordinates=routecoor)
        thisroute = Route(route)
        element[2] = thisroute.distance

    with open("locations.pkl", "wb") as fp:
        pcl.dump(locations, fp)

with open("locations.pkl", "rb") as fp:
    locations = pcl.load(fp)
locations = locations.reshape(161, 161, 3)
print(locations)
print(locations[:, :, 2])
plt.imshow(locations[:, :, 2])
plt.show()
print("lul")
print(locations[(locations[:, :, 2] > 70000) & (locations[:, :, 2] < 80000)][:, :2])
outerCoordinates = locations[(locations[:, :, 2] > 76000) & (locations[:, :, 2] < 79000)][:, :2]
center = outerCoordinates.mean(axis=0)
print(center)
angles = np.arctan2(outerCoordinates[:, 0] - center[0], outerCoordinates[:, 1] - center[1])

print(angles)
sortingArray = np.argsort(angles)
polygon = outerCoordinates[sortingArray]
distances = np.linalg.norm(polygon - np.roll(polygon, -1, axis=0), axis=1)

polygon = polygon[(distances < 0.01) | (distances > 0.3)]
print(polygon)

# Create a Basemap instance
myMap = folium.Map(location=myCoordinates, zoom_start=12)
folium.Marker(center).add_to(myMap)
folium.Polygon(polygon, fill=True, fill_color='YlGn',
               fill_opacity=0.7).add_to(myMap)
myMap.show_in_browser()
