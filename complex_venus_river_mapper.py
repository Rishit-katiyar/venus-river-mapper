# Import necessary libraries
from heapq import heappush, heappop, heapify
import sys
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt

# Constants for Venus surface
venus_average_surface_depth_km = 9  # in kilometers

# Calculate sea level based on average surface depth
sea_level = -venus_average_surface_depth_km

# Input and output file paths
file_input = "PIA00219.tif"  # Input image file path for Venus
file_output = "venus_rivers00001.tif"  # Output image file path for Venus

# Optional parameters for river mapping on Venus
seed = None  # Random seed for reproducibility
contrast = 500  # Contrast parameter for river-like structure generation
bit_depth = 16  # Bit depth of the output image
river_width_factor = 0.2  # Factor to control river-like structure width
river_limit = 0  # Limit for river-like structure detection

# Set random seed if provided
if seed:
    np.random.seed(seed=seed)

# Set recursion limit to avoid stack overflow
sys.setrecursionlimit(65536)

# Read the input image for Venus
print("Reading image")
input_image = imageio.imread(file_input)

# Convert to grayscale if image is color
if len(input_image.shape) == 3 and input_image.shape[2] == 3:
    input_image = np.dot(input_image[..., :3], [0.2989, 0.5870, 0.1140])

# Get the heightmap from the input image for Venus
heightmap = np.array(input_image)
(X, Y) = heightmap.shape

# Print information about the input image for Venus
print("Input image dimensions:", X, "x", Y)
print("Input image loaded successfully")

# Find start points for river-like structure mapping on Venus
print("Finding start points")

# Initialize visited array to keep track of visited pixels for Venus
visited = np.zeros((X, Y), dtype=bool)

# List to store start points for river-like structure mapping on Venus
start_points = []

# Function to add a start point to the list for Venus
def add_start_point(x, y):
    start_points.append((heightmap[x, y] + np.random.random(), x, y))
    visited[x, y] = True

# Counter for number of points to explore for Venus
to_explore = 0

# Iterate through each pixel to find start points for Venus
for x in range(1, X - 1):
    for y in range(1, Y - 1):
        # Check if the pixel is below sea level for Venus
        if heightmap[x, y] <= sea_level:
            continue
        to_explore += 1
        if to_explore % 1000000 == 0:
            print("Found", str(to_explore // 1000000), "millions points to explore")
        # Check if the pixel has a neighboring pixel below sea level for Venus
        if (heightmap[x - 1, y] <= sea_level or heightmap[x + 1, y] <= sea_level or
                heightmap[x, y - 1] <= sea_level or heightmap[x, y + 1] <= sea_level):
            add_start_point(x, y)

# Check pixels on the edges of the image for Venus
for x in range(X):
    if heightmap[x, 0] > sea_level:
        add_start_point(x, 0)
        to_explore += 1
    if heightmap[x, -1] > sea_level:
        add_start_point(x, Y - 1)
        to_explore += 1

for y in range(1, Y - 1):
    if heightmap[0, y] > sea_level:
        add_start_point(0, y)
        to_explore += 1
    if heightmap[-1, y] > sea_level:
        add_start_point(X - 1, y)
        to_explore += 1

# Print information about the start points for Venus
print("Found", str(len(start_points)), "start points for river-like structure mapping")

# Create a heap from the start points list for Venus
heap = start_points[:]
heapify(heap)

# Print information about river-like structure tree construction for Venus
print("Building river-like structure trees:", str(to_explore), "points to visit")

# Array to store flow directions for Venus
flow_dirs = np.zeros((X, Y), dtype=np.int8)

# Function to try pushing a pixel to the heap for Venus
def try_push(x, y):
    if not visited[x, y]:
        h = heightmap[x, y]
        if h > sea_level:
            heappush(heap, (h + np.random.random(), x, y))
            visited[x, y] = True
            return True
    return False

# Function to process neighboring pixels and simulate river-like structure flow for Venus
def process_neighbors(x, y):
    dirs = 0
    if x > 0 and try_push(x - 1, y):
        dirs += 1
    if y > 0 and try_push(x, y - 1):
        dirs += 2
    if x < X - 1 and try_push(x + 1, y):
        dirs += 4
    if y < Y - 1 and try_push(x, y + 1):
        dirs += 8
    # Simulate river-like structure flow direction based on neighboring pixels for Venus
    flow_dirs[x, y] = dirs

# Iterate until the heap is empty for Venus
while len(heap) > 0:
    t = heappop(heap)
    to_explore -= 1
    if to_explore % 1000000 == 0:
        print(str(to_explore // 1000000), "million points left", "Altitude:", int(t[0]), "Queue:", len(heap))
    process_neighbors(t[1], t[2])

# Cleanup visited and heightmap arrays for Venus
visited = None
heightmap = None

# Print information about water quantity calculation for Venus
print("Calculating water quantity")

# Array to store water quantity for Venus
waterq = np.ones((X, Y))

# Function to recursively set water quantity for each pixel for Venus
def set_water(x, y):
    water = 1
    dirs = flow_dirs[x, y]

    if dirs % 2 == 1:
        water += set_water(x - 1, y)
    dirs //= 2
    if dirs % 2 == 1:
        water += set_water(x, y - 1)
    dirs //= 2
    if dirs % 2 == 1:
        water += set_water(x + 1, y)
    dirs //= 2
    if dirs % 2 == 1:
        water += set_water(x, y + 1)
    waterq[x, y] = water
    return water

# Find the maximal water quantity for Venus
maxwater = 0
for start in start_points:
    water = set_water(start[1], start[2])
    if water > maxwater:
        maxwater = water

# Print information about maximal water quantity for Venus
print("Maximal water quantity:", str(maxwater))

# Cleanup flow_dirs array for Venus
flow_dirs = None

# Print information about image generation for Venus
print("Generating image")

# Calculate power for water quantity transformation for Venus
power = 1 / contrast

# Generate river-like structure map based on the specified parameters for Venus
if bit_depth <= 8:
    bit_depth = 8
    dtype = np.uint8
elif bit_depth <= 16:
    bit_depth = 16
    dtype = np.uint16
elif bit_depth <= 32:
    bit_depth = 32
    dtype = np.uint32
else:
    bit_depth = 64
    dtype = np.uint64

maxvalue = 2 ** bit_depth - 1
coeff = maxvalue / (maxwater ** power)

# Calculate river-like structure width based on water quantity for Venus
river_width = np.floor((waterq ** power) * (coeff * river_width_factor)).astype(dtype)

# Cleanup waterq array for Venus
waterq = None

# Save the generated image to the output file for Venus
imageio.imwrite(file_output, river_width)

# Print information about the output image for Venus
print("Output image saved to:", file_output)

# Display the input and output images for Venus
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Input image for Venus
axes[0].imshow(input_image, cmap='gray')
axes[0].set_title('Input Image')
axes[0].axis('off')

# Output image for Venus
axes[1].imshow(river_width)
axes[1].set_title('Generated River-like Structure Map')
axes[1].axis('off')

# Adjust layout to fit images for Venus
plt.tight_layout()

# Show the images for Venus
plt.show()
