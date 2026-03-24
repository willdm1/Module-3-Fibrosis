'''Module 3: count black and white pixels and compute the percentage of white pixels in a .jpg image and extrapolate points'''
try:
    from termcolor import colored
except ModuleNotFoundError:
    def colored(text, *_args, **_kwargs):
        return str(text)
try:
    import cv2
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Missing dependency: OpenCV. Install it with:\n"
        "  pip install opencv-python\n"
        "If you're in Jupyter, run:\n"
        "  import sys; !{sys.executable} -m pip install opencv-python"
    ) from e
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import pandas as pd

# Load the images you want to analyze

filenames = [
    r"C:\Users\willm\OneDrive - University of Virginia\Will M\Undergraduate Research Courses-Grants\05_Spring_2026\Computational_Biomedical_BME_2315\Module-3-Fibrosis\chosen images\MASK_SK658 Slobe ch010147.jpg",
    r"C:\Users\willm\\OneDrive - University of Virginia\Will M\Undergraduate Research Courses-Grants\05_Spring_2026\Computational_Biomedical_BME_2315\Module-3-Fibrosis\chosen images\MASK_SK658 Slobe ch010149.jpg",
    r"C:\Users\willm\\OneDrive - University of Virginia\Will M\Undergraduate Research Courses-Grants\05_Spring_2026\Computational_Biomedical_BME_2315\Module-3-Fibrosis\chosen images\MASK_SK658 Slobe ch010156.jpg",
    r"C:\Users\willm\\OneDrive - University of Virginia\Will M\Undergraduate Research Courses-Grants\05_Spring_2026\Computational_Biomedical_BME_2315\Module-3-Fibrosis\chosen images\MASK_SK658 Slobe ch010157.jpg",
    r"C:\Users\willm\\OneDrive - University of Virginia\Will M\Undergraduate Research Courses-Grants\05_Spring_2026\Computational_Biomedical_BME_2315\Module-3-Fibrosis\chosen images\MASK_SK658 Slobe ch010158.jpg",
    r"C:\Users\willm\\OneDrive - University of Virginia\Will M\Undergraduate Research Courses-Grants\05_Spring_2026\Computational_Biomedical_BME_2315\Module-3-Fibrosis\chosen images\MASK_SK658 Slobe ch010159.jpg"
]

# Enter the depth of each image (in the same order that the images are listed above; you can find these in the .csv file provided to you which is tilted: "Filenames and Depths for Students")

depths = [
    3000,  # ch010147
    3350,  # ch010149
    330,   # ch010156
    750,   # ch010157
    920,   # ch010158
    860    # ch010159
]

# Make the lists that will be used

images = []
white_counts = []
black_counts = []
white_percents = []

# Build the list of all the images you are analyzing

for filename in filenames:
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image file: {filename}")
    images.append(img)

# For each image (until the end of the list of images), calculate the number of black and white pixels and make a list that contains this information for each filename.

for x in range(len(filenames)):
    _, binary = cv2.threshold(images[x], 127, 255, cv2.THRESH_BINARY)

    white = cv2.countNonZero(binary)
    black = binary.size - white

    white_counts.append(white)
    black_counts.append(black)

# Print the number of white and black pixels in each image.

print(colored("Counts of pixel by color in each image", "yellow"))
for x in range(len(filenames)):
    print(colored(f"White pixels in image {x}: {white_counts[x]}", "white"))
    print(colored(f"Black pixels in image {x}: {black_counts[x]}", "black"))
    print()

# Calculate the percentage of pixels in each image that are white and make a list that contains these percentages for each filename

for x in range(len(filenames)):
    white_percent = (
        100 * (white_counts[x] / (black_counts[x] + white_counts[x])))
    white_percents.append(white_percent)

# Print the filename (on one line in red font), and below that line print the percent white pixels and depth into the lung where the image was obtained

print(colored("Percent white px:", "yellow"))
for x in range(len(filenames)):
    print(colored(f'{filenames[x]}:', "red"))
    print(f'{white_percents[x]}% White | Depth: {depths[x]} microns')
    print()

'''Write your data to a .csv file'''

# Create a DataFrame that includes the filenames, depths, and percentage of white pixels
df = pd.DataFrame({
    'Filenames': filenames,
    'Depths': depths,
    'White percents': white_percents
})

# Write that DataFrame to a .csv file

df.to_csv('Percent_White_Pixels.csv', index=False)

print("The .csv file 'Percent_White_Pixels.csv' has been created.")

'''the .csv writing subroutine ends here'''


##############
# LECTURE 2: UNCOMMENT BELOW


# Interpolate a point: given a depth, find the corresponding white pixel percentage

interpolate_depth = float(input(colored("Enter the depth at which you want to interpolate a point (in microns): ", "yellow")))

x = np.array(depths, dtype=float)
y = np.array(white_percents, dtype=float)

order = np.argsort(x)
x = x[order]
y = y[order]

# You can also use 'quadratic', 'cubic', etc.
i = interp1d(x, y, kind='linear', bounds_error=False, fill_value=np.nan)
interpolate_point = float(i(interpolate_depth))
print(colored(f'The interpolated point is at the x-coordinate {interpolate_depth} and y-coordinate {interpolate_point}.', "green"))

depths_i = [float(d) for d in depths]
depths_i.append(interpolate_depth)
white_percents_i = white_percents[:]
white_percents_i.append(interpolate_point)

order_i = np.argsort(np.array(depths_i, dtype=float))
depths_i = [depths_i[j] for j in order_i]
white_percents_i = [white_percents_i[j] for j in order_i]

# make two plots: one that doesn't contain the interpolated point, just the data calculated from your images, and one that also contains the interpolated point (shown in red)
fig, axs = plt.subplots(2, 1)

axs[0].plot(depths, white_percents, marker='o', linestyle='-', color='blue')
axs[0].set_title('Plot of depth of image vs percentage white pixels')
axs[0].set_xlabel('depth of image (in microns)')
axs[0].set_ylabel('white pixels as a percentage of total pixels')
axs[0].grid(True)


axs[1].plot(depths_i, white_percents_i, marker='o', linestyle='-', color='blue')
axs[1].set_title('Plot of depth of image vs percentage white pixels with interpolated point (in red)')
axs[1].set_xlabel('depth of image (in microns)')
axs[1].set_ylabel('white pixels as a percentage of total pixels')
axs[1].grid(True)
axs[1].scatter(interpolate_depth, interpolate_point, color='red', s=100, label='Highlighted point')
axs[1].legend()

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()