import rasterio
import numpy as np
import matplotlib.pyplot as plt

# List of 12 JP2 image paths
jp2_paths = [
    r"C:\Users\hp\Desktop\Talha\Study Projects\ES-304 Project\lahore_data\GRANULE\L2A_T43RDQ_A034923_20231113T054822\IMG_DATA\R60m\T43RDQ_20231113T054059_B01_60m.jp2",
    r"C:\Users\hp\Desktop\Talha\Study Projects\ES-304 Project\lahore_data\GRANULE\L2A_T43RDQ_A034923_20231113T054822\IMG_DATA\R60m\T43RDQ_20231113T054059_B02_60m.jp2",
    r"C:\Users\hp\Desktop\Talha\Study Projects\ES-304 Project\lahore_data\GRANULE\L2A_T43RDQ_A034923_20231113T054822\IMG_DATA\R60m\T43RDQ_20231113T054059_B03_60m.jp2",
    r"C:\Users\hp\Desktop\Talha\Study Projects\ES-304 Project\lahore_data\GRANULE\L2A_T43RDQ_A034923_20231113T054822\IMG_DATA\R60m\T43RDQ_20231113T054059_B04_60m.jp2",
    r"C:\Users\hp\Desktop\Talha\Study Projects\ES-304 Project\lahore_data\GRANULE\L2A_T43RDQ_A034923_20231113T054822\IMG_DATA\R60m\T43RDQ_20231113T054059_B05_60m.jp2",
    r"C:\Users\hp\Desktop\Talha\Study Projects\ES-304 Project\lahore_data\GRANULE\L2A_T43RDQ_A034923_20231113T054822\IMG_DATA\R60m\T43RDQ_20231113T054059_B06_60m.jp2",
    r"C:\Users\hp\Desktop\Talha\Study Projects\ES-304 Project\lahore_data\GRANULE\L2A_T43RDQ_A034923_20231113T054822\IMG_DATA\R60m\T43RDQ_20231113T054059_B07_60m.jp2",
    r"C:\Users\hp\Desktop\Talha\Study Projects\ES-304 Project\lahore_data\GRANULE\L2A_T43RDQ_A034923_20231113T054822\IMG_DATA\R60m\T43RDQ_20231113T054059_B8A_60m.jp2",
    r"C:\Users\hp\Desktop\Talha\Study Projects\ES-304 Project\lahore_data\GRANULE\L2A_T43RDQ_A034923_20231113T054822\IMG_DATA\R60m\T43RDQ_20231113T054059_B09_60m.jp2",
    
    r"C:\Users\hp\Desktop\Talha\Study Projects\ES-304 Project\lahore_data\GRANULE\L2A_T43RDQ_A034923_20231113T054822\IMG_DATA\R60m\T43RDQ_20231113T054059_B11_60m.jp2",
    r"C:\Users\hp\Desktop\Talha\Study Projects\ES-304 Project\lahore_data\GRANULE\L2A_T43RDQ_A034923_20231113T054822\IMG_DATA\R60m\T43RDQ_20231113T054059_B12_60m.jp2"
]

# Initialize an empty list to store the bands
band_list = []

# Create subplots for each band
fig, axes = plt.subplots(3, 4, figsize=(15, 10))

# Loop through each JP2 image
for i, (jp2_path, ax) in enumerate(zip(jp2_paths, axes.flatten()), 1):
    # Open the JP2 image
    src = rasterio.open(jp2_path)
    
    # Read the band and append it to the list
    band = src.read(1)
    band_list.append(band)
    
    # Display the band
    ax.imshow(band, cmap='hot')
    ax.set_title(f"Band {i}")
    ax.axis('off')

    # Close the raster file
    src.close()

# Concatenate the bands along the axis
concatenated_image = np.concatenate(band_list, axis=0)

# Visualize the concatenated image
plt.figure(figsize=(10, 8))
plt.imshow(concatenated_image, cmap='hot')
plt.title("Concatenated Image")
plt.axis('off')
plt.show()
