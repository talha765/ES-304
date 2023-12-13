import rasterio
import numpy as np
import matplotlib.pyplot as plt

# Function to standardize the data
def standardize(data):
    mean = np.mean(data, axis=0)
    std_dev = np.std(data, axis=0)
    standardized_data = (data - mean) / std_dev
    return standardized_data

# Function to calculate covariance matrix
def calculate_covariance_matrix(data):
    covariance_matrix = np.cov(data, rowvar=False)
    return covariance_matrix

# Function to calculate eigenvalues and eigenvectors
def calculate_eigen(covariance_matrix):
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    return eigenvalues, eigenvectors

# Function to sort eigenvalues and corresponding eigenvectors in descending order
def sort_eigen(eigenvalues, eigenvectors):
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    return sorted_eigenvalues, sorted_eigenvectors

# Function to project data onto principal components
def project_data(data, eigenvectors, num_components):
    projected_data = np.dot(data, eigenvectors[:, :num_components])
    return projected_data

# List of 12 JP2 image paths
jp2_paths = [
    # ... (your paths here)
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

# Standardize and concatenate the bands along the axis
stacked_bands = np.stack(band_list, axis=2)
reshaped_data = stacked_bands.reshape((-1, stacked_bands.shape[2]))

# Standardize the data
standardized_data = standardize(reshaped_data)

# Calculate the covariance matrix
covariance_matrix = calculate_covariance_matrix(standardized_data)

# Calculate eigenvalues and eigenvectors
eigenvalues, eigenvectors = calculate_eigen(covariance_matrix)

# Sort eigenvalues and corresponding eigenvectors
sorted_eigenvalues, sorted_eigenvectors = sort_eigen(eigenvalues, eigenvectors)

# Choose the number of principal components to retain (e.g., 3 for visualization)
num_components = 3

# Project the standardized data onto principal components
projected_data = project_data(standardized_data, sorted_eigenvectors, num_components)

# Reshape the projected data back to the original shape
pca_image = projected_data.reshape(stacked_bands.shape[0], stacked_bands.shape[1], num_components)

# Display the PCA image
fig, axes = plt.subplots(1, num_components, figsize=(15, 5))
for i in range(num_components):
    axes[i].imshow(pca_image[:, :, i], cmap='hot')
    axes[i].set_title(f"Transformed Image {i + 1}")
    axes[i].axis('off')

plt.show()
