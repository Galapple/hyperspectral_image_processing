import numpy as np
import matplotlib.pyplot as plt
from skimage.color import label2rgb
from sklearn.cluster import KMeans
from spectral import *
import pickle  # For saving the data


def ace_algorithm(cube, t_T, invers_cov, x_m):

    rows, cols, bands = cube.shape
    ace_scores = np.zeros((rows, cols))
    t = np.transpose(t_T)
    # Flatten t_T if it's not a 1D array already
    if t_T.ndim > 1:
        t_T = t_T.flatten()

    # Reshape x_m for matrix multiplication, if necessary
    if x_m.shape != cube.shape:
        x_m = x_m.reshape(cube.shape)

    # Compute the part of the ACE algorithm that doesn't change per pixel
    t_t_invers_cov = np.dot(t_T, invers_cov)

    # Calculate the denominator for the ACE algorithm
    denominator_t = np.dot(t_t_invers_cov, t)

    for i in range(rows):
        for j in range(cols):
            # Select the pixel spectrum
            pixel_spectrum = x_m[i, j, :]

            # Calculate the numerator for the ACE algorithm
            numerator = np.dot(t_t_invers_cov, pixel_spectrum)

            # Calculate the denominator for the current pixel spectrum
            phi_x_m = np.dot(invers_cov, pixel_spectrum)
            denominator_x = np.dot(np.dot(denominator_t, pixel_spectrum), phi_x_m)

            # Calculate ACE score for the pixel
            if denominator_t > 0 and denominator_x > 0:
                ace_scores[i, j] = (numerator ** 2) / (np.dot(denominator_t, denominator_x))
            else:
                ace_scores[i, j] = 0  # Handle the case of zero denominators

    return ace_scores


def load_cube(hdr_path, dat_path):
    header_file = str(hdr_path)
    spectral_file = str(dat_path)
    numpy_ndarr = envi.open(header_file, spectral_file)
    cube = numpy_ndarr.load().astype('float64')  # Load the entire cube as a numpy array
    return cube

# File paths
hdr_path = "converted_cube.hdr"
dat_path = "converted_cube.img"
# Load data
data = load_cube(hdr_path, dat_path)

# Normalize data
max_pix = np.max(data)
data = data / max_pix

# Prepare data for clustering
x_size, y_size, channels = data.shape
data_reshaped = np.reshape(data, (x_size * y_size, channels))

# Segmentation using KMeans
k = 5  # Number of clusters for segmentation
kmeans = KMeans(n_clusters=k, random_state=0).fit(data_reshaped)
seg_labels = kmeans.labels_
seg_map_struct = np.reshape(seg_labels, (x_size, y_size))

# Calculate mean pixels for each cluster
mean_pixels = np.zeros((k, channels))
for i in range(k):
    mean_pixels[i, :] = np.mean(data_reshaped[seg_labels == i], axis=0)

# Save the mean pixels to a file for later use
with open('mean_pixels.pkl', 'wb') as f:
    pickle.dump(mean_pixels, f)

# Display the segmented image
plt.figure()
plt.title('Tait Segmented Image - 5 segments')
plt.imshow(label2rgb(seg_map_struct))
plt.show()

print("Mean pixels for each cluster saved to 'mean_pixels.pkl'")
