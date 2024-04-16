import numpy as np
import matplotlib.pyplot as plt
from skimage.color import label2rgb
from sklearn.cluster import KMeans
from spectral import *
import pickle  # For saving the data

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
