import numpy as np
import matplotlib.pyplot as plt
from skimage.color import label2rgb
from sklearn.cluster import KMeans
from spectral import *
import pickle  # For saving the data
from docx import Document
from matplotlib.patches import Patch


def load_cube(hdr_path, dat_path):
    header_file = str(hdr_path)
    spectral_file = str(dat_path)
    numpy_ndarr = envi.open(header_file, spectral_file)
    cube = numpy_ndarr.load().astype('float64')  # Load the entire cube as a numpy array
    return cube


# Create a new Word Document
doc = Document()
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

# Assuming seg_map_struct is your segmentation map array with label indices
labeled_image = label2rgb(seg_map_struct, bg_label=0)  # Specify bg_label if there's a background label

# Reshape the labeled image for easier color extraction
reshaped_image = labeled_image.reshape((-1, 3))
unique_colors = np.unique(reshaped_image, axis=0)

# Plot the image
plt.figure(figsize=(10, 8))
plt.imshow(labeled_image)
plt.title('Tait Segmented Image')

# Create legend patches
from matplotlib.patches import Patch
legend_handles = [Patch(color=unique_colors[i], label=f'Segment {i+1}') for i in range(len(unique_colors))]
plt.legend(handles=legend_handles, loc='best')

# Save and show plot
plt.savefig('Segmented_img_with_legend.png', bbox_inches='tight')
plt.close()

doc.add_picture(f"Segmented_img.png")

doc.save('segmentation.docx')
