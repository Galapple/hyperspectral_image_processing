from matplotlib import pyplot as plt
import numpy as np
import spectral.io.envi as envi
from tabulate import tabulate
from scipy import integrate
from scipy.ndimage import convolve
# File paths
hdr_path = "converted_cube.hdr"  # Update this path
dat_path = "converted_cube.img"  # ENVI data file path, update this

def load_cube(hdr_path, dat_path):
    header_file = str(hdr_path)
    spectral_file = str(dat_path)
    numpy_ndarr = envi.open(header_file, spectral_file)
    cube = numpy_ndarr.load().astype('float64')  # Load the entire cube as a numpy array
    return cube


def compute_m(cube):
    num_lines, num_samples, num_bands = cube.shape

    # Define a 3x3 kernel with all ones and then zero the center
    # This kernel will sum up the values of neighbors excluding the center
    kernel = np.ones((3, 3))
    kernel[1, 1] = 0

    # Initialize the result array
    m_res = np.zeros_like(cube)

    # Apply convolution for each band separately
    for c in range(num_bands):
        # Compute the sum of neighbor pixels
        neighbor_sum = convolve(cube[:, :, c], kernel, mode='constant', cval=0.0)

        # Compute the number of valid neighbors for each pixel
        # To handle edges correctly, where neighbors can be outside the image
        valid_neighbor_count = convolve(np.ones_like(cube[:, :, c]), kernel, mode='constant', cval=0.0)

        # Compute the mean by dividing the sum by the count
        # Avoid division by zero by using np.where to leave zeros where the count is zero
        m_res[:, :, c] = np.where(valid_neighbor_count > 0, neighbor_sum / valid_neighbor_count, cube[:, :, c])

    return m_res


def plot_spectrum_graph(spectrum):
    # Plotting the spectrum graph
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(spectrum)), spectrum)
    plt.xlabel('Channel')
    plt.ylabel('Spectral Response')
    plt.title('Spectrum Graph of the Target Pixel')
    plt.grid(True)
    plt.show()


def show_img(cube):
    # Normalize the bands to [0, 1] range and select bands for RGB visualization
    red_normalized = (cube[:, :, 29] - cube[:, :, 29].min()) / (cube[:, :, 29].max() - cube[:, :, 29].min())
    green_normalized = (cube[:, :, 19] - cube[:, :, 19].min()) / (cube[:, :, 19].max() - cube[:, :, 19].min())
    blue_normalized = (cube[:, :, 9] - cube[:, :, 9].min()) / (cube[:, :, 9].max() - cube[:, :, 9].min())

    # Stack the bands to create an RGB image
    rgb_image = np.stack([red_normalized, green_normalized, blue_normalized], axis=-1)

    # Plotting the RGB image
    plt.figure(figsize=(8, 8))
    plt.imshow(rgb_image)
    plt.title('RGB Representation of Hyper-Spectral Image')
    plt.axis('off')  # To turn off axis labels and ticks
    plt.show()


def target_implant_table(NT, WT):
    # Define the table data as a list of lists
    table = [
        ["Pixel location", "NT", "WT"],
        ["(5,5)", NT[4, 4, 0], WT[4, 4, 0]],
        ["(10,10)", NT[9, 9, 0], WT[9, 9, 0]],
        ["(15,15)", NT[14, 14, 0], WT[14, 14, 0]],
        ["(20,20)", NT[19, 19, 0], WT[19, 19, 0]]
    ]

    print("\tvalues of (x'-m):\n")
    print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))


def draw_table(NT, WT):
    # Define the table data as a list of lists
    table = [
        ["Pixel location", "NT", "WT"],
        ["(5,5)", NT[4, 4], WT[4, 4]],
        ["(10,10)", NT[9, 9], WT[9, 9]],
        ["(15,15)", NT[14, 14], WT[14, 14]],
        ["(20,20)", NT[19, 19], WT[19, 19]]
    ]

    print("\tMatch Filter:\n")
    print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))


def histogram_of_match_filter(MatchFilter_NT_flat, MatchFilter_WT_flat, string_1, string2):
    # Calculate histograms without plotting them
    axis = np.linspace(-1000, 1500, 1000)
    if "ACE" in string_1:
        axis = np.linspace(-500, 500, 1000)
    NT_counts, NT_bins = np.histogram(MatchFilter_NT_flat, axis)
    WT_counts, WT_bins = np.histogram(MatchFilter_WT_flat, axis)

    # Plot the bin centers against the counts
    plt.figure(figsize=(10, 6))
    plt.plot(axis[:-1], NT_counts, label='NT')
    plt.plot(axis[:-1], WT_counts, label='WT')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title('Continuous Histogram of {} and {}'.format(string_1, string2))
    plt.legend()
    plt.show()
    return NT_counts, WT_counts


def ace_algorithm(cube, t_T, invers_cov, x_m):

    rows, cols, bands = cube.shape
    ace_scores = np.zeros((rows, cols))

    # Flatten t_T if it's not a 1D array already
    if t_T.ndim > 1:
        t_T = t_T.flatten()

    # Reshape x_m for matrix multiplication, if necessary
    if x_m.shape != cube.shape:
        x_m = x_m.reshape(cube.shape)

    # Compute the part of the ACE algorithm that doesn't change per pixel
    t_t_invers_cov = np.dot(t_T, invers_cov)

    # Calculate the denominator for the ACE algorithm
    denominator_t = np.sqrt(np.dot(t_t_invers_cov, t_T.T))

    for i in range(rows):
        for j in range(cols):
            # Select the pixel spectrum
            pixel_spectrum = x_m[i, j, :]

            # Calculate the numerator for the ACE algorithm
            numerator = np.dot(t_t_invers_cov, pixel_spectrum)

            # Calculate the denominator for the current pixel spectrum
            denominator_x = np.sqrt(np.dot(np.dot(pixel_spectrum.T, invers_cov), pixel_spectrum))

            # Calculate ACE score for the pixel
            if denominator_t > 0 and denominator_x > 0:
                ace_scores[i, j] = (numerator ** 2) / (denominator_t * denominator_x)
            else:
                ace_scores[i, j] = 0  # Handle the case of zero denominators

    return ace_scores


# Load the data cube
cube = load_cube(hdr_path, dat_path)
rows, col, bands = cube.shape

# Target pixel location (2, 4) in zero-indexed coordinates is (4, 2) in (lines, samples)
target_pixel_location = (652,97)#(410, 136)  # (lines, samples)
target_pixel_spectrum = cube[target_pixel_location[0], target_pixel_location[1], :]

# Plot the spectrum of the target pixel
plot_spectrum_graph(target_pixel_spectrum)

# calculate the parameters for x' eqution
print("calculate the parameters for x' eqution")
m_vector = compute_m(cube)
p = 0.01  # Target strength

# calculate the target implant matrix
print("calculate the target implant matrix")
t_i_NT = cube - m_vector  # x' = x - m
t_i_WT = t_i_NT + p * target_pixel_spectrum  # x' = x - m +p*t

# Reshape target_implant_NT to a 2D array where each row is a pixel and each column is a band
target_implant_2D = t_i_NT.reshape(-1, t_i_NT.shape[2])

# Compute the covariance matrix across all pixels (each pixel is an observation)
covariance_matrix = np.cov(target_implant_2D, rowvar=False)
inv_covariance_matrix = np.linalg.inv(covariance_matrix)
t_T = target_pixel_spectrum.transpose()
t = target_pixel_spectrum
temp = np.dot(t_T, inv_covariance_matrix)

MatchFilter_NT = np.zeros((rows,col))
MatchFilter_WT = np.zeros((rows,col))
ACE_NT = np.zeros((rows,col))
ACE_WT = np.zeros((rows,col))

for i in range(rows):
    for j in range(col):
        MatchFilter_NT[i,j] = np.dot(temp, t_i_NT[i, j, :])
        MatchFilter_WT[i,j] = np.dot(temp, t_i_WT[i, j, :])
ACE_WT = ace_algorithm(cube, t_T, inv_covariance_matrix, t_i_WT)
ACE_NT = ace_algorithm(cube, t_T, inv_covariance_matrix, t_i_NT)

# print the tables in choosen coordinates
target_implant_table(t_i_NT, t_i_WT)
draw_table(MatchFilter_NT, MatchFilter_WT)

# Flatten the MatchFilter_NT and MatchFilter_WT arrays to 1D arrays for histogram plotting
MatchFilter_NT_flat = MatchFilter_NT.flatten()
MatchFilter_WT_flat = MatchFilter_WT.flatten()
ACE_WT_flat = ACE_WT.flatten()
ACE_NT_flat = ACE_NT.flatten()

# Histogram of Match Filter with/without target.
MF_NT_counts, MF_WT_counts = histogram_of_match_filter(MatchFilter_NT_flat, MatchFilter_WT_flat, "MatchFilter_NT_flat", "MatchFilter_WT_flat")
ACE_NT_counts, ACE_WT_counts = histogram_of_match_filter(ACE_NT_flat, ACE_WT_flat, "ACE_Filter_NT_flat", "ACE_Filter_WT_flat")


X, Y, n_bands = cube.shape
x_size = len(MatchFilter_WT)
y_size = len(MatchFilter_NT)
axs = np.linspace(-1000, 1500, 101)

Pd_MF = 1 - integrate.cumulative_trapezoid(MF_WT_counts, axs[:-1], initial=0) / (25 * X * Y)
Pfa_MF = 1 - integrate.cumulative_trapezoid(MF_NT_counts, axs[:-1], initial=0) / (25 * X * Y)
Pd_ACE = 1 - integrate.cumulative_trapezoid(ACE_WT_counts, axs[:-1], initial=0) / (25 * X * Y)
Pfa_ACE = 1 - integrate.cumulative_trapezoid(ACE_NT_counts, axs[:-1], initial=0) / (25 * X * Y)

# Plot the ROC curve
plt.figure(figsize=(10, 6))
plt.plot(Pfa_MF, Pd_MF, label='ROC curve MF')
plt.plot(Pfa_ACE, Pd_ACE, label='ROC curve ACE')
plt.plot([0, 1], [0, 1], 'k--')  # Dashed diagonal
plt.xlim([0, 0.1])
plt.ylim([0, 1])
plt.xlabel('Pfa')
plt.ylabel('Pd')
plt.title('ROC Curve at Threshold = 0.1')
plt.legend()
plt.grid(True)
plt.show()

print("ace")
