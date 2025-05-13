import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import get_window
from scipy.optimize import curve_fit


# def sigmoid(x, a, b, c, d):
#     return a + (b - a) / (1 + 10 ** ((c - x) * d))

def sigmoid(x, a, b, c, d):
    z = (c - x) * d
    # Use the log-sum-exp trick for stability
    return a + (b - a) / (1 + np.exp(-z))

def sigm_fit(x, y, plot_flag=False):
    # Automatic initial parameter estimation
    min_est = np.quantile(y, 0.05)
    max_est = np.quantile(y, 0.95)
    x50_est = np.median(x)
    slope_est = 1
    initial_params = [min_est, max_est, x50_est, slope_est]

    try:
        popt, _ = curve_fit(sigmoid, x, y, p0=initial_params)
    except RuntimeError as e:
        print(f"Sigmoid fitting failed: {e}")
        return None, None

    # Calculate correlation coefficient for fit quality
    y_pred = sigmoid(x, *popt)
    rho = np.corrcoef(y, y_pred)[0, 1]

    # Plot if requested
    if plot_flag:
        plt.plot(x, y, 'k.', label="Data")
        plt.plot(x, y_pred, 'r-', label="Fitted Sigmoid")
        plt.title("Sigmoid Fit")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.show()

    return popt, rho

def get_mtf_from_disk_edge(imgdisk):
    # Image dimensions
    imgdisk = imgdisk[:, :, np.newaxis]
    nx, ny, nz = imgdisk.shape
    roisz = min(nx, ny)
    nn = 5  # Presampling rate
    delta = 1 / nn
    dist_uniform = np.arange(1, roisz / 2, delta)

    esfi = np.zeros((nz, len(dist_uniform)))

    # Meshgrid for distance calculation
    yy, xx = np.meshgrid(np.arange(0, ny), np.arange(0, nx)) #np.meshgrid(np.arange(1, ny + 1), np.arange(1, nx + 1))

    # Process each realization
    for iz in range(nz):
        disk_img = imgdisk[:, :, iz]

        # Find disk center
        disk_mean = round(np.mean(disk_img[roisz // 2 - 4:roisz // 2 + 5, roisz // 2 - 4:roisz // 2 + 5]))
        bkg_mean = round(np.mean(disk_img[:5, :]))
        if disk_mean < bkg_mean:
            disk_img = -disk_img
            disk_mean = -disk_mean
            bkg_mean = -bkg_mean

        # Edge threshold estimation
        edge_thr = np.arange(int(bkg_mean * 0.4 + disk_mean * 0.6), int(disk_mean), 10)
        x_centers = [np.mean(xx[disk_img >= thr]) for thr in edge_thr[:-1]]
        y_centers = [np.mean(yy[disk_img >= thr]) for thr in edge_thr[:-1]]
        x0, y0 = np.mean(x_centers), np.mean(y_centers) # Cleared from matlab

        # Calculate distances
        dist_to_ctr = np.sqrt((xx - x0) ** 2 + (yy - y0) ** 2).ravel()
        sorted_idx = np.argsort(dist_to_ctr)
        dist_sorted = dist_to_ctr[sorted_idx]
        pixel_values = disk_img.ravel()[sorted_idx]

        # Unique distances and mean pixel values
        dist_unique, unique_idx = np.unique(dist_sorted, return_index=True)
        esf_unique = np.array([np.mean(pixel_values[unique_idx == i]) for i in range(len(unique_idx))])
        esf_fitted = np.interp(dist_uniform, dist_unique, esf_unique)

        esfi[iz, :] = esf_fitted

    # Align and average ESF
    halflen = round(len(dist_uniform) / 4)
    esfi_aligned = np.zeros((nz, 2 * halflen + 1))
    
    for iz in range(nz):
        # Smooth the ESF with a Hann window
        l_flt = 11
    
        smoothflt = get_window('hann', l_flt, fftbins=False)
        smoothflt /= np.sum(smoothflt)  # Normalize
        esfi_smth = np.convolve(esfi[iz, :], smoothflt, mode='valid')        
        
        # Calculate LSF
        lsf_smth = (-esfi_smth[2:] + esfi_smth[:-2]) / (2 * delta)

        # Gaussian fit to find the edge location
        try:
            popt, _ = curve_fit(lambda x, a, b, c: a * np.exp(-(x - b) ** 2 / (2 * c ** 2)),
                                np.arange(len(lsf_smth)), lsf_smth, p0=[np.max(lsf_smth), np.argmax(lsf_smth), 10])
            edge_pix = int(popt[1])
        except RuntimeError:
            print("MTF failed because the disk image is too noisy.")
            return 0, 0, 0, 0

        if edge_pix - halflen < 0 or edge_pix + halflen >= len(esfi[iz, :]):
            print("MTF failed because the disk image is too noisy.")
            return 0, 0, 0, 0
        
        esfi_aligned[iz, :] = esfi[iz, edge_pix - halflen:edge_pix + halflen + 1]

    # Average aligned ESFs
    if nz>1:
        esf = np.mean(esfi_aligned, axis=0)
    else:
        esf = np.mean(esfi, axis=0) # Cleared from MATLAB output

    # Apply sigmoid fit to reduce noise on the flat part of the ESF
    center = round(len(esf) / 2)-1 # Python index strats with '0'
    half_width = round(len(esf) / 2.5)
    midseg = np.arange(center - half_width, center + half_width + 1)

    sigm_param, rho = sigm_fit(np.arange(0, len(esf[midseg])), esf[midseg])

    if rho is None or rho < 0.8:
        print("MTF failed because the disk image is too noisy.")
        return 0, 0, 0, 0

    # Replace ESF with the fitted version
    x_full = np.arange(len(esf))
    esf = sigmoid(x_full, *sigm_param) # Cleared from MATLAB output (var y in MATLAB)

    # Calculate LSF
    lsf = (- esf[2:] + esf[:-2] ) / (2 * delta) # Cleared from MATLAB code

    # Center the LSF
    center_pix = np.argmax(lsf)
    shortside_len = min(center_pix, len(lsf) - center_pix - 1)
    lsf_centered = lsf[center_pix - shortside_len:center_pix + shortside_len + 1]

    # Apply Hann filter
    dist_centered = np.arange(-shortside_len, shortside_len + 1) * delta
    hann_filter = 0.5 * (1 + np.cos(2 * np.pi * dist_centered / 20))
    lsf_filtered = lsf_centered * hann_filter

    # FFT for MTF
    lsf_fft = np.fft.fftshift(np.abs(np.fft.fft(lsf_filtered))) / np.sum(lsf_filtered)
    freq_vector = 1 / (2 * delta) * np.linspace(0, 1, (len(lsf_fft)+1) // 2)
    peak_index = np.argmax(lsf_fft)
    mtf = lsf_fft[peak_index:]
    #mtf = lsf_fft[len(lsf_fft) // 2:]

    # Return results
    success = 1
    return mtf, freq_vector, esf, success

# def get_mtf_from_disk_edge(imgdisk):
#     # Image dimensions
#     imgdisk = imgdisk[:, :, np.newaxis]
#     nx, ny, nz = imgdisk.shape
#     roisz = min(nx, ny)
#     nn = 5  # Presampling rate
#     delta = 1 / nn
#     dist_uniform = np.arange(1, roisz / 2, delta)
#     esfi = np.zeros((nz, len(dist_uniform)))

#     # Meshgrid for distance calculation
#     yy, xx = np.meshgrid(np.arange(1, ny + 1), np.arange(1, nx + 1))

#     # Process each realization
#     for iz in range(nz):
#         disk_img = imgdisk[:, :, iz]

#         # Find disk center
#         disk_mean = round(np.mean(disk_img[roisz // 2 - 4:roisz // 2 + 5, roisz // 2 - 4:roisz // 2 + 5]))
#         bkg_mean = round(np.mean(disk_img[:5, :]))
#         if disk_mean < bkg_mean:
#             disk_img = -disk_img
#             disk_mean = -disk_mean
#             bkg_mean = -bkg_mean

#         # Edge threshold estimation
#         #edge_thr = np.arange(int(bkg_mean * 0.4 + disk_mean * 0.6), int(disk_mean), 10)
#         edge_thr = np.linspace(bkg_mean * 0.4 + disk_mean * 0.6, disk_mean, int((disk_mean - (bkg_mean * 0.4 + disk_mean * 0.6)) / 10) + 1)
#         x_centers = [np.mean(xx[disk_img >= thr]) for thr in edge_thr]
#         y_centers = [np.mean(yy[disk_img >= thr]) for thr in edge_thr]
#         x0, y0 = np.mean(x_centers), np.mean(y_centers)

#         # Calculate distances
#         dist_to_ctr = np.sqrt((xx - x0) ** 2 + (yy - y0) ** 2).ravel()
#         sorted_idx = np.argsort(dist_to_ctr)
#         dist_sorted = dist_to_ctr[sorted_idx]
#         pixel_values = disk_img.ravel()[sorted_idx]

#         # Unique distances and mean pixel values
#         dist_unique, unique_idx = np.unique(dist_sorted, return_index=True)
#         esf_unique = np.array([np.mean(pixel_values[unique_idx == i]) for i in range(len(unique_idx))])
#         esf_fitted = np.interp(dist_uniform, dist_unique, esf_unique)

#         esfi[iz, :] = esf_fitted

#     # Align and average ESF
#     halflen = len(dist_uniform) // 4
#     esfi_aligned = np.zeros((nz, 2 * halflen + 1))
    
#     for iz in range(nz):
#         # Smooth the ESF with a Hann window
#         l_flt = 11
#         #smoothflt = get_window('hann', l_flt)
#         smoothflt = get_window('hann', l_flt, fftbins=False)
#         smoothflt /= np.sum(smoothflt)  # Normalize
#         esfi_smth = np.convolve(esfi[iz, :], smoothflt, mode='valid')        
#         #print(esfi_smth)
#         # Calculate LSF
#         lsf_smth = (esfi_smth[2:] - esfi_smth[:-2]) / (2 * delta)

#         # Gaussian fit to find the edge location
#         try:
#             popt, _ = curve_fit(lambda x, a, b, c: a * np.exp(-(x - b) ** 2 / (2 * c ** 2)),
#                                 np.arange(len(lsf_smth)), lsf_smth, p0=[np.max(lsf_smth), np.argmax(lsf_smth), 10])
#             edge_pix = int(popt[1])
#         except RuntimeError:
#             print("MTF failed because the disk image is too noisy.")
#             return 0, 0, 0, 0

#         if edge_pix - halflen < 0 or edge_pix + halflen >= len(esfi[iz, :]):
#             print("MTF failed because the disk image is too noisy.")
#             return 0, 0, 0, 0
        
#         esfi_aligned[iz, :] = esfi[iz, edge_pix - halflen:edge_pix + halflen + 1]

#     # Average aligned ESFs
#     if nz>1:
#         esf = np.mean(esfi_aligned, axis=0)
#     else:
#         esf = np.mean(esfi, axis=0)

#     # Apply sigmoid fit to reduce noise on the flat part of the ESF
#     center = round(len(esf) / 2)
#     half_width = round(len(esf) / 2.5)
#     midseg = np.arange(center - half_width, center + half_width + 1)
#     # sigm_param, rho = sigm_fit(midseg, esf[midseg])
#     sigm_param, rho = sigm_fit(np.arange(1, len(esf[midseg]) + 1), esf[midseg])
#     # sigm_param, rho = sigm_fit(np.arange(1,len(lsf_smth)), esf[midseg])
#     if rho is None or rho < 0.8:
#         print("MTF failed because the disk image is too noisy.")
#         return 0, 0, 0, 0

#     # Replace ESF with the fitted version
#     x_full = np.arange(len(esf))
#     esf = sigmoid(x_full, *sigm_param)

#     # Calculate LSF
#     #lsf = (esf[2:] - esf[:-2]) / (2 * delta) # This was the key bug from LLM conversion
#     lsf = (esf[:-2] - esf[2:]) / (2 * delta)

#     # Center the LSF
#     center_pix = np.argmax(lsf)
#     shortside_len = min(center_pix, len(lsf) - center_pix - 1)
#     lsf_centered = lsf[center_pix - shortside_len:center_pix + shortside_len + 1]

#     # Apply Hann filter
#     dist_centered = np.linspace(-shortside_len, shortside_len, len(lsf_centered))
#     hann_filter = 0.5 * (1 + np.cos(2 * np.pi * dist_centered / 20))
#     lsf_filtered = lsf_centered * hann_filter

#     # FFT for MTF
#     lsf_fft = np.fft.fftshift(np.abs(np.fft.fft(lsf_filtered))) / np.sum(lsf_filtered)
#     print()
#     freq_vector = 1 / (2 * delta) * np.linspace(0, 1, (97+1) // 2)
#     print("Delta and lsft")
#     print(delta, len(lsf_fft))
#     print(freq_vector)
#     mtf = lsf_fft[len(lsf_fft) // 2:]

#     # Return results
#     success = 1
#     return mtf, freq_vector, esf, success



def get_mtf_width(halfmtf, threshold, freq_vector):
    """
    Compute MTF width at the specified threshold value (default 50%).

    Parameters:
    - halfmtf (numpy array): The MTF curve (normalized or unnormalized).
    - threshold (float): The threshold at which to measure the MTF width.
    - freq_vector (numpy array): The corresponding frequency values.

    Returns:
    - width (float): The frequency at which the MTF crosses the given threshold.
    """
    # Normalize MTF if maximum is not 1
    mtf_max = np.max(halfmtf)
    if mtf_max != 1:
        halfmtf = halfmtf / mtf_max
    
    # Calculate delta from threshold
    delta = halfmtf - threshold

    # Find zero-cross points
    sign_delta = np.sign(delta[:-1] * delta[1:])
    id_cross_list = np.where(sign_delta <= 0)[0]

    # Check if no crossing found
    if len(id_cross_list) == 0:
        print("No crossing found at the specified threshold.")
        return -1  # Return -1 if no crossing is found

    # Use the first crossing
    id_cross = id_cross_list[0]

    # Check if the crossing is exact
    if sign_delta[id_cross] == 0:
        width = freq_vector[id_cross + 1]
    else:
        # Perform a linear interpolation to find the exact crossing frequency
        x1 = freq_vector[id_cross]
        y1 = halfmtf[id_cross]
        x2 = freq_vector[id_cross + 1]
        y2 = halfmtf[id_cross + 1]
        m = (y2 - y1) / (x2 - x1)
        width = (threshold - y1 + m * x1) / m

    return width