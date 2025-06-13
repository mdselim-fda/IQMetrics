import numpy as np
import glob

def extract_noise(img, roi_size, xy=None):
    """
    Extract a noise region of interest (ROI) given a center location and ROI size.
    
    Parameters:
    - img (numpy.ndarray): 3D array (nx, ny, nsim) of image data.
    - center (tuple): (x, y) center coordinates of the ROI.
    - roi_size (tuple): (height, width) dimensions of the ROI.

    Returns:
    - noise_roi (numpy.ndarray): Extracted ROI as a 3D array.
    """
    nsim = img.shape[2]
    img_mean = np.mean(img, axis=2, keepdims=True)
    noise = (img - img_mean) * np.sqrt(nsim / (nsim - 1))
    
    # Extract ROI
    if xy is None:
        x, y = int(img.shape[0]/2), int(img.shape[1]/2) 
        h, w = roi_size, roi_size
        x_start = max(0, x - h // 2)
        x_end = min(img.shape[0], x + h // 2)
        y_start = max(0, y - w // 2)
        y_end = min(img.shape[1], y + w // 2)
    else:
        x_start = xy[0]
        x_end = xy[0]+roi_size
        y_start = xy[1]
        y_end = xy[1]+roi_size

    return noise[x_start:x_end, y_start:y_end, :]

def compute_nps(in_array):
    """
    Compute Noise Power Spectrum (NPS) from input noise images.
    
    Parameters:
        in_array: np.ndarray
            Input array of shape:
                (M, N)     -> 1D NPS
                (M, N, K)  -> 2D NPS from multiple 2D realizations
                (M, N, O, P) -> 3D NPS (not commonly used in medical CT, but supported)

    Returns:
        nps: np.ndarray
            Computed NPS
    """
    dims = in_array.shape
    ndim = in_array.ndim
    nrealization = dims[-1]

    if ndim == 2:
        # 1D NPS: column-wise FFT
        nps = np.zeros((dims[0],), dtype=np.float64)
        for i in range(nrealization):
            s = np.fft.fftshift(np.fft.fft(in_array[:, i]))
            nps += np.abs(s) ** 2
        nps /= (dims[0] * dims[1])

    elif ndim == 3:
        # 2D NPS
        nps = np.zeros((dims[0], dims[1]), dtype=np.float64)
        for i in range(nrealization):
            s = np.fft.fftshift(np.fft.fft2(in_array[:, :, i]))
            nps += np.abs(s) ** 2
        nps /= (dims[0] * dims[1] * dims[2])

    elif ndim == 4:
        # 3D NPS
        nps = np.zeros((dims[0], dims[1], dims[2]), dtype=np.float64)
        for i in range(nrealization):
            s = np.fft.fftshift(np.fft.fftn(in_array[:, :, :, i]))
            nps += np.abs(s) ** 2
        nps /= (dims[0] * dims[1] * dims[2] * dims[3])

    else:
        raise NotImplementedError("compute_nps: input dimensionality not supported.")

    return nps


import numpy as np
from scipy.ndimage import map_coordinates

def radial_profiles_fulllength(nps, ang_deg, df=1.0):
    """
    Extract radial profiles from the 2D NPS image.

    Parameters:
        nps (ndarray): 2D NPS image
        ang_deg (array-like): list of angles (in degrees)
        df (float): frequency sampling interval

    Returns:
        cx (ndarray): x-coordinates along radial lines
        cy (ndarray): y-coordinates along radial lines
        c (ndarray): intensity values along radial lines
        mc (ndarray): mean intensity per radius across all angles
    """
    xnps, ynps = nps.shape
    ang_rad = np.deg2rad(ang_deg)

    if xnps % 2 == 0:
        x = np.arange(-xnps/2, xnps/2) * df
        y = np.arange(-ynps/2, ynps/2) * df
    else:
        x = np.arange(-(xnps-1)//2, (xnps+1)//2) * df
        y = np.arange(-(ynps-1)//2, (ynps+1)//2) * df

    xx, yy = np.meshgrid(x, y, indexing='ij')

    nl = int(np.floor((xnps // 2) * np.sqrt(2)))
    r = (nl - 1) * df

    c = np.zeros((len(ang_rad), nl))
    cx = np.zeros_like(c)
    cy = np.zeros_like(c)

    for i, theta in enumerate(ang_rad):
        # Coordinates in physical space
        x_line = np.linspace(0, r * np.cos(theta), nl)
        y_line = np.linspace(0, r * np.sin(theta), nl)

        # Convert to image pixel indices (centered at middle of image)
        x_sample = x_line / df + xnps // 2
        y_sample = y_line / df + ynps // 2

        coords = np.vstack((x_sample, y_sample))  # rows: (x, y)
        c[i, :] = map_coordinates(nps.T, coords, order=3, mode='reflect')
        out_of_bounds = (x_sample < 0) | (x_sample >= xnps) | (y_sample < 0) | (y_sample >= ynps)
        c[i, out_of_bounds] = np.nan
        cx[i, :] = x_line
        cy[i, :] = y_line

    #     cx[i, :] = x_sample
    #     cy[i, :] = y_sample        

    # out_of_bounds = (cx < 0) | (cx >= xnps) | (cy < 0) | (cy >= ynps)
    # c[out_of_bounds] = np.nan
    
    # Average over angles
    nan_mask = np.isnan(c)
    c[nan_mask] = 0
    csum = np.sum(c, axis=0)
    cweight = np.sum(~nan_mask, axis=0)
    cweight[cweight == 0] = 1
    mc = csum / cweight

    return cx, cy, c, mc

def read_image(file_path, size, transpose=True):
    try:
        with open(file_path, 'rb') as f:
            img = np.fromfile(f, dtype=np.int16).reshape((size, size))
            return img.T if transpose else img
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"Error reading file '{file_path}': {e}")
        return None

class NPSAnalyzer:
    def __init__(self, img_path, nx=320, roi_size=64, roi_xy=None):
        self.nx = nx
        self.roisz = roi_size
        self.roi_xy=roi_xy

        # Generate the file paths
        self.files = glob.glob(img_path)

        # Load images and extract ROIs
        self.imgs = self.load_images()
        
        # Calculate MTFs for each image
        self.nps = self.get_2d_nps()

    def load_images(self):
        # Pre-allocate array for efficiency
        num_files = len(self.files)
        imgs = np.zeros((self.nx, self.nx, num_files), dtype=np.int16)

        # Load each image file
        for i, file in enumerate(self.files):
            img = read_image(file, self.nx)
            if img is not None:
                imgs[:, :, i] = img
            else:
                print(f"Warning: Failed to load image '{file}'")

        return imgs 
    def get_2d_nps(self):
        noise_roi = extract_noise(self.imgs, self.roisz, self.roi_xy)
        nps = compute_nps(noise_roi)

        maxnps = np.max(nps)
        center = int(self.roisz/2)
        if nps[center, center] == maxnps:
            nps[center-1:center+2, center-1:center+2] = 0 
        return nps  
    def get_1d_nps(self):
        ang = np.arange(0, 181)
        cx, cy, c, mc = radial_profiles_fulllength(self.nps, ang)
        fr = 0.5 * np.linspace(0, 1, len(mc))        
        return fr, mc   