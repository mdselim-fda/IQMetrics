import numpy as np
from scipy.optimize import curve_fit

def calculate_mtf(roi_lp, roi_wat, roi_uni):
    # Calculate the signal difference and noise components
    m_dash = np.std(roi_lp)
    noise_variance = (np.std(roi_uni) ** 2 + np.std(roi_wat) ** 2) / 2
    m_0 = abs(np.mean(roi_wat) - np.mean(roi_uni)) / 2

    # Calculate the MTF based on the provided formula
    m_squared = m_dash ** 2 - noise_variance
    m = np.sqrt(m_squared) if m_squared > 0 else 0
    return (np.pi * np.sqrt(2) * m) / (4 * m_0)

def gaussian_model(x, A, B):
    # Avoid overflow by clipping the exponent
    exponent = np.clip(-B * x**2, -700, 700)
    return A * np.exp(exponent)

def find_mtf_threshold(lp_mtf, threshold, x_values=[2, 3, 4, 5, 6, 7, 8, 9]):
    # Fit the Gaussian model to the MTF data
    try:
        popt, _ = curve_fit(gaussian_model, x_values, lp_mtf, maxfev=10000)
        A, B = popt
        return np.sqrt(-np.log(threshold / A) / B)
    except (RuntimeError, ValueError) as e:
        print(f"Error fitting Gaussian model: {e}")
        return None

def read_image(file_path, size, transpose=False):
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

class ACRLinePair:
    def __init__(self, img_path, nx=320, roi_size=15):
        self.path = img_path
        self.nx = nx
        self.roi_size = roi_size
        
        self.roi_coordinates = [
            [67, 223], [36, 149], [67, 74], [142, 43], 
            [216, 74], [247, 149], [216, 223], [142, 254]
        ]
        self.uni_roi_coordinates = [100, 153]
        self.con_roi_coordinates = [153, 153]

        self.img = read_image(img_path, nx)
        if self.img is None:
            raise ValueError(f"Failed to load image from '{img_path}'")

        self.uni_roi = self.extract_roi(self.uni_roi_coordinates)
        self.con_roi = self.extract_roi(self.con_roi_coordinates)
        self.mtfs = self.calculate_mtfs()

    def extract_roi(self, coordinates):
        x, y = coordinates
        return self.img[x:x+self.roi_size, y:y+self.roi_size]

    def calculate_mtfs(self):
        mtfs = []
        for coords in self.roi_coordinates:
            lp_roi = self.extract_roi(coords)
            mtf = calculate_mtf(lp_roi, self.uni_roi, self.con_roi)
            mtfs.append(mtf)
        return mtfs
    
    def get_frequency_at(self, threshold):
        return find_mtf_threshold(self.mtfs, threshold)
    
class PatientLinePair:
    def __init__(self, img_path, nx=512, roi_size=15):
        self.nx = nx
        self.roi_size = roi_size

        # Generate the file paths
        self.files = self.generate_file_list(img_path)

        # Define ROI coordinates
        self.roi_coordinates = [[240, 320], [240, 382]]
        self.uni_roi_coordinates = [293, 362]
        self.con_roi_coordinates = [260, 355]

        # Load images and extract ROIs
        self.imgs = self.load_images()
        self.uni_roi = self.extract_roi(self.uni_roi_coordinates)
        self.con_roi = self.extract_roi(self.con_roi_coordinates)

        # Calculate MTFs for each image
        self.mtfs = self.calculate_mtfs()

    def generate_file_list(self, img_path):
        suffixes = ['_lp_2_3', '_lp_4_5', '_lp_6_7', '_lp_8_9']
        return [img_path.replace('_lp_2_3', suffix) for suffix in suffixes]

    def load_images(self):
        # Pre-allocate array for efficiency
        num_files = len(self.files)
        imgs = np.zeros((self.nx, self.nx, num_files), dtype=np.int16)

        # Load each image file
        for i, file in enumerate(self.files):
            img = read_image(file, self.nx, transpose=True)
            if img is not None:
                imgs[:, :, i] = img
            else:
                print(f"Warning: Failed to load image '{file}'")

        return imgs

    def extract_roi(self, coordinates, img=None):
        x, y = coordinates
        if img is None:
            # Average over all loaded images for uniformity
            return np.mean(self.imgs[x:x+self.roi_size, y:y+self.roi_size, :], axis=2)
        return img[x:x+self.roi_size, y:y+self.roi_size]

    def calculate_mtfs(self):
        mtfs = []
        for i in range(self.imgs.shape[2]):
            img = self.imgs[:, :, i]
            for coords in self.roi_coordinates:
                lp_roi = self.extract_roi(coords, img=img)
                mtf = calculate_mtf(lp_roi, self.uni_roi, self.con_roi)
                mtfs.append(mtf)
        return mtfs
    
    def get_frequency_at(self, threshold):
        return find_mtf_threshold(self.mtfs, threshold)    