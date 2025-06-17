# IQMetrics

**Physical Image Quality Estimation Metrics**

A Python and MATLAB toolkit for computing key physical image quality metrics—like Disk MTF, Line-Profile MTF, and NPS—from test images. Ideal for researchers and engineers working in medical imaging and general image quality assessment.

---

## 📁 Repository Structure

```
IQMetrics/
├── data/                   # Example or sample image datasets
├── matlab/                 # MATLAB scripts/functions (e.g., .m files)
├── diskmtf.py              # Python module to compute disk-based MTF
├── lpmtf.py                # Line Profile MTF computation
├── npsengine.py            # Noise Power Spectrum analysis tool
├── test.ipynb              # Quick Python notebook to test core functionality
├── README.md               # This readme
└── .gitignore              # Ignore rules
```

---

## 🧪 Features

- **Disk MTF** — Compute modulation transfer function using disk targets via `diskmtf.py`.
- **Line-Profile MTF** — Estimate MTF using line-profile methods in `lpmtf.py`.
- **Noise Power Spectrum (NPS)** — Analyze noise characteristics in images via `npsengine.py`.
- **MATLAB support** — Equivalent scripts available in the `matlab/` folder.
- **Interactive demos** — Use Jupyter notebooks (`.ipynb`) to visualize and benchmark metrics.

---

## 🧰 Installation & Requirements

### Python

1. Clone once:
   ```bash
   git clone https://github.com/mdselim-fda/IQMetrics.git
   cd IQMetrics
   ```
2. Install dependencies:
   ```bash
   pip install numpy scipy matplotlib scikit-image
   ```
3. Run test.ipynb notebook or import modules for your own workflow.
---

