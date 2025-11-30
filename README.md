# CamouflageBreaking

A computer vision project implementing multiple edge detection algorithms to identify camouflaged animals in images.

## Description

This project implements and compares various edge detection algorithms specifically designed to detect camouflaged animals. The main algorithm is the **D_arg Convexity Detector** using the Derivative of Gaussian (DoG) method, which is compared against traditional edge detection methods.

## Implemented Algorithms

- **D_arg Convexity Detector** - Custom convexity-based detection using DoG
- **Canny Edge Detector** - Classic edge detection algorithm
- **Sobel Edge Detector** - Gradient-based edge detection
- **Prewitt Edge Detector** - Similar to Sobel with different kernel
- **Roberts Cross Edge Detector** - Simple 2x2 gradient operator
- **Laplacian of Gaussian (LoG)** - Second derivative-based detection
- **Radial Symmetry Transform** - Symmetry-based feature detection

## Getting Started

### Prerequisites

- Python 3.12.10+
- Git

### Installation

1. Clone the repository:
```bash
git clone https://github.com/cardosandreiraul/CamouflageBreaking
cd CamouflageBreaking
```

2. Create a virtual environment:
```bash
python -m venv venv
```

3. Activate the virtual environment:
   - **Windows**:
     ```bash
     venv\Scripts\activate
     ```
   - **macOS/Linux**:
     ```bash
     source venv/bin/activate
     ```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

### Dataset Setup

 **Important**: The dataset is **not included** in this repository due to size constraints.

1. Download the "Camo Animals" dataset (or use your own camouflage images)
2. Place the dataset in the `data/` directory:
   ```
   data/
   └── Camo Animals/
       ├── Bear/
       ├── Bird 1/
       └── ... (other categories)
   ```

3. Update the dataset path in [`main.py`](main.py) if needed:
   ```python
   DATASET_PATH = "D:\\CamouflageBreaking\\data\\Camo Animals\\"
   ```

### Customizing the Analysis

You can modify the test images in [`main.py`](main.py):

```python
test_images = [
    os.path.join(ROOT_DATA_PATH, "Bear", "camourflage_00164.jpg"),
    os.path.join(ROOT_DATA_PATH, "Bear", "camourflage_00072.jpg")
]
```