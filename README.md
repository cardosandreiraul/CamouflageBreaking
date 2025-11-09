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

- Python 3.12.10 (recommended)
- Git

### Installation

#### Option 1: Using pip (recommended)

1. Clone the repository:
```bash
git clone https://github.com/Raul-Andrei-Cardos/CamouflageBreaking.git
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

#### Option 2: Using conda

1. Clone the repository:
```bash
git clone https://github.com/Raul-Andrei-Cardos/CamouflageBreaking.git
cd CamouflageBreaking
```

2. Create conda environment:
```bash
conda env create -f environment.yml
```

3. Activate the environment:
```bash
conda activate camouflage-breaking
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

## Usage

Run the main script:

```bash
python main.py
```

The script will:
1. Explore the dataset and count available images
2. Process test images from the Bear category
3. Apply all edge detection algorithms
4. Display step-by-step visualizations of the D_arg algorithm
5. Show a comparison of all algorithms side-by-side

### Customizing the Analysis

You can modify the test images in [`main.py`](main.py):

```python
test_images = [
    os.path.join(ROOT_DATA_PATH, "Bear", "camourflage_00164.jpg"),
    os.path.join(ROOT_DATA_PATH, "Bear", "camourflage_00072.jpg")
]
```

### Algorithm Parameters

- **D_arg Detector**: Adjust `percentile` (default: 75) for threshold sensitivity
- **Radial Symmetry**: Modify `radii` (default: [5, 10, 15]) for different object sizes
- **Gaussian Blur**: Kernel sizes can be adjusted in each algorithm function

## Output

The program generates:
- Detailed step-by-step visualizations of the D_arg algorithm
- Gradient computation steps (X, Y gradients, angle θ)
- Rotation results (0°, 90°, 180°, 270°)
- Statistical analysis (min, max, mean, threshold values)
- Side-by-side comparison of all 7 algorithms

## Dependencies

- **OpenCV (cv2)**: Image processing and computer vision
- **NumPy**: Numerical computations
- **Matplotlib**: Visualization and plotting

See [`requirements.txt`](requirements.txt) for exact versions.