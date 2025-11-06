# -*- coding: utf-8 -*-
"""
Camouflage Breaking Algorithm
Edge Detection and Convexity Analysis for Camouflaged Animals
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# CONFIGURATION
# =============================================================================

DATASET_PATH = "D:\\CamouflageBreaking\\data\\Camo Animals\\"
ROOT_DATA_PATH = DATASET_PATH

# =============================================================================
# SECTION 1: DATASET EXPLORATION
# =============================================================================

def explore_dataset(root_path):
    """List available categories and count images in the dataset."""
    print("=" * 50)
    print("DATASET EXPLORATION")
    print("=" * 50)
    
    # List available animal folders
    available_folders = [d for d in os.listdir(root_path) 
                        if os.path.isdir(os.path.join(root_path, d))]
    print(f"\nAvailable animal folders: {available_folders}")
    
    # Collect all image files
    all_image_files = []
    for dirpath, _, filenames in os.walk(root_path):
        for filename in filenames:
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                all_image_files.append(os.path.join(dirpath, filename))
    
    print(f"Found {len(all_image_files)} total images in the dataset\n")
    
    return all_image_files

# =============================================================================
# SECTION 2: IMAGE LOADING AND PREPROCESSING
# =============================================================================

def load_and_preprocess_image(image_path):
    """Load an image and convert it to RGB and grayscale."""
    original_image = cv2.imread(image_path)
    if original_image is None:
        raise FileNotFoundError(f"Failed to load image at: {image_path}")
    
    # Convert BGR to RGB for display
    original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    # Convert to grayscale for processing
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    
    # Print image properties
    height, width = gray_image.shape
    channels = original_image.shape[2] if len(original_image.shape) == 3 else 1
    file_format = os.path.splitext(image_path)[1].upper().replace('.', '')
    
    print(f"Image Properties:")
    print(f"  Path: {image_path}")
    print(f"  Format: {file_format}")
    print(f"  Dimensions: {height}x{width} pixels")
    print(f"  Channels: {channels}\n")
    
    return original_image_rgb, gray_image

# =============================================================================
# SECTION 3: EDGE DETECTION ALGORITHMS
# =============================================================================

def run_d_arg_pipeline(gray_image, percentile=65, show_steps=False):
    """
    D_arg Convexity Detector using Derivative of Gaussian (DoG) method.
    
    Args:
        gray_image: Single-channel grayscale image
        percentile: Percentile for automatic thresholding (0-100)
        show_steps: If True, display intermediate processing steps
    
    Returns:
        D_arg squared map
    """
    # Step 1: Noise reduction with Gaussian filter
    blurred_image = cv2.GaussianBlur(gray_image, (31, 31), 0)
    
    if show_steps:
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(gray_image, cmap='gray')
        plt.title("Step 1a: Original Grayscale")
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(blurred_image, cmap='gray')
        plt.title("Step 1b: After Gaussian Blur (31x31)")
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        diff = cv2.absdiff(gray_image, blurred_image)
        plt.imshow(diff, cmap='gray')
        plt.title("Step 1c: Difference (Noise Removed)")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def calculate_y_arg(image):
        """Calculate the y-derivative of gradient argument."""
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=11)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=11)
        theta = np.arctan2(grad_y, grad_x)
        return cv2.Sobel(theta, cv2.CV_64F, 0, 1, ksize=11)
    
    # Get original dimensions
    h, w = blurred_image.shape
    
    # Initialize result map with original dimensions
    d_arg_map = np.zeros((h, w), dtype=np.float64)
    
    # Rotation dictionaries
    rotations = {
        0: None, 
        90: cv2.ROTATE_90_CLOCKWISE, 
        180: cv2.ROTATE_180, 
        270: cv2.ROTATE_90_COUNTERCLOCKWISE
    }
    inv_rotations = {
        0: None, 
        90: cv2.ROTATE_90_COUNTERCLOCKWISE, 
        180: cv2.ROTATE_180, 
        270: cv2.ROTATE_90_CLOCKWISE
    }
    
    # Step 2: Process in all directions (isotropic operator)
    rotation_results = {}
    
    for angle, rot_code in rotations.items():
        rotated_img = cv2.rotate(blurred_image, rot_code) if rot_code else blurred_image
        y_arg_rotated = calculate_y_arg(rotated_img)
        y_arg_unrotated = cv2.rotate(y_arg_rotated, inv_rotations[angle]) if inv_rotations[angle] else y_arg_rotated
        
        # Ensure shape matches for accumulation
        if y_arg_unrotated.shape != (h, w):
            print(f"Warning: Shape mismatch at angle {angle}. Expected {(h, w)}, got {y_arg_unrotated.shape}")
            y_arg_unrotated = cv2.resize(y_arg_unrotated, (w, h))
        
        rotation_results[angle] = y_arg_unrotated
        d_arg_map += y_arg_unrotated
    
    if show_steps:
        # Show gradient computation details for 0° rotation
        grad_x = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=11)
        grad_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=11)
        theta = np.arctan2(grad_y, grad_x)
        
        plt.figure(figsize=(20, 10))
        plt.subplot(2, 4, 1)
        plt.imshow(blurred_image, cmap='gray')
        plt.title("Step 2a: Blurred Image")
        plt.axis('off')
        
        plt.subplot(2, 4, 2)
        plt.imshow(grad_x, cmap='gray')
        plt.title("Step 2b: Gradient X (Sobel)")
        plt.axis('off')
        
        plt.subplot(2, 4, 3)
        plt.imshow(grad_y, cmap='gray')
        plt.title("Step 2c: Gradient Y (Sobel)")
        plt.axis('off')
        
        plt.subplot(2, 4, 4)
        plt.imshow(theta, cmap='hsv')
        plt.title("Step 2d: Gradient Angle θ")
        plt.axis('off')
        
        plt.subplot(2, 4, 5)
        plt.imshow(rotation_results[0], cmap='gray')
        plt.title("Step 2e: Y-derivative of θ (0°)")
        plt.axis('off')
        
        plt.subplot(2, 4, 6)
        plt.imshow(rotation_results[90], cmap='gray')
        plt.title("Step 2f: Rotated result (90°)")
        plt.axis('off')
        
        plt.subplot(2, 4, 7)
        plt.imshow(rotation_results[180], cmap='gray')
        plt.title("Step 2g: Rotated result (180°)")
        plt.axis('off')
        
        plt.subplot(2, 4, 8)
        plt.imshow(rotation_results[270], cmap='gray')
        plt.title("Step 2h: Rotated result (270°)")
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Show accumulation
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(d_arg_map, cmap='gray')
        plt.title("Step 3: Sum of all rotations (D_arg map)")
        plt.axis('off')
        plt.colorbar()
        
        plt.subplot(1, 2, 2)
        plt.hist(d_arg_map.flatten(), bins=100, color='blue', alpha=0.7)
        plt.title("Step 3b: Distribution of D_arg values")
        plt.xlabel("D_arg value")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()
    
    # Step 3: Accentuate high values
    d_arg_squared = d_arg_map ** 2
    
    if show_steps:
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(d_arg_map, cmap='gray')
        plt.title("Step 4a: D_arg map")
        plt.axis('off')
        plt.colorbar()
        
        plt.subplot(1, 3, 2)
        plt.imshow(d_arg_squared, cmap='gray')
        plt.title("Step 4b: D_arg squared (enhanced)")
        plt.axis('off')
        plt.colorbar()
        
        plt.subplot(1, 3, 3)
        threshold_value = np.percentile(d_arg_squared, percentile)
        thresholded = np.zeros_like(d_arg_squared, dtype=np.uint8)
        thresholded[d_arg_squared > threshold_value] = 255
        plt.imshow(thresholded, cmap='gray')
        plt.title(f"Step 4c: Thresholded ({percentile}th percentile)")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
        print(f"D_arg statistics:")
        print(f"  Min: {d_arg_squared.min():.2f}")
        print(f"  Max: {d_arg_squared.max():.2f}")
        print(f"  Mean: {d_arg_squared.mean():.2f}")
        print(f"  Threshold ({percentile}th percentile): {threshold_value:.2f}\n")
    
    return d_arg_squared


def run_canny_edge_detector(gray_image):
    """Canny edge detection algorithm."""
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
    edges = cv2.Canny(blurred, 100, 200)
    return edges


def run_sobel_edge_detector(gray_image):
    """Sobel edge detection algorithm."""
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 2)
    grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5)
    gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    gradient_normalized = cv2.normalize(gradient_magnitude, None, 0, 255, 
                                       cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    _, binary_mask = cv2.threshold(gradient_normalized, 50, 255, cv2.THRESH_BINARY)
    return binary_mask


def run_prewitt_detector(gray_image):
    """Prewitt edge detection algorithm."""
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 2)
    kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    grad_x = cv2.filter2D(blurred, -1, kernel_x)
    grad_y = cv2.filter2D(blurred, -1, kernel_y)
    gradient_magnitude = np.sqrt(grad_x.astype(np.float64) ** 2 + grad_y.astype(np.float64) ** 2)
    gradient_normalized = cv2.normalize(gradient_magnitude, None, 0, 255, 
                                       cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    _, binary_mask = cv2.threshold(gradient_normalized, 50, 255, cv2.THRESH_BINARY)
    return binary_mask


def run_robert_cross_detector(gray_image):
    """Roberts Cross edge detection algorithm."""
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
    kernel_x = np.array([[1, 0], [0, -1]])
    kernel_y = np.array([[0, 1], [-1, 0]])
    grad_x = cv2.filter2D(blurred, -1, kernel_x)
    grad_y = cv2.filter2D(blurred, -1, kernel_y)
    gradient_magnitude = np.sqrt(grad_x.astype(np.float64) ** 2 + grad_y.astype(np.float64) ** 2)
    gradient_normalized = cv2.normalize(gradient_magnitude, None, 0, 255, 
                                       cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    _, binary_mask = cv2.threshold(gradient_normalized, 50, 255, cv2.THRESH_BINARY)
    return binary_mask


def run_log_detector(gray_image):
    """Laplacian of Gaussian (LoG) edge detection algorithm."""
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 1.5)
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    laplacian_abs = np.abs(laplacian)
    laplacian_normalized = cv2.normalize(laplacian_abs, None, 0, 255, 
                                        cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    _, binary_mask = cv2.threshold(laplacian_normalized, 50, 255, cv2.THRESH_BINARY)
    return binary_mask


def run_radial_symmetry_transform(gray_image, radii, alpha=2.0, beta=0.1, std_dev=1):
    """Radial Symmetry Transform algorithm."""
    h, w = gray_image.shape
    blurred = cv2.GaussianBlur(gray_image, (5, 5), std_dev)
    mag, ang = cv2.cartToPolar(cv2.Sobel(blurred, cv2.CV_32F, 1, 0),
                               cv2.Sobel(blurred, cv2.CV_32F, 0, 1))
    
    S = np.zeros((h, w), np.float32)
    mag_thresh = np.max(mag) * beta
    
    for y in range(h):
        for x in range(w):
            if mag[y, x] > mag_thresh:
                for r in radii:
                    p_x = int(round(x + r * np.cos(ang[y, x])))
                    p_y = int(round(y + r * np.sin(ang[y, x])))
                    if 0 <= p_x < w and 0 <= p_y < h:
                        S[p_y, p_x] += 1
    
    S = cv2.normalize(S, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    _, binary_mask = cv2.threshold(S, 0, 255, cv2.THRESH_BINARY)
    return binary_mask

# =============================================================================
# SECTION 4: VISUALIZATION
# =============================================================================

def compare_all_algorithms(image_path, radii=[5, 10, 15], show_d_arg_steps=True):
    """Run all algorithms on a single image and display results."""
    original_rgb, gray_image = load_and_preprocess_image(image_path)
    
    print("Running D_arg algorithm with detailed steps...")
    d_arg_output = run_d_arg_pipeline(gray_image, percentile=75, show_steps=show_d_arg_steps)
    
    print("\nRunning other edge detection algorithms...")
    canny_output = run_canny_edge_detector(gray_image)
    sobel_output = run_sobel_edge_detector(gray_image)
    prewitt_output = run_prewitt_detector(gray_image)
    robert_output = run_robert_cross_detector(gray_image)
    log_output = run_log_detector(gray_image)
    radial_symmetry_output = run_radial_symmetry_transform(gray_image, radii)
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    ax = axes.ravel()
    
    ax[0].imshow(original_rgb)
    ax[0].set_title("Original Image", fontsize=14, fontweight='bold')
    ax[0].axis('off')
    
    ax[1].imshow(d_arg_output, cmap='gray')
    ax[1].set_title("D_arg Convexity Detector", fontsize=14)
    ax[1].axis('off')
    
    ax[2].imshow(canny_output, cmap='gray')
    ax[2].set_title("Canny Edge Detector", fontsize=14)
    ax[2].axis('off')
    
    ax[3].imshow(sobel_output, cmap='gray')
    ax[3].set_title("Sobel Edge Detector", fontsize=14)
    ax[3].axis('off')
    
    ax[4].imshow(prewitt_output, cmap='gray')
    ax[4].set_title("Prewitt Edge Detector", fontsize=14)
    ax[4].axis('off')
    
    ax[5].imshow(robert_output, cmap='gray')
    ax[5].set_title("Roberts Cross Edge Detector", fontsize=14)
    ax[5].axis('off')
    
    ax[6].imshow(log_output, cmap='gray')
    ax[6].set_title("Laplacian of Gaussian (LoG)", fontsize=14)
    ax[6].axis('off')
    
    ax[7].imshow(radial_symmetry_output, cmap='gray')
    ax[7].set_title("Radial Symmetry Transform", fontsize=14)
    ax[7].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("Algorithm comparison complete!\n")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""
    print("\n" + "=" * 50)
    print("CAMOUFLAGE BREAKING ALGORITHM")
    print("=" * 50 + "\n")
    
    # Check if dataset path exists
    if not os.path.exists(ROOT_DATA_PATH):
        print(f"ERROR: Dataset path does not exist: {ROOT_DATA_PATH}")
        print("Please update DATASET_PATH at the top of the script.\n")
        return
    
    # Explore dataset
    all_image_files = explore_dataset(ROOT_DATA_PATH)
    
    if not all_image_files:
        print("ERROR: No images found in the dataset!")
        return
    
    # Test images from Bear folder
    test_images = [
        os.path.join(ROOT_DATA_PATH, "Bear", "camourflage_00164.jpg"),
        os.path.join(ROOT_DATA_PATH, "Bear", "camourflage_00072.jpg")
    ]
    
    # Verify test images exist
    for img_path in test_images:
        if not os.path.exists(img_path):
            print(f"ERROR: Test image not found: {img_path}")
            return
    
    # Run comparison on both test images
    for i, image_path in enumerate(test_images, 1):
        print(f"\n{'=' * 50}")
        print(f"PROCESSING TEST IMAGE {i}/2")
        print(f"{'=' * 50}\n")
        print(f"Selected image: {image_path}\n")
        compare_all_algorithms(image_path)
    
    print("=" * 50)
    print("ANALYSIS COMPLETE")
    print("=" * 50)


if __name__ == "__main__":
    main()