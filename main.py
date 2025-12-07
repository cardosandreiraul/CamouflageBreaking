# -*- coding: utf-8 -*-
"""Camouflage Breaking Algorithm Suite.

Full Comparison: D_arg (Convexity) vs Radial Symmetry vs Edge Detectors

This suite compares various computer vision techniques for detecting camouflaged
objects:
1. D_arg (Convexity-based detection) - Novel proposed method
2. Fast Radial Symmetry Transform - Detects symmetric features
3. Traditional edge detectors (Canny, Sobel, Prewitt, Roberts, LoG)
"""

import os
from typing import Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# CONFIGURATION
# =============================================================================


# Root directory containing test images of camouflaged animals
ROOT_DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "Camo Animals")

# Algorithm Constants
PERCENTILE_MIN = 0
PERCENTILE_MAX = 99.5
SYMMETRY_CLIP_LIMIT = 100
RADIAL_ALPHA_DEFAULT = 2.0
DEFAULT_RADII = [10, 20, 30, 40]

# Preconfigured settings for specific images
settings = {
    "Bear": {
        "path": os.path.join(
            ROOT_DATA_PATH, "Bear", "images - 2020-07-02T154335.549.jpg"
        ),
        "gaussianblur": 101,
        "gradient_ksize": 3,
        "y_derivative_ksize": 17,
    },
    "Canine1": {
        "path": os.path.join(ROOT_DATA_PATH, "Canine 1", "camourflage_00265.jpg"),
        "gaussianblur": 101,
        "gradient_ksize": 3,
        "y_derivative_ksize": 17,
    },
    "Feline": {
        "path": os.path.join(ROOT_DATA_PATH, "Feline 2", "images (61).jpg"),
        "gaussianblur": 105,
        "gradient_ksize": 3,
        "y_derivative_ksize": 13,
    },
    "FlatFish": {
        "path": os.path.join(ROOT_DATA_PATH, "Flat Fish 1", "download (10).jpg"),
        "gaussianblur": 101,
        "gradient_ksize": 3,
        "y_derivative_ksize": 13,
    },
    "Bird": {
        "path": os.path.join(ROOT_DATA_PATH, "Bird 1", "000000175774.jpg"),
        "gaussianblur": 101,
        "gradient_ksize": 3,
        "y_derivative_ksize": 11,
    },
    "Canine2": {
        "path": os.path.join(ROOT_DATA_PATH, "Canine 2", "download (25).jpg"),
        "gaussianblur": 101,
        "gradient_ksize": 3,
        "y_derivative_ksize": 13,
    },
    "Canine3": {
        "path": os.path.join(ROOT_DATA_PATH, "Canine 1", "camourflage_00333.jpg"),
        "gaussianblur": 103,
        "gradient_ksize": 3,
        "y_derivative_ksize": 17,
    },
    "Bear2": {
        "path": os.path.join(ROOT_DATA_PATH, "Bear", "download (5).jpg"),
        "gaussianblur": 101,
        "gradient_ksize": 3,
        "y_derivative_ksize": 11,
    },
    "Canine4": {
        "path": os.path.join(ROOT_DATA_PATH, "Canine 2", "images (28).jpg"),
        "gaussianblur": 101,
        "gradient_ksize": 3,
        "y_derivative_ksize": 13,
    },
    "Bird2": {
        "path": os.path.join(ROOT_DATA_PATH, "Bird 2", "camourflage_00822.jpg"),
        "gaussianblur": 101,
        "gradient_ksize": 3,
        "y_derivative_ksize": 11,
    },
    "Canine5": {
        "path": os.path.join(ROOT_DATA_PATH, "Canine 2", "images (90).jpg"),
        "gaussianblur": 101,
        "gradient_ksize": 3,
        "y_derivative_ksize": 13,
    },
}

# =============================================================================
# UTILITIES
# =============================================================================


def robust_normalize(img: np.ndarray) -> np.ndarray:
    """Normalize an image while ignoring extreme outliers (hotspots).

    This function prevents bright hotspots from compressing the dynamic range
    of the rest of the image, which would result in a mostly black output.

    Parameters
    ----------
    img : ndarray
        Input image (any dtype).

    Returns
    -------
    ndarray
        Normalized image as uint8 (0-255 range).

    Notes
    -----
    - Clips top 0.5% of brightest pixels before normalization
    - Fixes 'Black Image' issue without introducing noise
    """
    img_float = img.astype(np.float32)

    # Find percentile values (ignore top 0.5% brightest outliers)
    v_min, v_max = np.percentile(img_float, (0, 99.5))

    # Clip extreme values to prevent them from squashing the rest
    img_clipped = np.clip(img_float, v_min, v_max)

    # Normalize to full 0-255 range
    return cv2.normalize(img_clipped, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)


# =============================================================================
# SECTION 1: D_ARG (CONVEXITY-BASED CAMOUFLAGE DETECTION)
# =============================================================================


def visualize_darg_detailed(
    img, blurred, gx, gy, theta, rot_results, final_sum, final_sq
):
    """Visualize the step-by-step pipeline of the D_arg algorithm.

    Creates a comprehensive visualization showing:
    - Preprocessing stages (original, blur, gradients)
    - Gradient orientation (theta map)
    - Rotational processing results (0°, 90°, 180°, 270°)
    - Final accumulation and squaring

    Parameters
    ----------
    img : ndarray
        Original grayscale image.
    blurred : ndarray
        Gaussian-blurred version.
    gx : ndarray
        Gradient component in x direction.
    gy : ndarray
        Gradient component in y direction.
    theta : ndarray
        Gradient orientation map (in radians).
    rot_results : dict
        Dictionary of rotational processing results {angle: result}.
    final_sum : ndarray
        Accumulated Y_arg derivatives from all rotations.
    final_sq : ndarray
        Final squared result (D_arg output).
    """
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(18, 12))
    fig.canvas.manager.set_window_title("D_arg Internals")
    plt.suptitle("D_arg Operator: Step-by-Step Visualization", fontsize=16)

    # Row 1: Preprocessing and gradient computation
    plt.subplot(3, 4, 1)
    plt.imshow(img, cmap="gray")
    plt.title("1. Original")
    plt.axis("off")

    plt.subplot(3, 4, 2)
    plt.imshow(blurred, cmap="gray")
    plt.title("2. Blur")
    plt.axis("off")

    # Gradient magnitude = sqrt(gx² + gy²)
    mag = np.sqrt(gx**2 + gy**2)
    plt.subplot(3, 4, 3)
    plt.imshow(mag, cmap="gray")
    plt.title("3. Magnitude")
    plt.axis("off")

    # Theta (orientation) shown in HSV colormap for better visualization
    plt.subplot(3, 4, 4)
    plt.imshow(theta, cmap="hsv")
    plt.title("4. Theta")
    plt.axis("off")

    # Row 2: Rotational processing results
    # Shows Y_arg derivative for each rotation angle
    rot_angles = [0, 90, 180, 270]
    for i, angle in enumerate(rot_angles):
        plt.subplot(3, 4, 5 + i)
        # Normalize each rotation result for visibility
        norm_rot = cv2.normalize(
            rot_results[angle], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U
        )
        plt.imshow(norm_rot, cmap="gray")
        plt.title(f"5.{i+1} Y_arg ({angle}°)")
        plt.axis("off")

    # Row 3: Final aggregation
    plt.subplot(3, 4, 9)
    plt.imshow(final_sum, cmap="gray")
    plt.title("6. Sum")
    plt.axis("off")

    plt.subplot(3, 4, 10)
    plt.imshow(final_sq, cmap="gray")
    plt.title("7. Squared (Final)")
    plt.axis("off")

    # Histogram shows distribution of final values (log scale for better visibility)
    plt.subplot(3, 4, 11)
    plt.hist(final_sq.flatten(), bins=50, color="black")
    plt.title("Hist")
    plt.yscale("log")

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)


def run_d_arg_pipeline(
    gray_image: np.ndarray,
    blur_ksize: int = 101,
    gradient_ksize: int = 3,
    y_arg_ksize: int = 17,
    visualize_matplotlib: bool = False,
    return_intermediates: bool = False,
) -> Union[np.ndarray, Tuple]:
    """Execute the D_arg (Convexity Detection) algorithm pipeline.

    The D_arg operator detects convex/concave features by measuring how the
    gradient orientation changes in the perpendicular direction (Y_arg derivative).
    This is rotation-invariant by accumulating results from 4 rotations
    (0°, 90°, 180°, 270°).

    Algorithm steps:
    1. Blur image to reduce noise
    2. For each rotation angle (0°, 90°, 180°, 270°):
       a. Rotate image
       b. Compute gradients (gx, gy) and orientation (theta)
       c. Calculate Y_arg = ∂theta/∂y (vertical derivative of orientation)
       d. Rotate result back to original orientation
    3. Sum all rotational results
    4. Square the sum to enhance convexity features

    Parameters
    ----------
    gray_image : ndarray
        Input grayscale image (uint8).
    show_steps : bool, default=True
        If True, displays detailed visualization of pipeline.
    blur_ksize : int, default=101
        Kernel size for Gaussian blur (must be odd).
    gradient_ksize : int, default=3
        Kernel size for gradient computation.
    y_arg_ksize : int, default=17
        Kernel size for Y_arg derivative computation.

    Returns
    -------
    ndarray
        Final D_arg result (float64), high values indicate convex features.
    """
    # Preprocessing: Gaussian blur reduces noise and stabilizes gradient computation
    blurred = cv2.GaussianBlur(gray_image, (blur_ksize, blur_ksize), 0)

    def get_gradients_and_theta(img):
        """Compute image gradients and orientation.

        Parameters
        ----------
        img : ndarray
            Input image.

        Returns
        -------
        gx : ndarray
            Gradient in x-direction.
        gy : ndarray
            Gradient in y-direction.
        theta : ndarray
            Gradient orientation (arctan2(gy, gx)).
        """
        # Sobel operator computes derivatives in x and y directions
        gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=gradient_ksize)
        gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=gradient_ksize)

        # Theta = angle of gradient vector
        theta = np.arctan2(gy, gx)
        return gx, gy, theta

    def calculate_y_arg_derivative(theta_map):
        """Calculate the vertical derivative of the gradient orientation.

        Y_arg = ∂theta/∂y measures how gradient direction changes vertically.
        Large values indicate convex/concave boundaries.

        Parameters
        ----------
        theta_map : ndarray
            Gradient orientation map (in radians).

        Returns
        -------
        ndarray
            Y_arg derivative (float64).
        """
        # Use larger kernel (17x17) to capture broader structural features
        return cv2.Sobel(theta_map, cv2.CV_64F, 0, 1, ksize=y_arg_ksize)

    # Initialize accumulator for summing results from all rotations
    h, w = gray_image.shape
    d_arg_accumulator = np.zeros((h, w), dtype=np.float64)

    # Dictionary to store intermediate results for visualization
    rotation_snapshots = {}

    # Define rotation angles and their OpenCV rotation codes
    # Format: (angle, forward_rotation_code, inverse_rotation_code)
    rotations = [
        (0, None, None),  # No rotation
        (90, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE),  # 90° CW
        (180, cv2.ROTATE_180, cv2.ROTATE_180),  # 180°
        (
            270,
            cv2.ROTATE_90_COUNTERCLOCKWISE,
            cv2.ROTATE_90_CLOCKWISE,
        ),  # 270° CW (= 90° CCW)
    ]

    # Variables to store gradient info from 0° rotation for visualization
    g_gx, g_gy, g_theta = None, None, None

    # Process each rotation angle
    for angle, rot_code, inv_rot_code in rotations:
        # Step 1: Rotate image (if needed)
        if rot_code is not None:
            curr_img = cv2.rotate(blurred, rot_code)
        else:
            curr_img = blurred  # 0° rotation = original

        # Step 2: Compute gradients and orientation for rotated image
        gx, gy, theta = get_gradients_and_theta(curr_img)

        # Step 3: Calculate Y_arg derivative (vertical derivative of theta)
        y_arg = calculate_y_arg_derivative(theta)

        # Store gradient info from first rotation for visualization
        if angle == 0:
            g_gx, g_gy, g_theta = gx, gy, theta

        # Step 4: Rotate result back to original orientation
        if inv_rot_code is not None:
            y_arg_unrotated = cv2.rotate(y_arg, inv_rot_code)
        else:
            y_arg_unrotated = y_arg

        # Ensure dimensions match (safety check after rotation)
        if y_arg_unrotated.shape != (h, w):
            y_arg_unrotated = cv2.resize(y_arg_unrotated, (w, h))

        # Store snapshot for visualization
        rotation_snapshots[angle] = y_arg_unrotated

        # Step 5: Accumulate results from all rotations
        d_arg_accumulator += y_arg_unrotated

    # Step 6: Square the accumulated result to enhance features
    # Squaring makes positive and negative responses both positive and amplifies
    # strong features
    d_arg_squared = d_arg_accumulator**2

    # Visualize pipeline if requested
    if visualize_matplotlib:
        visualize_darg_detailed(
            gray_image,
            blurred,
            g_gx,
            g_gy,
            g_theta,
            rotation_snapshots,
            d_arg_accumulator,
            d_arg_squared,
        )

    if return_intermediates:
        return (
            gray_image,
            blurred,
            g_gx,
            g_gy,
            g_theta,
            rotation_snapshots,
            d_arg_accumulator,
            d_arg_squared,
        )

    return d_arg_squared


# =============================================================================
# SECTION 2: RADIAL SYMMETRY (FAST RADIAL SYMMETRY TRANSFORM)
# =============================================================================


def run_fast_radial_symmetry(gray_image, radii=None, alpha=2.0, blur_ksize=3):
    """Implement the Fast Radial Symmetry Transform (Loy & Zelinsky, 2003).

    This algorithm detects radially symmetric features (e.g., eyes, circular
    patterns) by analyzing how image gradients point toward or away from
    potential centers.

    Algorithm:
    1. Compute image gradients (magnitude and direction)
    2. For each radius r:
       a. Each pixel "votes" for a symmetry center at distance r along its gradient
       b. Positive vote: pixel + r*gradient_direction
       c. Negative vote: pixel - r*gradient_direction
       d. Accumulate orientation votes (O_n) and magnitude weights (M_n)
       e. Compute symmetry: S_n = |O_n|^alpha * M_n
       f. Smooth result with Gaussian blur
    3. Sum symmetry maps across all radii

    Parameters
    ----------
    gray_image : ndarray
        Input grayscale image (uint8).
    radii : list of int, default=[10, 20, 30, 40]
        List of radii to search for symmetric features.
    alpha : float, default=2.0
        Strictness parameter (higher = more selective).
        - alpha=1: Linear response
        - alpha=2: Standard (good balance)
        - alpha>2: Very strict (only strong symmetry)
    blur_ksize : int, default=3
        Kernel size for initial Gaussian blur.

    Returns
    -------
    ndarray
        Normalized symmetry map (uint8, 0-255).
        Bright regions indicate radially symmetric features.
    """
    if radii is None:
        radii = [10, 20, 30, 40]

    # Step 1: Compute image gradients
    # Blur first to reduce noise sensitivity
    blurred = cv2.GaussianBlur(gray_image, (blur_ksize, blur_ksize), 0)

    # Compute gradients using Sobel operator
    g_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    g_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)

    # Gradient magnitude = strength of edge
    mag = np.sqrt(g_x**2 + g_y**2)
    mag[mag == 0] = 1e-5  # Avoid division by zero

    # Normalized gradient direction (unit vectors)
    g_x_norm = g_x / mag
    g_y_norm = g_y / mag

    # Initialize total symmetry accumulator
    rows, cols = gray_image.shape
    s_total = np.zeros((rows, cols), dtype=np.float64)

    # Create coordinate grids for vectorized operations
    y_grid, x_grid = np.indices((rows, cols))

    # Step 2: Process each radius
    for r in radii:
        # Initialize accumulators for this radius
        o_n = np.zeros((rows, cols), dtype=np.float64)  # Orientation projection
        m_n = np.zeros((rows, cols), dtype=np.float64)  # Magnitude projection

        # Calculate pixel shifts based on gradient direction and current radius
        # Each pixel votes for a symmetry center at distance r along its gradient
        shift_x = np.round(r * g_x_norm).astype(int)
        shift_y = np.round(r * g_y_norm).astype(int)

        # Determine positive and negative voting locations
        # Positive: gradient points TOWARD this location (bright center)
        pos_x = np.clip(x_grid + shift_x, 0, cols - 1)
        pos_y = np.clip(y_grid + shift_y, 0, rows - 1)

        # Negative: gradient points AWAY from this location (dark center)
        neg_x = np.clip(x_grid - shift_x, 0, cols - 1)
        neg_y = np.clip(y_grid - shift_y, 0, rows - 1)

        # Accumulate votes using fast vectorized operations
        # np.add.at handles multiple votes to the same location correctly
        np.add.at(o_n, (pos_y, pos_x), 1)  # +1 vote for orientation
        np.add.at(o_n, (neg_y, neg_x), -1)  # -1 vote (opposite direction)
        np.add.at(m_n, (pos_y, pos_x), mag)  # Weight by gradient magnitude
        np.add.at(m_n, (neg_y, neg_x), -mag)

        # Step 3: Compute symmetry measure for this radius
        # Clip orientation to prevent extreme values from causing instability
        o_n = np.clip(o_n, -100, 100)

        # Symmetry formula: S = |O_n|^alpha * M_n
        # - |O_n|^alpha: Orientation consistency (high when gradients align radially)
        # - M_n: Magnitude weighting (strong edges contribute more)
        s_n = (np.abs(o_n) ** alpha) * m_n

        # Step 4: Smooth the symmetry map
        # Gaussian blur merges nearby votes into coherent blobs
        # Kernel size scales with radius for appropriate spatial integration
        ksize = int(r) | 1  # Ensure odd kernel size (bitwise OR with 1)
        s_n = cv2.GaussianBlur(s_n, (ksize, ksize), r * 0.5)

        # Accumulate symmetry across all radii
        s_total += s_n

    # Step 5: Final normalization
    # Take absolute value to capture both bright and dark symmetric centers
    s_abs = np.abs(s_total)

    # Normalize to 0-255 range for visualization
    s_norm = cv2.normalize(s_abs, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    return s_norm


# =============================================================================
# SECTION 3: TRADITIONAL EDGE DETECTORS
# =============================================================================


def get_other_detectors(gray_image, blur_ksize=3, canny_low=50, canny_high=150):
    """Apply traditional edge detection algorithms for comparison.

    Implements five classic edge detection methods:
    1. Canny: Multi-stage algorithm with hysteresis thresholding
    2. Sobel: First-order derivative operator
    3. Prewitt: Similar to Sobel with slightly different weights
    4. Roberts: 2x2 cross-gradient operator (fastest, noisiest)
    5. LoG: Laplacian of Gaussian (second-order derivative)

    Parameters
    ----------
    gray_image : ndarray
        Input grayscale image (uint8).
    blur_ksize : int, default=3
        Kernel size for Gaussian blur preprocessing.
    canny_low : int, default=50
        Lower threshold for Canny edge detection.
    canny_high : int, default=150
        Upper threshold for Canny edge detection.

    Returns
    -------
    canny : ndarray
        Canny edge detection result (uint8).
    sobel : ndarray
        Sobel edge detection result (uint8).
    prewitt : ndarray
        Prewitt edge detection result (uint8).
    roberts : ndarray
        Roberts edge detection result (uint8).
    log_img : ndarray
        Laplacian of Gaussian result (uint8).

    Notes
    -----
    All outputs are in uint8 format (0-255) for visualization.
    """
    # Preprocessing: blur to reduce noise
    blurred = cv2.GaussianBlur(gray_image, (blur_ksize, blur_ksize), 0)

    # 1. Canny Edge Detector
    # Multi-stage algorithm: gradient -> non-max suppression -> hysteresis
    canny = cv2.Canny(blurred, canny_low, canny_high)

    # 2. Sobel Edge Detector
    # Computes gradient magnitude using 3x3 Sobel kernels
    sx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)  # ∂I/∂x
    sy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)  # ∂I/∂y
    sobel = np.sqrt(sx**2 + sy**2)  # Magnitude

    # 3. Prewitt Edge Detector
    # Similar to Sobel but with uniform weights
    kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])  # Horizontal edges
    kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])  # Vertical edges
    px = cv2.filter2D(blurred, cv2.CV_64F, kernelx)
    py = cv2.filter2D(blurred, cv2.CV_64F, kernely)
    prewitt = np.sqrt(px**2 + py**2)

    # 4. Roberts Cross Operator
    # Simplest gradient operator (2x2 diagonal kernels)
    # Fast but very noise-sensitive
    roberts_x = np.array([[1, 0], [0, -1]])  # Diagonal gradient
    roberts_y = np.array([[0, 1], [-1, 0]])  # Anti-diagonal gradient
    rx = cv2.filter2D(blurred, cv2.CV_64F, roberts_x)
    ry = cv2.filter2D(blurred, cv2.CV_64F, roberts_y)
    roberts = np.sqrt(rx**2 + ry**2)

    # 5. Laplacian of Gaussian (LoG)
    # Second-order derivative -> detects zero-crossings
    # Good for finding blobs and fine details
    log_img = cv2.Laplacian(blurred, cv2.CV_64F)
    log_img = np.abs(log_img)  # Take absolute value to see all edges

    return canny, sobel, prewitt, roberts, log_img


# =============================================================================
# MAIN COMPARISON FUNCTION
# =============================================================================


def compare_all_algorithms(
    image_path,
    d_arg_params=None,
    radial_params=None,
    edge_params=None,
    return_images=False,
):
    """Run all algorithms on a single image and display comparative results.

    This function orchestrates the entire comparison pipeline:
    1. Loads and preprocesses image
    2. Runs D_arg (with detailed visualization)
    3. Runs Radial Symmetry
    4. Runs traditional edge detectors
    5. Displays all results in a grid for visual comparison

    Parameters
    ----------
    image_path : str
        Full path to input image file.
    d_arg_params : dict, optional
        Parameters for D_arg algorithm. Default: {"blur_ksize": 101, "y_arg_ksize": 17}.
    radial_params : dict, optional
        Parameters for radial symmetry. Default: {"radii": [10, 20, 30, 40], "alpha": 2.0}.
    edge_params : dict, optional
        Parameters for edge detectors. Default: {"blur_ksize": 3, "canny_low": 50,
        "canny_high": 150}.
    """  # noqa: E501
    # Set defaults if None
    if d_arg_params is None:
        d_arg_params = {"blur_ksize": 101, "y_arg_ksize": 17}
    if radial_params is None:
        radial_params = {"radii": [10, 20, 30, 40], "alpha": 2.0}
    if edge_params is None:
        edge_params = {"blur_ksize": 3, "canny_low": 50, "canny_high": 150}

    if not os.path.exists(image_path):
        print(f"ERROR: Image not found: {os.path.basename(image_path)}")
        return None
    else:
        print(f"Processing: {os.path.basename(image_path)}...")

    # Load image
    orig = cv2.imread(image_path)
    if orig is None:
        print(f"ERROR: Could not load image: {image_path}")
        return

    # Convert to RGB for display (OpenCV loads as BGR)
    orig_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)

    # Convert to grayscale for processing
    gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)

    # -------------------------------------------------------------------------
    # 1. D_arg (Convexity Detection) - Proposed Method
    # -------------------------------------------------------------------------
    print("- Running D_arg Pipeline...")
    d_arg_res = run_d_arg_pipeline(gray, visualize_matplotlib=True, **d_arg_params)
    d_arg_norm = cv2.normalize(
        d_arg_res, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )

    # -------------------------------------------------------------------------
    # 2. Radial Symmetry Transform
    # -------------------------------------------------------------------------
    print("- Running Radial Symmetry...")
    radial_res = run_fast_radial_symmetry(gray, **radial_params)

    # Use robust normalization to handle hotspots without introducing noise
    radial_norm = robust_normalize(radial_res)

    # -------------------------------------------------------------------------
    # 3. Traditional Edge Detectors
    # -------------------------------------------------------------------------
    print("- Running Standard Detectors...")
    canny, sobel, prewitt, roberts, log_res = get_other_detectors(gray, **edge_params)

    # Normalize continuous-valued detectors to 0-255 range
    sobel = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    prewitt = cv2.normalize(prewitt, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    roberts = cv2.normalize(roberts, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    log_res = cv2.normalize(log_res, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    if return_images:
        return d_arg_norm, radial_norm, canny, sobel, prewitt, roberts, log_res

    # -------------------------------------------------------------------------
    # Display Comparison Grid
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.canvas.manager.set_window_title("Algorithm Comparison")
    ax = axes.ravel()

    # Row 1: Original + Advanced Methods
    ax[0].imshow(orig_rgb)
    ax[0].set_title("Original Image", fontweight="bold")

    ax[1].imshow(d_arg_norm, cmap="gray")
    ax[1].set_title(
        "D_arg (Convexity)\n(Proposed Method)", fontweight="bold", color="blue"
    )

    ax[2].imshow(radial_norm, cmap="gray")
    ax[2].set_title("Radial Symmetry")

    # Empty slot
    ax[3].axis("off")

    # Row 2: Traditional Edge Detectors
    ax[4].imshow(canny, cmap="gray")
    ax[4].set_title("Canny")

    ax[5].imshow(sobel, cmap="gray")
    ax[5].set_title("Sobel")

    ax[6].imshow(prewitt, cmap="gray")
    ax[6].set_title("Prewitt")

    ax[7].imshow(log_res, cmap="gray")
    ax[7].set_title("LoG")

    # Remove axes from all subplots for cleaner visualization
    for a in ax:
        a.axis("off")

    plt.tight_layout()
    print("Displaying results...")
    plt.show()


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


def main():
    """Execute main function.

    Defines test images and runs comparison pipeline on each.
    Add more images to the test_images list to process multiple files.
    """
    # List of test images to process
    test_images = [
        os.path.join(ROOT_DATA_PATH, "Bear", "images - 2020-07-02T154335.549.jpg"),
        # Uncomment to process additional images:
        # os.path.join(ROOT_DATA_PATH, "Bear", "camourflage_00072.jpg")
    ]

    # Process each test image
    for img_path in test_images:
        # Verify file exists before processing
        if not os.path.exists(img_path):
            print(f"ERROR: Test image not found: {img_path}")
            return

        # Run full comparison pipeline
        compare_all_algorithms(img_path)


# Standard Python idiom: only run if script is executed directly
if __name__ == "__main__":
    main()
