# -*- coding: utf-8 -*-
"""Graphical User Interface for Camouflage Breaking Algorithm Suite.

This module provides a Tkinter-based GUI for interacting with camouflage
detection algorithms, allowing users to:
- Browse and select test images from categorized datasets
- Adjust algorithm parameters in real-time
- Visualize pipeline steps and compare different detection methods
- View preconfigured examples with optimal parameters

The GUI is organized into left control panel and right results panel with
scrollable content display.
"""
import os
import tkinter as tk
from tkinter import messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

import main


class Window:
    """Main application window for Camouflage Breaking GUI.

    This class creates and manages the primary application interface with
    a split-pane layout: left panel for controls and image selection,
    right panel for displaying results.

    Attributes
    ----------
    root : tk.Tk
        Main Tkinter window instance.
    left_frame : tk.Frame
        Container for control widgets (left side of window).
    right_frame : tk.Frame
        Container for results display (right side of window).
    left_left_frame : tk.Frame
        Sub-panel for image selection controls.
    left_right_frame : tk.Frame
        Sub-panel for parameters and action buttons.
    canvas : tk.Canvas
        Scrollable canvas for displaying results.
    results_frame : tk.Frame
        Frame inside canvas where result images are placed.
    base_path : str
        Root directory path for image datasets.
    categories : list of str
        Available image categories (subdirectories).
    category_var : tk.StringVar
        Currently selected category name.
    image_list : tk.Listbox
        Widget displaying images in selected category.
    selected_label : tk.Label
        Label showing currently selected image filename.
    image_canvas : tk.Label
        Widget displaying preview of selected input image.
    blur_ksize : tk.IntVar
        Gaussian blur kernel size parameter (default: 101).
    gradient_ksize : tk.IntVar
        Gradient computation kernel size parameter (default: 3).
    y_arg_ksize : tk.IntVar
        Y-arg derivative kernel size parameter (default: 17).
    current_image : ImageTk.PhotoImage
        Currently displayed preview image reference.

    Methods
    -------
    create_left_panel()
        Build left control panel with widgets.
    create_right_panel()
        Build right results panel with scrolling.
    get_selected_image_path()
        Get full path to currently selected image.
    load_images(category)
        Populate image list for selected category.
    image_selected(event)
        Handle image selection from listbox.
    display_input_image(img_path)
        Show preview of selected input image.
    run_pipeline()
        Execute D_arg pipeline and display all steps.
    compare_algorithms()
        Run all algorithms and display comparison.
    display_only_result()
        Show only final D_arg result (large format).
    display_preconfigured_examples()
        Display results for all preconfigured test cases.
    resize_for_display(img_array, max_w, max_h)
        Resize image to fit display constraints.
    add_to_grid(title, img_array, row, col, max_w, max_h)
        Add labeled image to results grid.
    put(title, img, row, col, cols, max_w, max_h)
        Place image in grid and return next position.
    _on_mousewheel(event)
        Handle mouse wheel scrolling in results panel.

    Examples
    --------
    >>> app = Window()
    >>> app.root.mainloop()
    """

    def __init__(self):
        self.root = tk.Tk()
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        # Set window size to 90% of screen to ensure visibility
        w = int(screen_width * 0.9)
        h = int(screen_height * 0.9)
        self.root.geometry(f"{w}x{h}+0+0")

        self.root.resizable(False, False)  # no resizing
        self.root.title("Camouflage Breaking")

        # -----------------------------
        # WINDOW SPLITTING INTO LEFT AND RIGHT PANELS
        # -----------------------------
        self.left_frame = tk.Frame(self.root, bg="lightgray", width=screen_width // 3)

        self.left_frame.pack(side="left", fill="y")
        self.left_frame.pack_propagate(
            False
        )  # disable the panel extension : sets to fix width

        self.right_frame = tk.Frame(self.root, bg="white")
        self.right_frame.pack(side="left", fill="both", expand=True)
        self.create_right_panel()
        self.create_left_panel()

    # -----------------------------
    # WIDGETS HANDLING FOR LEFT PANEL
    # -----------------------------
    def create_left_panel(self):
        """Create and populate the left control panel with all widgets.

        Builds the following UI components:
        - Image category dropdown menu
        - Image selection listbox
        - Input image preview canvas
        - Algorithm parameter controls (spinboxes)
        - Action buttons (run pipeline, compare, etc.)

        The panel is split into two sub-panels (left_left and left_right)
        for better organization of controls.

        Notes
        -----
        If the data directory doesn't exist or contains no categories,
        displays an error message instead of controls.
        """
        # -----------------------------
        # SPLIT THE LEFT PANEL INTO TWO PANELS
        # -----------------------------
        self.left_left_frame = tk.Frame(self.left_frame, bg="lightgray")
        self.left_left_frame.pack(
            side="left", fill="both", expand=True, padx=10, pady=10
        )

        self.left_right_frame = tk.Frame(self.left_frame, bg="lightgray")
        self.left_right_frame.pack(
            side="left", fill="both", expand=True, padx=10, pady=10
        )

        # -----------------------------
        # DROP DOWN LIST WITH IMAGES FROM DIFFERENT CATEGORIES
        # -----------------------------

        # Data directory path :
        self.base_path = main.ROOT_DATA_PATH

        # Check path exists
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path, exist_ok=True)

        # Categories (folders) :
        self.categories = [
            d
            for d in os.listdir(self.base_path)
            if os.path.isdir(os.path.join(self.base_path, d))
        ]

        if not self.categories:
            tk.Label(self.left_left_frame, text="No Data Found", fg="red").pack()
            return

        # Categories label :
        category_label = tk.Label(
            self.left_left_frame,
            text="Camouflaged animals categories :",
            font=("Arial", 10, "bold"),
            fg="white",
            bg="brown",
            width=30,
            height=1,
        )
        category_label.pack(pady=5)

        # Drop-down list of categories with event :
        self.category_var = (
            tk.StringVar()
        )  # string variable : option selected in the menu
        self.category_var.set(self.categories[0])  # default value
        category_menu = ttk.OptionMenu(
            self.left_left_frame,
            self.category_var,
            self.categories[0],
            *self.categories,
            command=self.load_images,
        )
        category_menu.pack()

        # List of images for each category :
        self.image_list = tk.Listbox(self.left_left_frame, width=40, height=25)
        self.image_list.pack(pady=10, fill="x")

        # Label displaying the current selection :
        self.selected_label = tk.Label(
            self.left_left_frame,
            text="No image selected",
            font=("Arial", 8, "bold"),
            bg="lightgrey",
            wraplength=150,
        )
        self.selected_label.pack(pady=10)

        # -----------------------------
        # RUN BUTTON FOR THE LEFT SIDE OF THE LEFT PANEL
        # -----------------------------
        self.run_examples = tk.Button(
            self.left_left_frame,
            text="DISPLAY PRECONFIGURED EXAMPLES",
            command=self.display_preconfigured_examples,
            font=("Arial", 10, "bold"),
            fg="white",
            bg="purple",
        )
        self.run_examples.pack(pady=10, fill="x")

        # -----------------------------
        # CANVAS TO DISPLAY SELECTED IMAGE
        # -----------------------------
        self.image_list.bind("<<ListboxSelect>>", self.image_selected)
        self.preview_label = tk.Label(
            self.left_right_frame,
            text="Selected input image:",
            font=("Arial", 8, "bold"),
            bg="brown",
            fg="white",
        )
        self.preview_label.pack(pady=5, fill="x")

        self.image_canvas = tk.Label(self.left_right_frame, bg="white", height=200)

        self.image_canvas.pack(pady=5, fill="x")

        self.load_images(self.categories[0])  # default category displayed
        # -----------------------------
        # USER PARAMETERS FOR THE DARG ALGORITHM
        # -----------------------------
        params_title = tk.Label(
            self.left_right_frame,
            text="Pipeline parameters:",
            font=("Arial", 8, "bold"),
            bg="brown",
            fg="white",
        )
        params_title.pack(pady=5, fill="x")

        # Gaussian blur kernel size :
        tk.Label(
            self.left_right_frame,
            text="Gaussian Blur kernel:",
            bg="lightgray",
            font=("Arial", 8),
        ).pack(anchor="w")
        self.blur_ksize = tk.IntVar(value=101)  # Default from main.py
        tk.Spinbox(
            self.left_right_frame,
            from_=3,
            to=201,
            increment=2,
            textvariable=self.blur_ksize,
            width=10,
        ).pack(anchor="w", padx=5)

        # Gradient kernel size :
        tk.Label(
            self.left_right_frame,
            text="Gradient kernel:",
            bg="lightgray",
            font=("Arial", 8),
        ).pack(anchor="w")
        self.gradient_ksize = tk.IntVar(value=3)
        tk.Spinbox(
            self.left_right_frame,
            from_=3,
            to=31,
            increment=2,
            textvariable=self.gradient_ksize,
            width=10,
        ).pack(anchor="w", padx=5)

        # Y-derivative kernel size :
        tk.Label(
            self.left_right_frame,
            text="Y-derivative kernel:",
            bg="lightgray",
            font=("Arial", 8),
        ).pack(anchor="w")
        self.y_arg_ksize = tk.IntVar(value=17)  # Default from main.py
        tk.Spinbox(
            self.left_right_frame,
            from_=3,
            to=31,
            increment=2,
            textvariable=self.y_arg_ksize,
            width=10,
        ).pack(anchor="w", padx=5)

        # -----------------------------
        # RUN BUTTONs FOR THE RIGHT SIDE OF THE LEFT PANEL
        # -----------------------------
        self.run_pipeline_button = tk.Button(
            self.left_right_frame,
            text="DISPLAY DARG PIPELINE STEPS",
            command=self.run_pipeline,
            font=("Arial", 10, "bold"),
            fg="white",
            bg="blue",
        )
        self.run_pipeline_button.pack(pady=2, fill="x")

        self.display_only_result_button = tk.Button(
            self.left_right_frame,
            text="DISPLAY ONLY DARG RESULT",
            command=self.display_only_result,
            font=("Arial", 10, "bold"),
            fg="white",
            bg="red",
        )
        self.display_only_result_button.pack(pady=2, fill="x")

        self.compare_results_button = tk.Button(
            self.left_right_frame,
            text="COMPARE ALGORITHMS",
            command=self.compare_algorithms,
            font=("Arial", 8, "bold"),
            fg="white",
            bg="green",
        )
        self.compare_results_button.pack(pady=2, fill="x")

    # -----------------------------
    # WIDGETS HANDLING FOR RIGHT PANEL
    # -----------------------------
    def create_right_panel(self):
        """Create the right results panel with scrolling capability.

        Builds a scrollable canvas with vertical scrollbar for displaying
        algorithm results. The canvas contains a results_frame where all
        output images and visualizations are placed.

        The panel supports mouse wheel scrolling for easy navigation through
        multiple results.

        Notes
        -----
        The scrollregion is automatically updated when content changes via
        the Configure event binding.
        """
        # Scroll canvas
        self.canvas = tk.Canvas(self.right_frame, bg="white")
        self.canvas.pack(side="left", fill="both", expand=True)

        # Vertical scrollbar
        self.scrollbar = tk.Scrollbar(
            self.right_frame, orient="vertical", command=self.canvas.yview
        )
        self.scrollbar.pack(side="right", fill="y")

        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        # create frame to display results
        self.results_frame = tk.Frame(self.canvas, bg="white")
        # anchor="nw" ensures content starts at top-left
        self.canvas.create_window((0, 0), window=self.results_frame, anchor="nw")

        # Bind MouseWheel for scrolling
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

        # Scrollbar updating
        self.results_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )

    def _on_mousewheel(self, event):
        """Handle mouse wheel scrolling events in the results panel.

        Parameters
        ----------
        event : tk.Event
            Tkinter event object containing mouse wheel delta value.

        Notes
        -----
        The scroll amount is calculated as -1 * (delta / 120) to provide
        natural scrolling direction (scroll down = content moves up).
        """
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    # -----------------------------
    # EVENT HANDLERS AND FUNCTIONS
    # -----------------------------
    def get_selected_image_path(self):
        """Get the complete file path of the currently selected image.

        Returns
        -------
        str or None
            Full path to selected image file, or None if no image is selected.

        Notes
        -----
        Constructs path from base_path, current category, and filename
        extracted from selected_label text.
        """
        category = self.category_var.get()
        filename = self.selected_label.cget("text").replace("Image selected : ", "")
        if filename == "No image selected":
            return None
        return os.path.join(self.base_path, category, filename)

    # event handler when a category is selected in the menu :
    def load_images(self, category):
        """Load and display images from the specified category directory.

        Populates the image listbox with all image files found in the
        selected category subdirectory, sorted alphabetically.

        Parameters
        ----------
        category : str
            Name of the category subdirectory to load images from.

        Notes
        -----
        Only files with extensions .jpg, .jpeg, .png, or .bmp are included.
        Resets the selected image label to "No image selected".
        """
        category_path = os.path.join(self.base_path, category)
        self.image_list.delete(0, tk.END)

        if os.path.exists(category_path):
            for f in sorted(os.listdir(category_path)):
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    self.image_list.insert(tk.END, f)

        self.selected_label.config(text="No image selected")

    # event handler when an image is selected in the list box:
    def image_selected(self, event):
        """Handle image selection event from the listbox.

        Updates the selected image label and displays a preview of the
        chosen image in the canvas.

        Parameters
        ----------
        event : tk.Event
            Tkinter listbox selection event (not used directly).

        Notes
        -----
        Safely handles cases where no selection exists or selection is invalid.
        """
        try:
            if not self.image_list.curselection():
                return
            index = self.image_list.curselection()[0]
        except (IndexError, tk.TclError):
            return

        filename = self.image_list.get(index)
        self.selected_label.config(text=f"Image selected : {filename}")

        img_path = self.get_selected_image_path()
        if img_path:
            self.display_input_image(img_path)

    def display_input_image(self, img_path):
        """Display a preview of the selected input image.

        Loads the image from disk, resizes it to fit the preview canvas
        while maintaining aspect ratio, and displays it.

        Parameters
        ----------
        img_path : str
            Full path to the image file to display.

        Notes
        -----
        - Maximum preview size: 350x250 pixels
        - Image is converted from BGR (OpenCV) to RGB for proper display
        - If image cannot be loaded, function returns silently
        """
        img = cv2.imread(img_path)
        if img is None:
            return
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # image width and heigt adaptation to the canvas :
        max_w, max_h = 350, 250
        h, w = img.shape[:2]
        scale = min(max_w / w, max_h / h)
        resized_w = int(w * scale)
        resized_h = int(h * scale)
        img = cv2.resize(
            img, (resized_w, resized_h), interpolation=cv2.INTER_AREA
        )  # adjust size to fit left frame

        img = Image.fromarray(img)  # format conversion from cv2 to PIL
        tk_img = ImageTk.PhotoImage(img)

        self.current_image = tk_img
        self.image_canvas.config(image=tk_img)  # input image diplaying
        self.image_canvas.image = tk_img  # attached to the canvas widget

    # event handler, run pipeline button pressed
    def run_pipeline(self):
        """Execute the complete D_arg pipeline and visualize all steps.

        Runs the D_arg algorithm on the selected image using current parameter
        settings, then displays all intermediate processing steps in a grid:
        - Original grayscale image
        - Blurred image
        - Gradient components (x, y)
        - Gradient orientation (theta)
        - Rotational processing results (0°, 90°, 180°, 270°)
        - Accumulated result
        - Final squared result

        Raises
        ------
        Shows warning dialog if no image is selected.
        Shows error dialog if image file cannot be loaded.

        Notes
        -----
        Clears previous results before displaying new ones.
        All images displayed at 200x200 pixels in a 2-column grid.
        """
        # Get image selected :
        img_path = self.get_selected_image_path()
        if img_path is None:
            messagebox.showwarning("No Selection", "Please select an image first")
            return

        img = cv2.imread(img_path)
        if img is None:
            messagebox.showerror("Error", f"Could not load image: {img_path}")
            return

        # Convert to RGB for display (OpenCV loads as BGR) :
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Convert to grayscale for processing :
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

        # Run Darg pipeline :
        (
            gray_image,
            blurred,
            g_gx,
            g_gy,
            g_theta,
            rotation_snapshots,
            d_arg_accumulator,
            d_arg_squared,
        ) = main.run_d_arg_pipeline(
            gray,
            blur_ksize=self.blur_ksize.get(),
            gradient_ksize=self.gradient_ksize.get(),
            y_arg_ksize=self.y_arg_ksize.get(),
            return_intermediates=True,
        )

        # Clear right panel :
        for w in self.results_frame.winfo_children():
            w.destroy()

        # Grid setting to display the steps of the pipeline :
        COLS = 2
        SIZE = 200
        row = 0
        col = 0

        # Plots placement on the grid :
        row, col = self.put("Gray Image", gray_image, row, col, COLS, SIZE, SIZE)
        row, col = self.put("Blurred", blurred, row, col, COLS, SIZE, SIZE)
        row, col = self.put("Gradient x", g_gx, row, col, COLS, SIZE, SIZE)
        row, col = self.put("Gradient y", g_gy, row, col, COLS, SIZE, SIZE)
        row, col = self.put("Theta", g_theta, row, col, COLS, SIZE, SIZE)

        for angle in sorted(rotation_snapshots.keys()):
            row, col = self.put(
                f"Rotation {angle}°",
                rotation_snapshots[angle],
                row,
                col,
                COLS,
                SIZE,
                SIZE,
            )

        row, col = self.put("Darg", d_arg_accumulator, row, col, COLS, SIZE, SIZE)
        row, col = self.put("Darg²", d_arg_squared, row, col, COLS, SIZE, SIZE)

    # event handler, compare detectors button pressed
    def compare_algorithms(self):
        """Run all camouflage detection algorithms and display comparison.

        Executes the following algorithms on the selected image:
        - D_arg (convexity-based detection)
        - Fast Radial Symmetry Transform
        - Traditional edge detectors: Canny, Sobel, Prewitt, Roberts, LoG

        Results are displayed in a vertical stack (1 column) for easy
        side-by-side visual comparison.

        Raises
        ------
        Shows warning dialog if no image is selected.
        Shows error dialog if processing fails.

        Notes
        -----
        Uses current parameter settings for D_arg algorithm.
        All results normalized to 350x350 pixels for consistent comparison.
        """
        # Get image selected :
        img_path = self.get_selected_image_path()
        if img_path is None:
            messagebox.showwarning("No Selection", "Please select an image first")
            return

        d_arg_params = {
            "blur_ksize": self.blur_ksize.get(),
            "gradient_ksize": self.gradient_ksize.get(),
            "y_arg_ksize": self.y_arg_ksize.get(),
        }

        result = main.compare_all_algorithms(img_path, d_arg_params, return_images=True)

        if result is None:
            messagebox.showerror("Error", "Failed to process image")
            return

        d_arg_norm, radial_norm, canny, sobel, prewitt, roberts, log_res = result

        # Clears right panel
        for w in self.results_frame.winfo_children():
            w.destroy()

        # Display all algorithm results
        results = {
            "Darg": d_arg_norm,
            "Radial": radial_norm,
            "Canny": canny,
            "Sobel": sobel,
            "Prewitt": prewitt,
            "Roberts": roberts,
            "LoG": log_res,
        }
        COLS = 1
        row = 0
        col = 0
        for name, result in results.items():
            row, col = self.put(name, result, row, col, COLS, 350, 350)

    def resize_for_display(self, img_array, max_w=180, max_h=130):
        """Resize image array to fit within specified dimensions.

        Maintains aspect ratio while ensuring the image fits within the
        maximum width and height constraints.

        Parameters
        ----------
        img_array : np.ndarray
            Input image array (grayscale or RGB).
        max_w : int, default=180
            Maximum width in pixels.
        max_h : int, default=130
            Maximum height in pixels.

        Returns
        -------
        np.ndarray
            Resized image array with preserved aspect ratio.

        Notes
        -----
        Uses INTER_AREA interpolation for high-quality downsampling.
        """
        h, w = img_array.shape[:2]
        scale = min(max_w / w, max_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(img_array, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized

    def add_to_grid(self, title, img_array, row, col, max_w=180, max_h=150):
        """Add a labeled image to the results grid at specified position.

        Creates a frame containing a title label and the image, positioned
        at the given row and column in the results grid.

        Parameters
        ----------
        title : str
            Text label to display above the image.
        img_array : np.ndarray
            Image array to display (any dtype, will be normalized to uint8).
        row : int
            Grid row position (0-indexed).
        col : int
            Grid column position (0-indexed).
        max_w : int, default=180
            Maximum image width in pixels.
        max_h : int, default=150
            Maximum image height in pixels.

        Notes
        -----
        - Float arrays are automatically normalized to 0-255 uint8 range
        - Images maintain aspect ratio when resized
        - Each image is placed in a white-background frame with 10px padding
        """
        block = tk.Frame(self.results_frame, bg="white")
        block.grid(row=row, column=col, padx=10, pady=10)

        tk.Label(block, text=title, bg="white", font=("Arial", 10, "bold")).pack()

        img_array = self.resize_for_display(img_array, max_w, max_h)

        if img_array.dtype != np.uint8:
            img_array = cv2.normalize(
                img_array, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
            )

        img = Image.fromarray(img_array)
        tk_img = ImageTk.PhotoImage(img)

        lbl = tk.Label(block, image=tk_img, bg="white")
        lbl.image = tk_img
        lbl.pack()

    def put(self, title, img, row, col, cols, max_w=180, max_h=150):
        """Place image in grid and calculate next grid position.

        Convenience method that adds an image to the grid and automatically
        computes the next available grid position for subsequent images.

        Parameters
        ----------
        title : str
            Text label for the image.
        img : np.ndarray
            Image array to display.
        row : int
            Current grid row position.
        col : int
            Current grid column position.
        cols : int
            Total number of columns in the grid layout.
        max_w : int, default=180
            Maximum image width in pixels.
        max_h : int, default=150
            Maximum image height in pixels.

        Returns
        -------
        tuple of (int, int)
            New (row, col) position for next image placement.

        Notes
        -----
        Grid fills left-to-right, then wraps to next row when column limit
        is reached.

        Examples
        --------
        >>> row, col = 0, 0
        >>> row, col = self.put("Image 1", img1, row, col, cols=3)
        >>> row, col = self.put("Image 2", img2, row, col, cols=3)
        """
        self.add_to_grid(title, img, row, col, max_w, max_h)

        # Grid incrementation
        col += 1
        if col == cols:
            col = 0
            row += 1

        return row, col

    def display_only_result(self):
        """Display only the final D_arg result in large format.

        Runs the D_arg pipeline on the selected image and displays only the
        final squared result (d_arg²) in a large 500x500 pixel format,
        without showing intermediate processing steps.

        Notes
        -----
        Useful for quick evaluation of algorithm performance on a single image.
        Clears previous results before displaying new result.
        Returns silently if no image is selected.
        """
        # Get image selected :
        img_path = self.get_selected_image_path()
        if not img_path:
            return

        img = cv2.imread(img_path)

        # Convert to RGB for display (OpenCV loads as BGR) :
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Convert to grayscale for processing
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

        intermediates = main.run_d_arg_pipeline(
            gray,
            blur_ksize=self.blur_ksize.get(),
            gradient_ksize=self.gradient_ksize.get(),
            y_arg_ksize=self.y_arg_ksize.get(),
            return_intermediates=True,
        )
        d_arg_squared = intermediates[-1]  # Extract result

        for w in self.results_frame.winfo_children():
            w.destroy()

        # Display larger
        self.put("Final D_arg Result", d_arg_squared, 0, 0, 1, 500, 500)

    def display_preconfigured_examples(self):
        """Display results for all preconfigured test images.

        Processes and displays results for up to 11 preconfigured test cases
        from main.settings, showing:
        - Original input image
        - D_arg result
        - Parameter settings used

        Each example is displayed in a 3-column layout:
        Column 0: Input image (200x200)
        Column 1: Result image (200x200)
        Column 2: Parameter text description

        Notes
        -----
        Uses optimal parameter settings stored in main.settings for each
        test image. Skips images that cannot be loaded.
        Useful for demonstrating algorithm performance across diverse cases.
        """
        # Clears right panel :
        for w in self.results_frame.winfo_children():
            w.destroy()

        # Grid settings :
        row = 0

        # Extract images and preconfigured params from main.settings :
        items = list(main.settings.items())[:11]

        # Get pipeline params for each image :
        for name, config in items:
            img_path = config["path"]
            blur = config["gaussianblur"]
            grad = config["gradient_ksize"]
            yarg = config["y_derivative_ksize"]

            # Load image
            img = cv2.imread(img_path)
            if img is None:
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

            # Run pipeline
            intermediates = main.run_d_arg_pipeline(
                gray,
                blur_ksize=blur,
                gradient_ksize=grad,
                y_arg_ksize=yarg,
                return_intermediates=True,
            )
            d_arg_squared = intermediates[-1]  # Last element is d_arg_squared

            # Column 0: Input Image
            self.add_to_grid(f"{name} - INPUT", img_rgb, row, 0, max_w=200, max_h=200)

            # Column 1: Result Image
            self.add_to_grid(
                f"{name} - RESULT", d_arg_squared, row, 1, max_w=200, max_h=200
            )

            # Column 2: Parameters Text
            param_text = (
                f"Parameters for {name}:\n\n"
                f"Gaussian Blur: {blur}\n"
                f"Derivative Gradient Kernel: {grad}\n"
                f"Y-Arg Kernel: {yarg}"
            )

            text_frame = tk.Frame(self.results_frame, bg="white")
            text_frame.grid(row=row, column=2, padx=20, pady=10, sticky="w")

            lbl = tk.Label(
                text_frame,
                text=param_text,
                bg="white",
                font=("Arial", 10),
                justify="left",
            )
            lbl.pack()

            # Move to next row
            row += 1


if __name__ == "__main__":
    app = Window()
    app.root.mainloop()
