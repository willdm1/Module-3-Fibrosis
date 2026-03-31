"""
Module 3 main.py
----------------
# Written by Roy Chen and Will Marschall
# Updated 3/27/2026

What this script does:
1. Reads the selected black-and-white fibrosis images.
2. Matches each image to its lung depth using the provided class CSV file.
3. Counts white and black pixels in each image.
4. Computes percent white pixels (fibrosis proxy).
5. Writes the results to the results folder as a CSV file.
6. Plots measured fibrosis vs. depth.
7. Performs linear and quadratic interpolation at a user-selected depth.
8. Verifies interpolation calculations.
9. Validates the interpolated estimate using a nearby real image from the full dataset.
10. Optionally benchmarks runtime and estimates how long the code would take for many images.

We designed to be called either:
- as a single script:    python main.py

- from our report notebook:     import main
                                main.run_image_analysis()
                                main.run_linear_interpolation()
                                main.run_quadratic_interpolation()
                                main.run_validation()

"""

# This file is intended to be the main entry point for running the image analysis and interpolation tasks.
from __future__ import annotations

# standard library imports
from pathlib import Path
from time import perf_counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import warnings

# third-party imports
try:
    from termcolor import colored
except ModuleNotFoundError:
    def colored(text, *_args, **_kwargs):
        return str(text)

# OpenCV is a critical dependency for image processing. 
# We check for its presence and provide instructions if it's missing.
try:
    import cv2
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Missing dependency: OpenCV. Install it with:\n"
        "  pip install opencv-python\n"
        "If you are in Jupyter, you can run:\n"
        "  import sys\n"
        "  !{sys.executable} -m pip install opencv-python"
    ) from e

# local imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


# =============================================================================
# USER SETTINGS
# =============================================================================

# INTERPOLATION_DEPTH_UM is the depth at which we will estimate the fibrosis proxy using interpolation.
INTERPOLATION_DEPTH_UM = 345.0
THRESHOLD_VALUE = 127
SHOW_PLOTS = True
SAVE_FIGURES = True
RUN_VALIDATION_STEP = True
RUN_BENCHMARK_STEP = True
VALIDATION_METHOD = "linear"   # choose: "linear" or "quadratic"

# Leave this list empty to automatically analyze every image in "chosen images".
# If I wanted to force a specific order, I could list basenames here exactly as they appear.
MANUAL_SELECTED_IMAGE_BASENAMES: List[str] = []

# If True, the script uses the matrix approach for linear/quadratic
# interpolation in addition to SciPy's built-in interpolation.
RUN_MANUAL_POLYNOMIAL_INTERPOLATION = True


# =============================================================================
# PATHS
# =============================================================================

# Define paths relative to this script's location for better portability.
CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent
CHOSEN_IMAGES_DIR = PROJECT_ROOT / "chosen images"
ALL_IMAGES_DIR = PROJECT_ROOT / "images"
RESULTS_DIR = PROJECT_ROOT / "results"
DEPTHS_CSV_PATH = PROJECT_ROOT / "Filenames and Depths for Students.csv"

# Define specific output paths for results and plots.
RESULTS_CSV_PATH = RESULTS_DIR / "Percent_White_Pixels.csv"
MEASURED_PLOT_PATH = RESULTS_DIR / "Measured_Percent_White_vs_Depth.png"
LINEAR_PLOT_PATH = RESULTS_DIR / "Linear_Interpolation.png"
QUADRATIC_PLOT_PATH = RESULTS_DIR / "Quadratic_Interpolation.png"
VALIDATION_CSV_PATH = RESULTS_DIR / "Interpolation_Validation.csv"
VALIDATION_PLOT_PATH = RESULTS_DIR / "Interpolation_Validation.png"
BENCHMARK_CSV_PATH = RESULTS_DIR / "Runtime_Benchmark.csv"
INTERPOLATION_RESULTS_CSV_PATH = RESULTS_DIR / "Interpolation_Results.csv"

# Supported image file extensions for searching.
SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

# =============================================================================
# BASIC HELPERS
# =============================================================================

# This function ensures that the results directory exists before we attempt to save any files there, which helps prevent errors when writing output.
def ensure_results_dir() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# Here we define a helper function to normalize filenames by extracting the basename, stripping whitespace, and converting to lowercase. 
# This will help us match filenames from the CSV to actual files on disk in a case-insensitive way.
def normalize_filename(name: str) -> str:
    return Path(str(name)).name.strip().lower()


# This function attempts to infer which column in the provided list of columns likely contains filename information by looking for keywords. 
# If it cannot find a clear match, it defaults to the first column if there is at least one column, or raises an error if it cannot make a reasonable inference.
def infer_filename_column(columns: Sequence[str]) -> str:
    lower_map = {col.lower(): col for col in columns}

    for candidate in columns:
        text = candidate.lower()
        if "file" in text or "image" in text or "name" in text:
            return candidate

    if len(columns) >= 1:
        return columns[0]

    raise ValueError("Could not infer a filename column from the depths CSV.")


# This function attempts to infer which column in the provided list of columns likely contains depth information by looking for keywords.
#  If it cannot find a clear match, it defaults to the second column if there are at least two columns, or raises an error if it cannot make a reasonable inference.
def infer_depth_column(columns: Sequence[str]) -> str:
    for candidate in columns:
        text = candidate.lower()
        if "depth" in text:
            return candidate

    if len(columns) >= 2:
        return columns[1]

    raise ValueError("Could not infer a depth column from the depths CSV.")


# This function loads the depth lookup from the provided CSV file, cleans and normalizes the data, and returns a DataFrame with basenames and corresponding depths. 
# It also handles various error cases such as missing files, empty data, and missing columns.
def load_depth_lookup(depth_csv_path: Path = DEPTHS_CSV_PATH) -> pd.DataFrame:
    if not depth_csv_path.exists():
        raise FileNotFoundError(
            f"Could not find the class depths CSV file: {depth_csv_path}"
        )

    df = pd.read_csv(depth_csv_path)
    if df.empty:
        raise ValueError(f"The depths CSV appears to be empty: {depth_csv_path}")

    filename_col = infer_filename_column(df.columns) # type: ignore
    depth_col = infer_depth_column(df.columns) # type: ignore

    cleaned = df.copy()
    cleaned["original_filename_entry"] = cleaned[filename_col].astype(str)
    cleaned["basename"] = cleaned[filename_col].astype(str).map(normalize_filename)
    cleaned["depth_um"] = pd.to_numeric(cleaned[depth_col], errors="coerce")
    cleaned = cleaned.dropna(subset=["basename", "depth_um"])

# After cleaning, we check if there are any valid rows left. 
# If not, we raise an error to alert the user that the CSV file may not contain usable data.
    if cleaned.empty:
        raise ValueError(
            "No valid filename/depth rows were found after cleaning the depths CSV."
        )

    cleaned = cleaned[["basename", "depth_um", "original_filename_entry"]].drop_duplicates(
        subset=["basename"], keep="first"
    )
    return cleaned.reset_index(drop=True)


# Here we define a helper function to search for an image file by its basename in one or more directories, which will be useful for matching the selected images to their corresponding files on disk.
def find_image_path_by_basename(basename: str, search_dirs: Iterable[Path]) -> Optional[Path]:
    normalized = normalize_filename(basename)

    for folder in search_dirs:
        if not folder.exists():
            continue

        direct = folder / basename
        if direct.exists():
            return direct

        for path in folder.rglob("*"):
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES:
                if normalize_filename(path.name) == normalized:
                    return path

    return None


# Now we define a helper function to run one of the exploratory analysis scripts from the "Code" folder, which allows us to keep those scripts organized while still being able to execute them from this main script.
def get_selected_image_paths() -> List[Path]:
 
 # This function retrieves the file paths of the selected images to analyze. It checks for the existence of the chosen images directory, optionally uses a manually specified list of image basenames, and searches for matching image files in the specified directories. It also validates that supported image files are found and raises appropriate errors if not.
    if not CHOSEN_IMAGES_DIR.exists():
        raise FileNotFoundError(
            f"Could not find the chosen images folder: {CHOSEN_IMAGES_DIR}"
        )

# If the user has specified a manual list of image basenames, we will attempt to find those specific images in the provided directories. 
# If any of the specified basenames cannot be found, we raise an error with details on which images were missing.
    if MANUAL_SELECTED_IMAGE_BASENAMES:
        image_paths: List[Path] = []
        for basename in MANUAL_SELECTED_IMAGE_BASENAMES:
            match = find_image_path_by_basename(basename, [CHOSEN_IMAGES_DIR, ALL_IMAGES_DIR])
            if match is None:
                raise FileNotFoundError(
                    f"Could not locate the image named '{basename}' in either\n"
                    f"  {CHOSEN_IMAGES_DIR}\n"
                    f"or\n"
                    f"  {ALL_IMAGES_DIR}"
                )
            image_paths.append(match)
        return image_paths

    image_paths = sorted(
        path for path in CHOSEN_IMAGES_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    )

# If no supported image files are found in the chosen images directory, we raise an error to alert the user that there may be an issue with the directory contents or the file formats.
    if not image_paths:
        raise FileNotFoundError(
            f"No supported image files were found in: {CHOSEN_IMAGES_DIR}"
        )

    return image_paths


# Next, we define a helper function to read an image in grayscale mode and return it as a NumPy array, which will be used for our pixel counting analysis.
def read_grayscale_image(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image file: {image_path}")
    return image


# This function takes a grayscale image and a threshold value, applies binary thresholding to separate white and black pixels, and then counts the number of white and black pixels, as well as the total number of pixels and the percentage of white pixels.
def count_black_and_white_pixels(
    image: np.ndarray,
    threshold_value: int = THRESHOLD_VALUE,
) -> Dict[str, float]:

# We apply a binary threshold to the image, which converts it to a binary image where pixels above the threshold are set to 255 (white) and those below are set to 0 (black). 
# We then count the number of white pixels using OpenCV's countNonZero function, calculate the total number of pixels, and derive the number of black pixels and the percentage of white pixels.
    _, binary = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)

    white_pixels = int(cv2.countNonZero(binary))
    total_pixels = int(binary.size)
    black_pixels = int(total_pixels - white_pixels)
    percent_white = 100.0 * white_pixels / total_pixels

    return {
        "white_pixels": white_pixels,
        "black_pixels": black_pixels,
        "total_pixels": total_pixels,
        "percent_white_pixels": percent_white,
    }


# This function analyzes a single image by reading it, counting the black and white pixels, and returning a dictionary of results that can be used to build our analysis table.
def analyze_single_image(image_path: Path, depth_um: float) -> Dict[str, object]:
    image = read_grayscale_image(image_path)
    stats = count_black_and_white_pixels(image)

    return {
        "filename": str(image_path),
        "basename": image_path.name,
        "depth_um": float(depth_um),
        "white_pixels": stats["white_pixels"],
        "black_pixels": stats["black_pixels"],
        "total_pixels": stats["total_pixels"],
        "percent_white_pixels": stats["percent_white_pixels"],
    }


# This function builds the analysis table by loading the depth lookup, finding the selected images, analyzing each image, and compiling the results into a DataFrame. It also checks for any missing depths and handles errors appropriately.
def build_analysis_table() -> pd.DataFrame:
    depth_lookup = load_depth_lookup()
    depth_map = dict(zip(depth_lookup["basename"], depth_lookup["depth_um"]))

    image_paths = get_selected_image_paths()
    records: List[Dict[str, object]] = []
    missing_depths: List[str] = []

    for image_path in image_paths:
        basename_key = normalize_filename(image_path.name)
        if basename_key not in depth_map:
            missing_depths.append(image_path.name)
            continue

        record = analyze_single_image(image_path, depth_map[basename_key])
        records.append(record)

    if missing_depths:
        missing_text = "\n".join(f"  - {name}" for name in missing_depths)
        raise KeyError(
            "The following selected images were not found in the depth CSV file:\n"
            f"{missing_text}"
        )

    if not records:
        raise ValueError("No images were successfully analyzed.")

    df = pd.DataFrame(records).sort_values("depth_um").reset_index(drop=True)
    return df


# This function prints a summary of the image analysis results in a console output format, showing the counts of white and black pixels for each image, as well as the percentage of white pixels and the corresponding depth.
def print_image_analysis_summary(df: pd.DataFrame) -> None:
    print(colored("Counts of pixels by color in each image", "yellow"))
    for row in df.itertuples(index=False):
        print(colored(f"White pixels in {row.basename}: {row.white_pixels}", "white"))
        print(colored(f"Black pixels in {row.basename}: {row.black_pixels}", "blue"))
        print()

    print(colored("Percent white pixels for each image", "yellow"))
    for row in df.itertuples(index=False):
        print(colored(f"{row.basename}", "red"))
        print(f"{row.percent_white_pixels:.6f}% White | Depth: {row.depth_um:.1f} microns")
        print()


# This function saves the analysis table DataFrame to a CSV file in the results directory, ensuring that the directory exists and providing feedback on where the file was saved.
def save_analysis_table(df: pd.DataFrame, save_path: Path = RESULTS_CSV_PATH) -> Path:
    ensure_results_dir()
    df.to_csv(save_path, index=False)
    print(colored(f"Saved analysis table to: {save_path}", "green"))
    return save_path


# This function plots the measured percent white pixels vs. depth using Matplotlib, with options to show the plot and/or save it to a file.
def plot_measured_data(
    df: pd.DataFrame,
    show_plot: bool = SHOW_PLOTS,
    save_plot: bool = SAVE_FIGURES,
    save_path: Path = MEASURED_PLOT_PATH,
) -> None:
    x = df["depth_um"].to_numpy(dtype=float)
    y = df["percent_white_pixels"].to_numpy(dtype=float)

# We create a scatter plot with lines connecting the points to visualize the relationship between depth and percent white pixels.
    plt.figure(figsize=(9, 6))
    plt.plot(x, y, marker="o", linestyle="-", linewidth=2)
    plt.xlabel("Depth into lung (microns)")
    plt.ylabel("White pixels (% of total pixels)")
    plt.title("Measured Fibrosis Proxy vs. Lung Depth")
    plt.grid(True)
    plt.tight_layout()

# Here we handle saving the plot to a file if the user has enabled that option, ensuring that the results directory exists and providing feedback on where the plot was saved. 
# We also handle showing or closing the plot based on user settings.
    if save_plot:
        ensure_results_dir()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(colored(f"Saved measured-data plot to: {save_path}", "green"))

    if show_plot:
        plt.show()
    else:
        plt.close()

# This function builds the analysis table, prints a summary of the results, saves the table to a CSV file, and plots the measured data.
def build_unique_interpolation_arrays(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    grouped = (
        df.groupby("depth_um", as_index=False)["percent_white_pixels"]
        .mean()
        .reset_index()
        .sort_values("depth_um")
        .reset_index(drop=True)
    )

    x = grouped["depth_um"].to_numpy(dtype=float)
    y = grouped["percent_white_pixels"].to_numpy(dtype=float)

    if len(x) < 2:
        raise ValueError("At least 2 distinct depths are required for interpolation.")

    return x, y


# =============================================================================
# MANUAL POLYNOMIAL INTERPOLATION (MATRIX APPROACH)
# =============================================================================

# The following functions implement manual linear and quadratic interpolation using matrix algebra, which will be used to verify the results from SciPy's interpolation and to provide an educational example of how polynomial interpolation works under the hood.
def choose_two_points_for_linear(x: np.ndarray, y: np.ndarray, target_x: float) -> Tuple[np.ndarray, np.ndarray]:
    idx = int(np.searchsorted(x, target_x))

    if idx <= 0:
        indices = [0, 1]
    elif idx >= len(x):
        indices = [len(x) - 2, len(x) - 1]
    else:
        indices = [idx - 1, idx]

    return x[indices], y[indices]


# This function selects three points from the data that are closest to the target x-value for use in quadratic interpolation. It handles edge cases where the target x-value is near the beginning or end of the data range.
def choose_three_points_for_quadratic(x: np.ndarray, y: np.ndarray, target_x: float) -> Tuple[np.ndarray, np.ndarray]:
    if len(x) < 3:
        raise ValueError("At least 3 distinct depths are required for quadratic interpolation.")

    idx = int(np.searchsorted(x, target_x))

    if idx <= 1:
        indices = [0, 1, 2]
    elif idx >= len(x) - 1:
        indices = [len(x) - 3, len(x) - 2, len(x) - 1]
    else:
        indices = [idx - 1, idx, idx + 1]

    return x[indices], y[indices]


# This function performs linear interpolation by solving for the coefficients of a linear polynomial that fits the two selected points and then evaluating that polynomial at the target x-value.
def solve_linear_coefficients(x_points: Sequence[float], y_points: Sequence[float]) -> np.ndarray:
    x1, x2 = [float(v) for v in x_points]
    y1, y2 = [float(v) for v in y_points]

    z_matrix = np.array([[1.0, x1], [1.0, x2]], dtype=float)
    y_vector = np.array([y1, y2], dtype=float)
    return np.linalg.solve(z_matrix, y_vector)


# This function performs quadratic interpolation by solving for the coefficients of a quadratic polynomial that fits the three selected points and then evaluating that polynomial at the target x-value.
def solve_quadratic_coefficients(x_points: Sequence[float], y_points: Sequence[float]) -> np.ndarray:
    x1, x2, x3 = [float(v) for v in x_points]
    y1, y2, y3 = [float(v) for v in y_points]

    z_matrix = np.array(
        [
            [1.0, x1, x1 ** 2],
            [1.0, x2, x2 ** 2],
            [1.0, x3, x3 ** 2],
        ],
        dtype=float,
    )
    y_vector = np.array([y1, y2, y3], dtype=float)
    return np.linalg.solve(z_matrix, y_vector)


# This function evaluates a linear polynomial at a given x-value using the coefficients obtained from the linear interpolation.
def evaluate_linear_polynomial(coefficients: Sequence[float], x_value: float) -> float:
    a1, a2 = [float(v) for v in coefficients]
    return a1 + a2 * float(x_value)


# This function evaluates a quadratic polynomial at a given x-value using the coefficients obtained from the quadratic interpolation.
def evaluate_quadratic_polynomial(coefficients: Sequence[float], x_value: float) -> float:
    a1, a2, a3 = [float(v) for v in coefficients]
    x_value = float(x_value)
    return a1 + a2 * x_value + a3 * (x_value ** 2)


# =============================================================================
# SCIPY INTERPOLATION (main_example.py)
# =============================================================================

# Here we define a function that uses SciPy's interp1d to perform interpolation. 
# This function takes in the x and y data points, the target x-value for interpolation, and the kind of interpolation (e.g., "linear" or "quadratic") and returns the interpolated y-value at the target x.
def scipy_interpolate(
    x: np.ndarray,
    y: np.ndarray,
    target_x: float,
    kind: str = "linear",
) -> float:
    interpolator = interp1d(
        x,
        y,
        kind=kind,
        bounds_error=False,
        fill_value="extrapolate", # type: ignore
        assume_sorted=True,
    )
    return float(interpolator(target_x))


# This function saves the results of the interpolation analysis to a CSV file, ensuring that the results directory exists and providing feedback on where the file was saved.
def save_interpolation_results(results: List[Dict[str, object]]) -> Path:
    ensure_results_dir()
    df = pd.DataFrame(results)
    df.to_csv(INTERPOLATION_RESULTS_CSV_PATH, index=False)
    print(colored(f"Saved interpolation results to: {INTERPOLATION_RESULTS_CSV_PATH}", "green"))
    return INTERPOLATION_RESULTS_CSV_PATH


# This function plots the measured data points along with the interpolated point at the target depth, using Matplotlib. It includes options to show the plot and/or save it to a file.
def plot_interpolation_result(
    x: np.ndarray,
    y: np.ndarray,
    target_x: float,
    target_y: float,
    method_name: str,
    save_path: Path,
    show_plot: bool = SHOW_PLOTS,
    save_plot: bool = SAVE_FIGURES,
) -> None:

# We plot the original measured data points as a line with markers, and we highlight the interpolated point with a red scatter marker. 
# The plot includes labels, a title indicating the interpolation method and target depth, a grid, and a legend.
    plt.figure(figsize=(9, 6))
    plt.plot(x, y, marker="o", linestyle="-", linewidth=2, label="Measured data")
    plt.scatter(
        target_x,
        target_y,
        s=110,
        color="red",
        zorder=5,
        label=f"Interpolated point ({target_x:.1f}, {target_y:.4f})",
    )
    plt.xlabel("Depth into lung (microns)")
    plt.ylabel("White pixels (% of total pixels)")
    plt.title(f"{method_name} Interpolation of Fibrosis at {target_x:.1f} microns")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if save_plot:
        ensure_results_dir()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(colored(f"Saved {method_name.lower()} interpolation plot to: {save_path}", "green"))

    if show_plot:
        plt.show()
    else:
        plt.close()

# =============================================================================
# VERIFICATION AND VALIDATION
# =============================================================================

# This function performs verification checks on the analysis table to ensure that the pixel counts and percentages are consistent and valid.
def verify_analysis_table(df: pd.DataFrame) -> pd.DataFrame:
    checks = pd.DataFrame(
        {
            "basename": df["basename"],
            "pixel_sum_matches_total": (df["white_pixels"] + df["black_pixels"]) == df["total_pixels"],
            "percent_between_0_and_100": df["percent_white_pixels"].between(0, 100),
        }
    )

    print(colored("Verification checks on the image-analysis table:", "yellow"))
    print(checks)
    print()
    return checks


# This function verifies the linear interpolation by manually calculating the coefficients of the linear polynomial that fits the two selected points and evaluating it at the target x-value, then comparing it to the value obtained from SciPy's interpolation. 
# It returns a dictionary containing the manual linear value, the SciPy linear value, and their absolute difference.
def verify_linear_interpolation(
    x: np.ndarray,
    y: np.ndarray,
    target_x: float,
) -> Dict[str, float]:

    chosen_x, chosen_y = choose_two_points_for_linear(x, y, target_x)
    manual_coeffs = solve_linear_coefficients(chosen_x.tolist(), chosen_y.tolist())
    manual_value = evaluate_linear_polynomial(manual_coeffs, target_x) # type: ignore
    numpy_value = float(np.interp(target_x, x, y))
    absolute_difference = abs(manual_value - numpy_value)

    result = {
        "manual_linear_value": manual_value,
        "numpy_linear_value": numpy_value,
        "absolute_difference": absolute_difference,
    }

    print(colored("Linear interpolation verification:", "yellow"))
    for key, value in result.items():
        print(f"{key}: {value}")
    print()

    return result


# This function verifies the quadratic interpolation by manually calculating the coefficients of the quadratic polynomial that fits the three selected points and evaluating it at the target x-value, then comparing it to the value obtained from SciPy's interpolation.
# It returns a dictionary containing the manual quadratic value, the SciPy quadratic value, and their
def find_nearest_validation_image(
    target_depth_um: float,
    selected_basenames: Sequence[str],
) -> Tuple[Path, float, str]:

    lookup = load_depth_lookup().copy()
    selected_normalized = {normalize_filename(name) for name in selected_basenames}

    lookup["abs_depth_error"] = (lookup["depth_um"] - float(target_depth_um)).abs()
    lookup = lookup.sort_values(["abs_depth_error", "depth_um"]).reset_index(drop=True)

    # First try to find an image not already used in the selected subset.
    for row in lookup.itertuples(index=False):
        if row.basename in selected_normalized:
            continue
        image_path = find_image_path_by_basename(str(row.basename), [ALL_IMAGES_DIR, CHOSEN_IMAGES_DIR])
        if image_path is not None:
            return image_path, float(row.depth_um), str(row.basename) # type: ignore

    # If every nearby image is in the selected subset, allow reuse.
    for row in lookup.itertuples(index=False):
        image_path = find_image_path_by_basename(str(row.basename), [ALL_IMAGES_DIR, CHOSEN_IMAGES_DIR])
        if image_path is not None:
            return image_path, float(row.depth_um), str(row.basename) # type: ignore

    raise FileNotFoundError(
        "Could not locate any validation image in the full dataset."
    )


# Now we define the main validation function that performs the validation of the interpolation against a real measured image. 
# It takes in the analysis DataFrame, the target depth for interpolation, the interpolation method to use, and options for saving and showing the results. 
# It returns a DataFrame containing the validation results, including the predicted percent white pixels from interpolation, the measured percent white pixels from the validation image, and the errors between them.
def run_validation(
    analysis_df: pd.DataFrame,
    target_depth_um: float = INTERPOLATION_DEPTH_UM,
    method: str = VALIDATION_METHOD,
    save_csv_path: Path = VALIDATION_CSV_PATH,
    show_plot: bool = SHOW_PLOTS,
    save_plot: bool = SAVE_FIGURES,
) -> pd.DataFrame:
 
 # First, we build the unique interpolation arrays from the analysis DataFrame, which will be used for both the interpolation and the validation steps.
    x, y = build_unique_interpolation_arrays(analysis_df)

    method_lower = method.lower().strip()
    if method_lower == "linear":
        predicted_percent = scipy_interpolate(x, y, target_depth_um, kind="linear")
    elif method_lower == "quadratic":
        if len(x) < 3:
            raise ValueError("Quadratic validation requires at least 3 distinct depths.")
        predicted_percent = scipy_interpolate(x, y, target_depth_um, kind="quadratic")
    else:
        raise ValueError("Validation method must be 'linear' or 'quadratic'.")

    validation_image_path, validation_depth_um, validation_basename = find_nearest_validation_image(
        target_depth_um=target_depth_um,
        selected_basenames=analysis_df["basename"].tolist(),
    )

    validation_record = analyze_single_image(validation_image_path, validation_depth_um)
    measured_percent = float(validation_record["percent_white_pixels"]) # type: ignore

    absolute_error = abs(measured_percent - predicted_percent)
    percent_relative_error = (
        np.nan if measured_percent == 0 else 100.0 * absolute_error / abs(measured_percent)
    )

    # We compile the validation results into a DataFrame for easy saving and display. 
    # This includes the target depth, interpolation method, predicted percent white pixels, validation image details, measured percent white pixels, and error metrics.
    validation_df = pd.DataFrame(
        {
            "target_depth_um": [float(target_depth_um)],
            "interpolation_method": [method_lower],
            "predicted_percent_white": [predicted_percent],
            "validation_image_basename": [validation_basename],
            "validation_image_depth_um": [validation_depth_um],
            "measured_percent_white": [measured_percent],
            "absolute_error": [absolute_error],
            "percent_relative_error": [percent_relative_error],
        }
    )

# Ok, now we save the validation results to a CSV file and print the results. 
# We also create a plot that shows the original measured data points, the interpolated estimate at the target depth, and the actual measured value from the validation image, all in one figure for easy comparison.
    ensure_results_dir()
    validation_df.to_csv(save_csv_path, index=False)
    print(colored(f"Saved interpolation validation table to: {save_csv_path}", "green"))
    print(colored("Validation result:", "yellow"))
    print(validation_df)
    print()

    # Validation plot
    plt.figure(figsize=(9, 6))
    plt.plot(analysis_df["depth_um"], analysis_df["percent_white_pixels"], marker="o", linewidth=2, label="Selected measured data")
    plt.scatter(target_depth_um, predicted_percent, s=120, color="red", zorder=5, label=f"Interpolated estimate ({method_lower})")
    plt.scatter(validation_depth_um, measured_percent, s=120, color="green", zorder=5, label="Nearest real validation image")
    plt.xlabel("Depth into lung (microns)")
    plt.ylabel("White pixels (% of total pixels)")
    plt.title("Validation of Interpolated Fibrosis Estimate")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if save_plot:
        plt.savefig(VALIDATION_PLOT_PATH, dpi=300, bbox_inches="tight")
        print(colored(f"Saved validation plot to: {VALIDATION_PLOT_PATH}", "green"))

    if show_plot:
        plt.show()
    else:
        plt.close()

    return validation_df


# Finally, we define a function to benchmark the runtime of the image analysis pipeline. 
# This function runs the analysis multiple times, measures the total time taken, and estimates how long it would take to run on a larger number of images (e.g., 10,000). 
# The results are saved to a CSV file and printed to the console.
def benchmark_pipeline(
    repeats: int = 5,
    save_csv_path: Path = BENCHMARK_CSV_PATH,
) -> pd.DataFrame:
 
 # We load the selected image paths and the depth lookup, then we run the analysis of all selected images multiple times while measuring the time taken for each run.
    image_paths = get_selected_image_paths()
    depth_lookup = load_depth_lookup()
    depth_map = dict(zip(depth_lookup["basename"], depth_lookup["depth_um"]))

    times_seconds: List[float] = []
    n_images = len(image_paths)

    for _ in range(repeats):
        start = perf_counter()
        for image_path in image_paths:
            basename_key = normalize_filename(image_path.name)
            _ = analyze_single_image(image_path, depth_map[basename_key])
        end = perf_counter()
        times_seconds.append(end - start)

    mean_total_seconds = float(np.mean(times_seconds))
    mean_seconds_per_image = mean_total_seconds / max(n_images, 1)
    estimated_seconds_for_10000 = mean_seconds_per_image * 10000.0
    estimated_minutes_for_10000 = estimated_seconds_for_10000 / 60.0

    benchmark_df = pd.DataFrame(
        {
            "n_images_tested": [n_images],
            "repeats": [repeats],
            "mean_total_seconds": [mean_total_seconds],
            "mean_seconds_per_image": [mean_seconds_per_image],
            "estimated_seconds_for_10000_images": [estimated_seconds_for_10000],
            "estimated_minutes_for_10000_images": [estimated_minutes_for_10000],
        }
    )

# We save the benchmark results to a CSV file and print the results in a clear format.
    ensure_results_dir()
    benchmark_df.to_csv(save_csv_path, index=False)
    print(colored(f"Saved runtime benchmark to: {save_csv_path}", "green"))
    print(colored("Runtime benchmark:", "yellow"))
    print(benchmark_df)
    print()

    return benchmark_df


# =============================================================================
# NOTEBOOK-FRIENDLY WRAPPERS
# =============================================================================

# This function runs the entire image analysis pipeline, including building the analysis table, printing a summary, saving the results, plotting the measured data, and verifying the analysis table. 
# It returns the analysis DataFrame for further use in interpolation and validation steps.
def run_image_analysis() -> pd.DataFrame:
    analysis_df = build_analysis_table()
    print_image_analysis_summary(analysis_df)
    save_analysis_table(analysis_df)
    plot_measured_data(analysis_df)
    verify_analysis_table(analysis_df)
    return analysis_df

# This function runs the linear interpolation analysis, including both the SciPy method and the manual matrix method (if enabled), and saves the results to a CSV file. 
# It also plots the interpolation result and verifies the linear interpolation against a manual calculation.
def run_linear_interpolation(
    analysis_df: Optional[pd.DataFrame] = None,
    target_depth_um: float = INTERPOLATION_DEPTH_UM,
) -> Dict[str, object]:

    if analysis_df is None:
        analysis_df = build_analysis_table()

    x, y = build_unique_interpolation_arrays(analysis_df)
    scipy_value = scipy_interpolate(x, y, target_depth_um, kind="linear")

    result: Dict[str, object] = {
        "method": "linear",
        "target_depth_um": float(target_depth_um),
        "scipy_interpolated_percent_white": scipy_value,
    }

    print(colored("Linear interpolation result:", "yellow"))
    print(f"Target depth: {target_depth_um:.1f} microns")
    print(f"SciPy linear interpolation: {scipy_value:.6f}% white pixels")

    if RUN_MANUAL_POLYNOMIAL_INTERPOLATION:
        chosen_x, chosen_y = choose_two_points_for_linear(x, y, target_depth_um)
        coeffs = solve_linear_coefficients(chosen_x.tolist(), chosen_y.tolist())
        manual_value = evaluate_linear_polynomial(coeffs, target_depth_um) # type: ignore
        result.update(
            {
                "manual_coefficients": coeffs.tolist(),
                "manual_interpolated_percent_white": manual_value,
                "manual_points_used_depth_um": chosen_x.tolist(),
                "manual_points_used_percent_white": chosen_y.tolist(),
                "manual_vs_scipy_absolute_difference": abs(manual_value - scipy_value),
            }
        )
        print(f"Manual matrix linear interpolation: {manual_value:.6f}% white pixels")
        print(f"Coefficients [a1, a2]: {coeffs}")

    print()
    verify_linear_interpolation(x, y, target_depth_um)
    plot_interpolation_result(x, y, target_depth_um, scipy_value, "Linear", LINEAR_PLOT_PATH)
    save_interpolation_results([result])
    return result


# This function runs the quadratic interpolation analysis, including both the SciPy method and the manual matrix method (if enabled), and saves the results to a CSV file. 
# It also plots the interpolation result and verifies the quadratic interpolation against a manual calculation.
def run_quadratic_interpolation(
    analysis_df: Optional[pd.DataFrame] = None,
    target_depth_um: float = INTERPOLATION_DEPTH_UM,
) -> Dict[str, object]:

# We check if the analysis DataFrame is provided; if not, we build it. Then we prepare the x and y arrays for interpolation, ensuring that there are enough distinct depths for quadratic interpolation. We perform the SciPy quadratic interpolation and store the results in a dictionary.
    if analysis_df is None:
        analysis_df = build_analysis_table()

    x, y = build_unique_interpolation_arrays(analysis_df)
    if len(x) < 3:
        raise ValueError("Quadratic interpolation requires at least 3 distinct depths.")

    scipy_value = scipy_interpolate(x, y, target_depth_um, kind="quadratic")

    result: Dict[str, object] = {
        "method": "quadratic",
        "target_depth_um": float(target_depth_um),
        "scipy_interpolated_percent_white": scipy_value,
    }

    print(colored("Quadratic interpolation result:", "yellow"))
    print(f"Target depth: {target_depth_um:.1f} microns")
    print(f"SciPy quadratic interpolation: {scipy_value:.6f}% white pixels")

    # If the manual polynomial interpolation option is enabled, we select three points for quadratic interpolation, solve for the coefficients of the quadratic polynomial, evaluate it at the target depth, and compare it to the SciPy result. 
    # We also print the coefficients and the manual interpolation result.
    if RUN_MANUAL_POLYNOMIAL_INTERPOLATION:
        chosen_x, chosen_y = choose_three_points_for_quadratic(x, y, target_depth_um)
        coeffs = solve_quadratic_coefficients(chosen_x.tolist(), chosen_y.tolist())
        manual_value = evaluate_quadratic_polynomial(coeffs, target_depth_um) # type: ignore
        result.update(
            {
                "manual_coefficients": coeffs.tolist(),
                "manual_interpolated_percent_white": manual_value,
                "manual_points_used_depth_um": chosen_x.tolist(),
                "manual_points_used_percent_white": chosen_y.tolist(),
                "manual_vs_scipy_absolute_difference": abs(manual_value - scipy_value),
            }
        )
        print(f"Manual matrix quadratic interpolation: {manual_value:.6f}% white pixels")
        print(f"Coefficients [a1, a2, a3]: {coeffs}")

    print()
    plot_interpolation_result(x, y, target_depth_um, scipy_value, "Quadratic", QUADRATIC_PLOT_PATH)

    existing_rows = []
    if INTERPOLATION_RESULTS_CSV_PATH.exists():
        try:
            existing_rows = pd.read_csv(INTERPOLATION_RESULTS_CSV_PATH).to_dict(orient="records")
        except Exception:
            warnings.warn("Could not read existing interpolation results CSV; it will be overwritten.")
            existing_rows = []

    existing_rows.append(result) # type: ignore
    save_interpolation_results(existing_rows) # type: ignore
    return result


# This function runs the full pipeline of image analysis, interpolation, validation, and benchmarking.
#  It takes in parameters for the target depth for interpolation, whether to run the validation step, and whether to run the benchmark step. 
# It returns a dictionary containing all the results from each step of the pipeline.
def run_full_pipeline(
    target_depth_um: float = INTERPOLATION_DEPTH_UM,
    run_validation_step: bool = RUN_VALIDATION_STEP,
    run_benchmark_step: bool = RUN_BENCHMARK_STEP,
) -> Dict[str, object]:
    ensure_results_dir()

    analysis_df = run_image_analysis()
    linear_result = run_linear_interpolation(analysis_df=analysis_df, target_depth_um=target_depth_um)
    quadratic_result = run_quadratic_interpolation(analysis_df=analysis_df, target_depth_um=target_depth_um)

    output: Dict[str, object] = {
        "analysis_table": analysis_df,
        "linear_result": linear_result,
        "quadratic_result": quadratic_result,
    }

    if run_validation_step:
        validation_df = run_validation(
            analysis_df=analysis_df,
            target_depth_um=target_depth_um,
            method=VALIDATION_METHOD,
        )
        output["validation_table"] = validation_df

    if run_benchmark_step:
        benchmark_df = benchmark_pipeline()
        output["benchmark_table"] = benchmark_df

    return output


# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================

# Finally, we define the main function that serves as the entry point for the script. 
# It prints some introductory information about the pipeline, runs the full pipeline, and then prints a summary of the results, including where the main results CSV file is located and any additional files from validation and benchmarking steps.
def main() -> None:
    print(colored("Running Module 3 fibrosis image-analysis pipeline...", "cyan"))
    print(colored(f"Project root: {PROJECT_ROOT}", "cyan"))
    print(colored(f"Results folder: {RESULTS_DIR}", "cyan"))
    print()

    run_full_pipeline(target_depth_um=INTERPOLATION_DEPTH_UM)

    print(colored("Module 3 pipeline finished successfully.", "green"))
    print(colored(f"Main results CSV: {RESULTS_CSV_PATH}", "green"))
    print(colored(f"Interpolation results CSV: {INTERPOLATION_RESULTS_CSV_PATH}", "green"))
    if RUN_VALIDATION_STEP:
        print(colored(f"Validation CSV: {VALIDATION_CSV_PATH}", "green"))
    if RUN_BENCHMARK_STEP:
        print(colored(f"Benchmark CSV: {BENCHMARK_CSV_PATH}", "green"))

# This block checks if the script is being run directly (as the main module) and, if so, it calls the main function to execute the pipeline.
if __name__ == "__main__":
    main()
