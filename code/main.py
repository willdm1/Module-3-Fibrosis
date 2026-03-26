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

Designed to be called either:
- as a single script:    python main.py
- from a notebook:       import main
                         main.run_image_analysis()
                         main.run_linear_interpolation()
                         main.run_quadratic_interpolation()
                         main.run_validation()

"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import warnings

try:
    from termcolor import colored
except ModuleNotFoundError:
    def colored(text, *_args, **_kwargs):
        return str(text)

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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


# =============================================================================
# USER SETTINGS
# =============================================================================

INTERPOLATION_DEPTH_UM = 345.0
THRESHOLD_VALUE = 127
SHOW_PLOTS = True
SAVE_FIGURES = True
RUN_VALIDATION_STEP = True
RUN_BENCHMARK_STEP = True
VALIDATION_METHOD = "linear"   # choose: "linear" or "quadratic"

# Leave this list empty to automatically analyze every image in "chosen images".
# If I wanted to force a specific order, list basenames here exactly as they appear.
MANUAL_SELECTED_IMAGE_BASENAMES: List[str] = []

# If True, the script uses the lecture-style matrix approach for linear/quadratic
# interpolation in addition to SciPy's built-in interpolation.
RUN_MANUAL_POLYNOMIAL_INTERPOLATION = True


# =============================================================================
# PATHS
# =============================================================================
CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent
CHOSEN_IMAGES_DIR = PROJECT_ROOT / "chosen images"
ALL_IMAGES_DIR = PROJECT_ROOT / "images"
RESULTS_DIR = PROJECT_ROOT / "results"
DEPTHS_CSV_PATH = PROJECT_ROOT / "Filenames and Depths for Students.csv"

RESULTS_CSV_PATH = RESULTS_DIR / "Percent_White_Pixels.csv"
MEASURED_PLOT_PATH = RESULTS_DIR / "Measured_Percent_White_vs_Depth.png"
LINEAR_PLOT_PATH = RESULTS_DIR / "Linear_Interpolation.png"
QUADRATIC_PLOT_PATH = RESULTS_DIR / "Quadratic_Interpolation.png"
VALIDATION_CSV_PATH = RESULTS_DIR / "Interpolation_Validation.csv"
VALIDATION_PLOT_PATH = RESULTS_DIR / "Interpolation_Validation.png"
BENCHMARK_CSV_PATH = RESULTS_DIR / "Runtime_Benchmark.csv"
INTERPOLATION_RESULTS_CSV_PATH = RESULTS_DIR / "Interpolation_Results.csv"

SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


# =============================================================================
# BASIC HELPERS
# =============================================================================
def ensure_results_dir() -> None:
    """Create the results directory if it does not already exist."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)



def normalize_filename(name: str) -> str:
    """Return a normalized basename for robust filename matching."""
    return Path(str(name)).name.strip().lower()



def infer_filename_column(columns: Sequence[str]) -> str:
    """Infer the filename column from a CSV file."""
    lower_map = {col.lower(): col for col in columns}

    for candidate in columns:
        text = candidate.lower()
        if "file" in text or "image" in text or "name" in text:
            return candidate

    if len(columns) >= 1:
        return columns[0]

    raise ValueError("Could not infer a filename column from the depths CSV.")



def infer_depth_column(columns: Sequence[str]) -> str:
    """Infer the depth column from a CSV file."""
    for candidate in columns:
        text = candidate.lower()
        if "depth" in text:
            return candidate

    if len(columns) >= 2:
        return columns[1]

    raise ValueError("Could not infer a depth column from the depths CSV.")



def load_depth_lookup(depth_csv_path: Path = DEPTHS_CSV_PATH) -> pd.DataFrame:
    """
    Load the filename-depth table provided in class and standardize it.

    Returns a DataFrame with at least these columns:
    - basename
    - depth_um
    - original_filename_entry
    """
    if not depth_csv_path.exists():
        raise FileNotFoundError(
            f"Could not find the class depths CSV file: {depth_csv_path}"
        )

    df = pd.read_csv(depth_csv_path)
    if df.empty:
        raise ValueError(f"The depths CSV appears to be empty: {depth_csv_path}")

    filename_col = infer_filename_column(df.columns)
    depth_col = infer_depth_column(df.columns)

    cleaned = df.copy()
    cleaned["original_filename_entry"] = cleaned[filename_col].astype(str)
    cleaned["basename"] = cleaned[filename_col].astype(str).map(normalize_filename)
    cleaned["depth_um"] = pd.to_numeric(cleaned[depth_col], errors="coerce")
    cleaned = cleaned.dropna(subset=["basename", "depth_um"])

    if cleaned.empty:
        raise ValueError(
            "No valid filename/depth rows were found after cleaning the depths CSV."
        )

    cleaned = cleaned[["basename", "depth_um", "original_filename_entry"]].drop_duplicates(
        subset=["basename"], keep="first"
    )
    return cleaned.reset_index(drop=True)



def find_image_path_by_basename(basename: str, search_dirs: Iterable[Path]) -> Optional[Path]:
    """Search for an image by basename inside one or more directories."""
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



def get_selected_image_paths() -> List[Path]:
    """
    Build the list of selected images to analyze.

    Priority:
    1. If MANUAL_SELECTED_IMAGE_BASENAMES is non-empty, use those names.
    2. Otherwise, automatically load every valid image inside "chosen images".
    """
    if not CHOSEN_IMAGES_DIR.exists():
        raise FileNotFoundError(
            f"Could not find the chosen images folder: {CHOSEN_IMAGES_DIR}"
        )

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

    if not image_paths:
        raise FileNotFoundError(
            f"No supported image files were found in: {CHOSEN_IMAGES_DIR}"
        )

    return image_paths



def read_grayscale_image(image_path: Path) -> np.ndarray:
    """Load one image in grayscale."""
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image file: {image_path}")
    return image



def count_black_and_white_pixels(
    image: np.ndarray,
    threshold_value: int = THRESHOLD_VALUE,
) -> Dict[str, float]:
    """
    Threshold the image and count white/black pixels.

    This follows the lecture/main_example.py approach:
    - grayscale image
    - binary threshold at 127
    - white pixels correspond to fibrotic lesion
    - black pixels correspond to healthy lung
    """
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



def analyze_single_image(image_path: Path, depth_um: float) -> Dict[str, object]:
    """Analyze one image and return a record for the results table."""
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



def build_analysis_table() -> pd.DataFrame:
    """
    Analyze the selected images and return a sorted DataFrame.

    This is the core Lecture 1 / main_example.py workflow.
    """
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



def print_image_analysis_summary(df: pd.DataFrame) -> None:
    """Print lecture-style console output for the selected images."""
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



def save_analysis_table(df: pd.DataFrame, save_path: Path = RESULTS_CSV_PATH) -> Path:
    """Write the analysis DataFrame to the results folder."""
    ensure_results_dir()
    df.to_csv(save_path, index=False)
    print(colored(f"Saved analysis table to: {save_path}", "green"))
    return save_path



def plot_measured_data(
    df: pd.DataFrame,
    show_plot: bool = SHOW_PLOTS,
    save_plot: bool = SAVE_FIGURES,
    save_path: Path = MEASURED_PLOT_PATH,
) -> None:
    """Plot the measured percent white pixels vs. depth."""
    x = df["depth_um"].to_numpy(dtype=float)
    y = df["percent_white_pixels"].to_numpy(dtype=float)

    plt.figure(figsize=(9, 6))
    plt.plot(x, y, marker="o", linestyle="-", linewidth=2)
    plt.xlabel("Depth into lung (microns)")
    plt.ylabel("White pixels (% of total pixels)")
    plt.title("Measured Fibrosis Proxy vs. Lung Depth")
    plt.grid(True)
    plt.tight_layout()

    if save_plot:
        ensure_results_dir()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(colored(f"Saved measured-data plot to: {save_path}", "green"))

    if show_plot:
        plt.show()
    else:
        plt.close()



def build_unique_interpolation_arrays(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare sorted, unique x/y arrays for interpolation.

    If multiple images happen to share the same depth, the code averages their
    percent white values so the interpolation input remains valid.
    """
    grouped = (
        df.groupby("depth_um", as_index=False)["percent_white_pixels"]
        .mean()
        .sort_values("depth_um")
        .reset_index(drop=True)
    )

    x = grouped["depth_um"].to_numpy(dtype=float)
    y = grouped["percent_white_pixels"].to_numpy(dtype=float)

    if len(x) < 2:
        raise ValueError("At least 2 distinct depths are required for interpolation.")

    return x, y


# =============================================================================
# MANUAL POLYNOMIAL INTERPOLATION (LECTURE-STYLE / interpolation_example.py)
# =============================================================================
def choose_two_points_for_linear(x: np.ndarray, y: np.ndarray, target_x: float) -> Tuple[np.ndarray, np.ndarray]:
    """Choose two nearby points for manual linear interpolation."""
    idx = int(np.searchsorted(x, target_x))

    if idx <= 0:
        indices = [0, 1]
    elif idx >= len(x):
        indices = [len(x) - 2, len(x) - 1]
    else:
        indices = [idx - 1, idx]

    return x[indices], y[indices]



def choose_three_points_for_quadratic(x: np.ndarray, y: np.ndarray, target_x: float) -> Tuple[np.ndarray, np.ndarray]:
    """Choose three nearby points for manual quadratic interpolation."""
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



def solve_linear_coefficients(x_points: Sequence[float], y_points: Sequence[float]) -> np.ndarray:
    """Solve for a1 and a2 in y = a1 + a2*x using matrix algebra."""
    x1, x2 = [float(v) for v in x_points]
    y1, y2 = [float(v) for v in y_points]

    z_matrix = np.array([[1.0, x1], [1.0, x2]], dtype=float)
    y_vector = np.array([y1, y2], dtype=float)
    return np.linalg.solve(z_matrix, y_vector)



def solve_quadratic_coefficients(x_points: Sequence[float], y_points: Sequence[float]) -> np.ndarray:
    """Solve for a1, a2, a3 in y = a1 + a2*x + a3*x^2 using matrix algebra."""
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



def evaluate_linear_polynomial(coefficients: Sequence[float], x_value: float) -> float:
    """Evaluate y = a1 + a2*x."""
    a1, a2 = [float(v) for v in coefficients]
    return a1 + a2 * float(x_value)



def evaluate_quadratic_polynomial(coefficients: Sequence[float], x_value: float) -> float:
    """Evaluate y = a1 + a2*x + a3*x^2."""
    a1, a2, a3 = [float(v) for v in coefficients]
    x_value = float(x_value)
    return a1 + a2 * x_value + a3 * (x_value ** 2)


# =============================================================================
# SCIPY INTERPOLATION (LECTURE-STYLE / main_example.py)
# =============================================================================
def scipy_interpolate(
    x: np.ndarray,
    y: np.ndarray,
    target_x: float,
    kind: str = "linear",
) -> float:
    """Use SciPy interp1d to estimate y at the target depth."""
    interpolator = interp1d(
        x,
        y,
        kind=kind,
        bounds_error=False,
        fill_value="extrapolate",
        assume_sorted=True,
    )
    return float(interpolator(target_x))



def save_interpolation_results(results: List[Dict[str, object]]) -> Path:
    """Save interpolation summaries to CSV."""
    ensure_results_dir()
    df = pd.DataFrame(results)
    df.to_csv(INTERPOLATION_RESULTS_CSV_PATH, index=False)
    print(colored(f"Saved interpolation results to: {INTERPOLATION_RESULTS_CSV_PATH}", "green"))
    return INTERPOLATION_RESULTS_CSV_PATH



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
    """Plot measured data plus one interpolated point."""
    x_plot = np.append(x, target_x)
    y_plot = np.append(y, target_y)
    order = np.argsort(x_plot)

    plt.figure(figsize=(9, 6))
    plt.plot(x[order[:-1]] if len(order[:-1]) == len(x) else x, y, marker="o", linestyle="-", linewidth=2, label="Measured data")
    plt.scatter(target_x, target_y, s=110, color="red", zorder=5,
                label=f"Interpolated point ({target_x:.1f}, {target_y:.4f})")
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
def verify_analysis_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic verification checks for image analysis output.

    These checks help support the notebook's verification section.
    """
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



def verify_linear_interpolation(
    x: np.ndarray,
    y: np.ndarray,
    target_x: float,
) -> Dict[str, float]:
    """
    Compare manual matrix-based linear interpolation to a second linear check.

    Here we compare:
    - manual lecture-style matrix interpolation
    - NumPy's piecewise linear interpolation
    """
    chosen_x, chosen_y = choose_two_points_for_linear(x, y, target_x)
    manual_coeffs = solve_linear_coefficients(chosen_x, chosen_y)
    manual_value = evaluate_linear_polynomial(manual_coeffs, target_x)
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



def find_nearest_validation_image(
    target_depth_um: float,
    selected_basenames: Sequence[str],
) -> Tuple[Path, float, str]:
    """
    Find the nearest real image from the full dataset to validate the interpolation.

    Preference is given to an image not already in the selected subset.
    """
    lookup = load_depth_lookup().copy()
    selected_normalized = {normalize_filename(name) for name in selected_basenames}

    lookup["abs_depth_error"] = (lookup["depth_um"] - float(target_depth_um)).abs()
    lookup = lookup.sort_values(["abs_depth_error", "depth_um"]).reset_index(drop=True)

    # First try to find an image not already used in the selected subset.
    for row in lookup.itertuples(index=False):
        if row.basename in selected_normalized:
            continue
        image_path = find_image_path_by_basename(row.basename, [ALL_IMAGES_DIR, CHOSEN_IMAGES_DIR])
        if image_path is not None:
            return image_path, float(row.depth_um), row.basename

    # If every nearby image is in the selected subset, allow reuse.
    for row in lookup.itertuples(index=False):
        image_path = find_image_path_by_basename(row.basename, [ALL_IMAGES_DIR, CHOSEN_IMAGES_DIR])
        if image_path is not None:
            return image_path, float(row.depth_um), row.basename

    raise FileNotFoundError(
        "Could not locate any validation image in the full dataset."
    )



def run_validation(
    analysis_df: pd.DataFrame,
    target_depth_um: float = INTERPOLATION_DEPTH_UM,
    method: str = VALIDATION_METHOD,
    save_csv_path: Path = VALIDATION_CSV_PATH,
    show_plot: bool = SHOW_PLOTS,
    save_plot: bool = SAVE_FIGURES,
) -> pd.DataFrame:
    """
    Validate the interpolation against the nearest real measured image.

    Lecture 3 idea:
    - choose a target depth
    - interpolate using the selected subset
    - locate a real image near that same depth from the full class dataset
    - analyze the real image
    - compare interpolated vs measured
    """
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
    measured_percent = float(validation_record["percent_white_pixels"])

    absolute_error = abs(measured_percent - predicted_percent)
    percent_relative_error = (
        np.nan if measured_percent == 0 else 100.0 * absolute_error / abs(measured_percent)
    )

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



def benchmark_pipeline(
    repeats: int = 5,
    save_csv_path: Path = BENCHMARK_CSV_PATH,
) -> pd.DataFrame:
    """
    Rough runtime benchmark for image analysis.

    This helps answer the class prompt about efficiency and about estimating how
    long the code would take for a larger number of images.
    """
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
def run_image_analysis() -> pd.DataFrame:
    """
    Section 1 for the notebook:
    - analyze chosen images
    - save CSV
    - plot measured data
    - run basic verification checks
    """
    analysis_df = build_analysis_table()
    print_image_analysis_summary(analysis_df)
    save_analysis_table(analysis_df)
    plot_measured_data(analysis_df)
    verify_analysis_table(analysis_df)
    return analysis_df



def run_linear_interpolation(
    analysis_df: Optional[pd.DataFrame] = None,
    target_depth_um: float = INTERPOLATION_DEPTH_UM,
) -> Dict[str, object]:
    """
    Section 2 for the notebook:
    - perform linear interpolation
    - include both manual matrix method and SciPy method
    - save a plot and save a summary row
    """
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
        coeffs = solve_linear_coefficients(chosen_x, chosen_y)
        manual_value = evaluate_linear_polynomial(coeffs, target_depth_um)
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



def run_quadratic_interpolation(
    analysis_df: Optional[pd.DataFrame] = None,
    target_depth_um: float = INTERPOLATION_DEPTH_UM,
) -> Dict[str, object]:
    """
    Section 3 for the notebook:
    - perform quadratic interpolation
    - include both manual matrix method and SciPy method
    - save a plot and save a summary row
    """
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

    if RUN_MANUAL_POLYNOMIAL_INTERPOLATION:
        chosen_x, chosen_y = choose_three_points_for_quadratic(x, y, target_depth_um)
        coeffs = solve_quadratic_coefficients(chosen_x, chosen_y)
        manual_value = evaluate_quadratic_polynomial(coeffs, target_depth_um)
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

    existing_rows.append(result)
    save_interpolation_results(existing_rows)
    return result



def run_full_pipeline(
    target_depth_um: float = INTERPOLATION_DEPTH_UM,
    run_validation_step: bool = RUN_VALIDATION_STEP,
    run_benchmark_step: bool = RUN_BENCHMARK_STEP,
) -> Dict[str, object]:
    """
    Run the entire Module 3 workflow in one call.

    This is the easiest way to reproduce all key outputs for the final notebook.
    """
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


if __name__ == "__main__":
    main()
