"""
Advanced Multi-Method Image Fusion for White-Light and ALS Images
----------------------------------------------------------------

This module provides multiple high-quality fusion algorithms for combining white-light
and ALS (Alternate Light Source) images. Implements state-of-the-art methods including:

1. Frequency Domain Fusion - Enhanced spatial-frequency blending
2. Laplacian Pyramid Fusion - Multi-scale decomposition with adaptive weights
3. Wavelet-Based Fusion - DWT with perceptual quality optimization
4. Gradient-Based Fusion - Edge-preserving fusion with local contrast enhancement
5. Hybrid Methods - Combining multiple approaches for optimal results

Key Features:
- Multiple fusion algorithms with academic backing
- Adaptive parameter selection based on image content
- Quality assessment metrics (SSIM, PSNR, MI)
- Support for various image formats
- Real-time processing feedback
- Comprehensive debug visualization

Usage (CLI):
    python core.py --white /path/to/white.jpg --als /path/to/als.jpg --out /path/to/fused.jpg \
        --method laplacian_pyramid --preserve_color lab --debug_dir /tmp/debug

Author: AI Assistant
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import cv2
import numpy as np
import pywt
from typing import Optional, Tuple, Dict, Any, Callable, List, Union
from enum import Enum
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy.ndimage import gaussian_filter
import warnings
warnings.filterwarnings('ignore')
from src.settings import logger

class FusionMethod(Enum):
    """Available fusion methods."""
    FREQUENCY_DOMAIN = "frequency_domain"
    LAPLACIAN_PYRAMID = "laplacian_pyramid"
    WAVELET_DWT = "wavelet_dwt"
    GRADIENT_BASED = "gradient_based"
    HYBRID_ADAPTIVE = "hybrid_adaptive"


@dataclass
class FusionConfig:
    # Method selection
    method: FusionMethod = FusionMethod.FREQUENCY_DOMAIN

    # Alignment & pre-processing
    max_size: int = 2200             # resize longest side before processing
    try_ecc: bool = False            # run ECC refinement after homography

    # Frequency domain parameters
    sigma_low: float = 8.0           # Gaussian sigma for low-pass (white)
    sigma_high: float = 2.0          # Gaussian sigma for high-pass (ALS)
    w_low: float = 0.5               # Weight for low-pass white component
    w_high: float = 1.0              # Weight for high-pass ALS component

    # Laplacian pyramid parameters
    pyramid_levels: int = 5          # Number of pyramid levels
    pyramid_sigma: float = 1.0       # Gaussian sigma for pyramid construction

    # Wavelet parameters
    wavelet_type: str = 'db4'        # Wavelet type for DWT
    wavelet_levels: int = 4          # Decomposition levels

    # Gradient-based parameters
    gradient_sigma: float = 1.0      # Sigma for gradient computation
    edge_threshold: float = 0.1      # Threshold for edge detection

    # Adaptive parameters
    local_window_size: int = 15      # Window size for local analysis
    contrast_threshold: float = 0.2  # Threshold for contrast-based decisions

    # Color handling: 'lab' (replace L), 'hsv' (replace V), or 'gray'
    preserve_color: str = "lab"

    # Quality assessment
    compute_metrics: bool = True     # Compute quality metrics

    # Diagnostics
    debug_dir: Optional[Path] = None # directory to save intermediates
    progress_callback: Optional[Callable[[str, float], None]] = None  # Progress callback


class AdvancedBruiseFusion:
    """Advanced multi-method fusion for white-light and ALS images.

    Provides multiple state-of-the-art fusion algorithms with quality assessment
    and real-time progress feedback.
    """
    def __init__(self, config: FusionConfig) -> None:
        self.cfg: FusionConfig = config
        if self.cfg.debug_dir is not None:
            self.cfg.debug_dir = Path(self.cfg.debug_dir)
            self.cfg.debug_dir.mkdir(parents=True, exist_ok=True)

        # Initialize quality metrics storage
        self.quality_metrics: Dict[str, float] = {}

        # Method dispatch table
        self.fusion_methods: Dict[FusionMethod, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
            FusionMethod.FREQUENCY_DOMAIN: self.fuse_frequency_domain,
            FusionMethod.LAPLACIAN_PYRAMID: self.fuse_laplacian_pyramid,
            FusionMethod.WAVELET_DWT: self.fuse_wavelet_dwt,
            FusionMethod.GRADIENT_BASED: self.fuse_gradient_based,
            FusionMethod.HYBRID_ADAPTIVE: self.fuse_hybrid_adaptive
        }

    def _update_progress(self, message: str, progress: float) -> None:
        """Update progress if callback is provided."""
        if self.cfg.progress_callback:
            self.cfg.progress_callback(message, progress)

    # -------------------- Enhanced IO helpers --------------------
    @staticmethod
    def imread_color(path: Union[os.PathLike, str]) -> np.ndarray:
        """Enhanced image reading with multiple format support and fallback mechanisms."""
        path_str: str = str(path)

        # Check if file exists
        if not os.path.exists(path_str):
            raise FileNotFoundError(f"Image file not found: {path_str}")

        # Get file extension for format-specific handling
        ext: str = Path(path_str).suffix.lower()

        # Handle NEF (Nikon RAW) files specifically
        if ext == '.nef':
            try:
                import rawpy
                # Load the NEF file
                with rawpy.imread(path_str) as raw:
                    # Process the raw data to RGB
                    img_rgb: np.ndarray = raw.postprocess()
                    # Convert RGB to BGR for OpenCV compatibility
                    img_bgr: np.ndarray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                    return img_bgr
            except ImportError:
                warnings.warn("rawpy library not installed. Install with: pip install rawpy")
            except Exception as e:
                warnings.warn(f"rawpy failed to read NEF file {path_str}: {e}")

        # Try OpenCV first (fastest for common formats)
        try:
            img: Optional[np.ndarray] = cv2.imread(path_str, cv2.IMREAD_COLOR)
            if img is not None:
                return img
        except Exception as e:
            warnings.warn(f"OpenCV failed to read {path_str}: {e}")

        # Try imageio for better format support
        try:
            import imageio.v3 as iio
            img_rgb: np.ndarray = iio.imread(path_str)

            # Handle different image types
            if len(img_rgb.shape) == 2:  # Grayscale
                img_bgr: np.ndarray = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2BGR)
            elif img_rgb.shape[2] == 3:  # RGB
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            elif img_rgb.shape[2] == 4:  # RGBA
                img_rgb_no_alpha: np.ndarray = img_rgb[:, :, :3]
                img_bgr = cv2.cvtColor(img_rgb_no_alpha, cv2.COLOR_RGB2BGR)
            else:
                raise ValueError(f"Unsupported image format with {img_rgb.shape[2]} channels")

            return img_bgr
        except Exception as e:
            warnings.warn(f"imageio failed to read {path_str}: {e}")

        # Try PIL as fallback for exotic formats
        try:
            from PIL import Image
            pil_img = Image.open(path_str)

            # Convert to RGB if needed
            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')

            img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            return img_bgr
        except Exception as e:
            warnings.warn(f"PIL failed to read {path_str}: {e}")

        # Try skimage as last resort
        try:
            from skimage import io
            img_rgb = io.imread(path_str)

            if len(img_rgb.shape) == 2:  # Grayscale
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2BGR)
            elif len(img_rgb.shape) == 3:
                if img_rgb.shape[2] == 3:  # RGB
                    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                elif img_rgb.shape[2] == 4:  # RGBA
                    img_rgb_no_alpha = img_rgb[:, :, :3]
                    img_bgr = cv2.cvtColor(img_rgb_no_alpha, cv2.COLOR_RGB2BGR)
                else:
                    raise ValueError(f"Unsupported image format")
            else:
                raise ValueError(f"Unsupported image dimensions: {img_rgb.shape}")

            return img_bgr
        except Exception as e:
            warnings.warn(f"skimage failed to read {path_str}: {e}")

        raise FileNotFoundError(f"Could not read image with any available library: {path_str}")

    @staticmethod
    def imwrite_color(path: Union[os.PathLike, str], img: np.ndarray, quality: int = 95) -> bool:
        """Enhanced image writing with format detection, quality control, and error handling."""
        path_obj: Path = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        ext: str = path_obj.suffix.lower()
        path_str: str = str(path_obj)

        # Validate input image
        if img is None or img.size == 0:
            raise ValueError("Invalid image data provided")

        # Ensure image is in correct format
        if len(img.shape) != 3 or img.shape[2] != 3:
            raise ValueError(f"Expected 3-channel BGR image, got shape: {img.shape}")

        # Try OpenCV first (fastest and most reliable for common formats)
        try:
            if ext in ['.jpg', '.jpeg']:
                success = cv2.imwrite(path_str, img, [cv2.IMWRITE_JPEG_QUALITY, quality])
            elif ext == '.png':
                # PNG compression level (0-9, higher = more compression)
                compression = min(9, max(0, int((100 - quality) / 10)))
                success = cv2.imwrite(path_str, img, [cv2.IMWRITE_PNG_COMPRESSION, compression])
            elif ext in ['.tiff', '.tif']:
                success = cv2.imwrite(path_str, img)
            elif ext == '.bmp':
                success = cv2.imwrite(path_str, img)
            elif ext == '.webp':
                success = cv2.imwrite(path_str, img, [cv2.IMWRITE_WEBP_QUALITY, quality])
            else:
                # Default to JPEG for unknown extensions
                path_str = str(path_obj.with_suffix('.jpg'))
                success = cv2.imwrite(path_str, img, [cv2.IMWRITE_JPEG_QUALITY, quality])

            if success:
                return True
        except Exception as e:
            warnings.warn(f"OpenCV failed to write {path_str}: {e}")

        # Try PIL as fallback
        try:
            from PIL import Image

            # Convert BGR to RGB for PIL
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)

            # Format-specific saving
            if ext in ['.jpg', '.jpeg']:
                pil_img.save(path_str, 'JPEG', quality=quality, optimize=True)
            elif ext == '.png':
                pil_img.save(path_str, 'PNG', optimize=True)
            elif ext in ['.tiff', '.tif']:
                pil_img.save(path_str, 'TIFF')
            elif ext == '.bmp':
                pil_img.save(path_str, 'BMP')
            elif ext == '.webp':
                pil_img.save(path_str, 'WEBP', quality=quality)
            else:
                # Default to JPEG
                path_str = str(path_obj.with_suffix('.jpg'))
                pil_img.save(path_str, 'JPEG', quality=quality, optimize=True)

            return True
        except Exception as e:
            warnings.warn(f"PIL failed to write {path_str}: {e}")

        # Try imageio as last resort
        try:
            import imageio.v3 as iio
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if ext in ['.jpg', '.jpeg']:
                iio.imwrite(path_str, img_rgb, quality=quality)
            else:
                iio.imwrite(path_str, img_rgb)

            return True
        except Exception as e:
            warnings.warn(f"imageio failed to write {path_str}: {e}")

        raise IOError(f"Could not write image with any available library: {path_str}")

    @staticmethod
    def resize_max_side(img: np.ndarray, max_side: int) -> Tuple[np.ndarray, float]:
        h: int
        w: int
        h, w = img.shape[:2]
        if max(h, w) <= max_side:
            return img, 1.0
        scale: float = max_side / float(max(h, w))
        out: np.ndarray = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        return out, scale

    # -------------------- Color space utils --------------------
    @staticmethod
    def to_luminance_lab(img_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        lab: np.ndarray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        L: np.ndarray = lab[:, :, 0].astype(np.float32)
        return L, lab

    @staticmethod
    def to_luminance_hsv(img_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        hsv: np.ndarray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        V: np.ndarray = hsv[:, :, 2].astype(np.float32)
        return V, hsv

    @staticmethod
    def put_luminance(base_bgr: np.ndarray, new_L: np.ndarray, method: str = "lab", base_conv: Optional[np.ndarray] = None) -> np.ndarray:
        new_L = np.clip(new_L, 0, 255).astype(np.uint8)
        if method == "lab":
            if base_conv is None:
                base_conv = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2LAB)
            lab: np.ndarray = base_conv.copy()
            lab[:, :, 0] = new_L
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        elif method == "hsv":
            if base_conv is None:
                base_conv = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2HSV)
            hsv: np.ndarray = base_conv.copy()
            hsv[:, :, 2] = new_L
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        else:  # gray
            return cv2.cvtColor(new_L, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def normalize_to_uint8(x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32)
        x -= x.min()
        denom: float = (x.max() - x.min())
        if denom < 1e-6:
            return np.zeros_like(x, dtype=np.uint8)
        x = x / denom
        return (x * 255.0).clip(0, 255).astype(np.uint8)

    @staticmethod
    def clahe(img8: np.ndarray, clip: float = 2.0) -> np.ndarray:
        clahe: cv2.CLAHE = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
        return clahe.apply(img8)

    # -------------------- Alignment --------------------
    def align_als_to_white(self, als_bgr: np.ndarray, white_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Align als_bgr to white_bgr using ORB+RANSAC homography. Returns warped ALS and H."""
        orb: cv2.ORB = cv2.ORB_create(5000)
        g1: np.ndarray = cv2.cvtColor(als_bgr, cv2.COLOR_BGR2GRAY)
        g2: np.ndarray = cv2.cvtColor(white_bgr, cv2.COLOR_BGR2GRAY)
        kp1: List[cv2.KeyPoint]
        des1: Optional[np.ndarray]
        kp2: List[cv2.KeyPoint]
        des2: Optional[np.ndarray]
        kp1, des1 = orb.detectAndCompute(g1, None)
        kp2, des2 = orb.detectAndCompute(g2, None)
        if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
            raise RuntimeError("Not enough ORB features to compute alignment. Try larger max_size or clearer images.")
        bf     : cv2.BFMatcher          = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches: List[List[cv2.DMatch]] = bf.knnMatch(des1, des2, k=2)
        good   : List[cv2.DMatch]       = [m for m, n in matches if m.distance < 0.75 * n.distance]
        if len(good) < 8:
            raise RuntimeError(f"Too few good ORB matches ({len(good)}) to estimate homography.")
        src_pts: np.ndarray = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts: np.ndarray = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H      : Optional[np.ndarray]
        mask   : Optional[np.ndarray]
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4.0)
        if H is None:
            raise RuntimeError("Homography estimation failed.")
        h: int
        w: int
        h, w = white_bgr.shape[:2]
        warped: np.ndarray = cv2.warpPerspective(als_bgr, H, (w, h), flags=cv2.INTER_LINEAR)
        if self.cfg.debug_dir is not None:
            vis: np.ndarray = cv2.drawMatches(g1, kp1, g2, kp2, [m for i, m in enumerate(good) if mask.ravel()[i] == 1], None, matchesMask=mask.ravel().tolist(), flags=2)
            cv2.imwrite(str(self.cfg.debug_dir / "01_matches.jpg"), vis)
        return warped, H

    def ecc_refine(self, warped_bgr: np.ndarray, white_bgr: np.ndarray) -> np.ndarray:
        """Optional ECC refinement (affine) if intensities are correlated enough."""
        warp_mode = cv2.MOTION_AFFINE
        number_of_iterations = 200
        termination_eps = 1e-5
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)
        im1 = cv2.GaussianBlur(cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 1.0)
        im2 = cv2.GaussianBlur(cv2.cvtColor(white_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 1.0)
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        try:
            cc, warp_matrix = cv2.findTransformECC(im2, im1, warp_matrix, warp_mode, criteria)
            h, w = white_bgr.shape[:2]
            refined = cv2.warpAffine(warped_bgr, warp_matrix, (w, h), flags=cv2.INTER_LINEAR)
            if self.cfg.debug_dir is not None:
                (self.cfg.debug_dir / "02_ecc_score.txt").write_text(f"ECC correlation: {cc}\n")
            return refined
        except cv2.error:
            return warped_bgr

    # -------------------- Frequency fusion --------------------
    def als_pseudo_luminance(self, als_bgr: np.ndarray) -> np.ndarray:
        """Enhanced ALS luminance extraction with better bruise contrast."""
        b, g, r = cv2.split(als_bgr.astype(np.float32))

        # Enhanced blue-weighted luminance for better bruise visibility
        als_like = 0.7 * b + 0.15 * g + 0.15 * r  # stronger blue bias

        # Apply contrast enhancement to emphasize bruise details
        als_like = cv2.convertScaleAbs(als_like, alpha=1.2, beta=10)

        # Apply adaptive histogram equalization for local contrast
        als_like_uint8 = self.normalize_to_uint8(als_like)
        als_enhanced = self.clahe(als_like_uint8, clip=3.0)

        return als_enhanced.astype(np.float32)

    def lowpass(self, L: np.ndarray, sigma: float) -> np.ndarray:
        return cv2.GaussianBlur(L, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma).astype(np.float32)

    def highpass(self, L: np.ndarray, sigma: float) -> np.ndarray:
        """Enhanced high-pass filter with better detail preservation."""
        blur = cv2.GaussianBlur(L, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma).astype(np.float32)
        hp = L - blur

        # Enhanced normalization to preserve more detail
        hp_mean = float(np.mean(hp))
        hp_std = float(np.std(hp)) + 1e-6

        # Use a more aggressive normalization to enhance details
        hp_norm = (hp - hp_mean) / (hp_std * 0.5)  # reduced divisor for stronger details

        # Apply edge enhancement using unsharp masking
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        hp_enhanced = cv2.filter2D(hp_norm, -1, kernel)

        # Normalize to 0-255 range with better contrast
        hp_min, hp_max = hp_enhanced.min(), hp_enhanced.max()
        if hp_max - hp_min > 1e-6:
            hp_norm = (hp_enhanced - hp_min) / (hp_max - hp_min)
            hp_norm = np.power(hp_norm, 0.8)  # gamma correction for better contrast
            hp_norm *= 255.0
        else:
            hp_norm = np.zeros_like(hp_enhanced)

        return hp_norm.astype(np.float32)

    def adaptive_blend_weights(self, lp_white: np.ndarray, hp_als: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate adaptive blending weights based on local contrast and detail strength."""
        # Calculate local variance to identify areas with high detail
        kernel = np.ones((5, 5), np.float32) / 25

        # Local variance in ALS high-pass (indicates bruise detail strength)
        hp_var = cv2.filter2D(hp_als * hp_als, -1, kernel) - cv2.filter2D(hp_als, -1, kernel) ** 2
        hp_var = np.clip(hp_var, 0, None)

        # Local variance in white low-pass (indicates texture strength)
        lp_var = cv2.filter2D(lp_white * lp_white, -1, kernel) - cv2.filter2D(lp_white, -1, kernel) ** 2
        lp_var = np.clip(lp_var, 0, None)

        # Normalize variances
        hp_var_norm = hp_var / (np.max(hp_var) + 1e-6)
        lp_var_norm = lp_var / (np.max(lp_var) + 1e-6)

        # Adaptive weights: higher ALS weight where there's more detail
        w_high_adaptive = self.cfg.w_high * (0.5 + 1.5 * hp_var_norm)
        w_low_adaptive = self.cfg.w_low * (0.8 + 0.4 * lp_var_norm)

        # Ensure weights don't exceed reasonable bounds
        w_high_adaptive = np.clip(w_high_adaptive, 0.3, 1.5)
        w_low_adaptive = np.clip(w_low_adaptive, 0.4, 1.0)

        return w_low_adaptive, w_high_adaptive

    # -------------------- Laplacian Pyramid Fusion --------------------
    def build_laplacian_pyramid(self, img: np.ndarray, levels: int) -> list:
        """Build Laplacian pyramid for an image."""
        gaussian_pyramid = [img.astype(np.float32)]

        # Build Gaussian pyramid
        for i in range(levels):
            img = cv2.pyrDown(img)
            gaussian_pyramid.append(img.astype(np.float32))

        # Build Laplacian pyramid
        laplacian_pyramid = []
        for i in range(levels):
            size = (gaussian_pyramid[i].shape[1], gaussian_pyramid[i].shape[0])
            expanded = cv2.pyrUp(gaussian_pyramid[i+1], dstsize=size)
            laplacian = gaussian_pyramid[i] - expanded
            laplacian_pyramid.append(laplacian)

        # Add the smallest Gaussian level
        laplacian_pyramid.append(gaussian_pyramid[-1])
        return laplacian_pyramid

    def reconstruct_from_laplacian_pyramid(self, pyramid: list) -> np.ndarray:
        """Reconstruct image from Laplacian pyramid."""
        img = pyramid[-1]  # Start with the smallest level

        for i in range(len(pyramid) - 2, -1, -1):
            size = (pyramid[i].shape[1], pyramid[i].shape[0])
            img = cv2.pyrUp(img, dstsize=size) + pyramid[i]

        return np.clip(img, 0, 255).astype(np.uint8)

    def fuse_laplacian_pyramid(self, white_bgr: np.ndarray, als_bgr_aligned: np.ndarray) -> np.ndarray:
        """Fuse images using Laplacian pyramid decomposition."""
        if self.cfg.progress_callback:
            self.cfg.progress_callback("Building pyramids...", 0.2)

        # Convert to luminance
        Lw, lab_w = self.to_luminance_lab(white_bgr)
        Lals = self.als_pseudo_luminance(als_bgr_aligned)

        # Build pyramids
        pyramid_white = self.build_laplacian_pyramid(Lw, self.cfg.pyramid_levels)
        pyramid_als = self.build_laplacian_pyramid(Lals, self.cfg.pyramid_levels)

        if self.cfg.progress_callback:
            self.cfg.progress_callback("Fusing pyramid levels...", 0.6)

        # Fuse pyramids level by level
        fused_pyramid = []
        for i, (pw, pa) in enumerate(zip(pyramid_white, pyramid_als)):
            if i == len(pyramid_white) - 1:  # Lowest resolution level
                fused_level = 0.5 * pw + 0.5 * pa
            else:
                # Use local energy to determine fusion weights
                # Convert to float64 for Laplacian computation to avoid format issues
                pw_64 = pw.astype(np.float64)
                pa_64 = pa.astype(np.float64)

                energy_white = cv2.Laplacian(pw_64, cv2.CV_64F) ** 2
                energy_als = cv2.Laplacian(pa_64, cv2.CV_64F) ** 2

                # Smooth the energy maps
                energy_white = cv2.GaussianBlur(energy_white, (5, 5), self.cfg.pyramid_sigma)
                energy_als = cv2.GaussianBlur(energy_als, (5, 5), self.cfg.pyramid_sigma)

                # Create weight maps
                total_energy = energy_white + energy_als + 1e-6
                weight_white = energy_white / total_energy
                weight_als = energy_als / total_energy

                # Apply adaptive weighting for bruise enhancement
                weight_als = weight_als * 1.2  # Boost ALS contribution
                weight_white = 1.0 - weight_als

                fused_level = weight_white * pw + weight_als * pa

            fused_pyramid.append(fused_level)

        if self.cfg.progress_callback:
            self.cfg.progress_callback("Reconstructing image...", 0.8)

        # Reconstruct fused image
        fused_L = self.reconstruct_from_laplacian_pyramid(fused_pyramid)

        # Restore color
        if self.cfg.preserve_color == "lab":
            out = self.put_luminance(white_bgr, fused_L, method="lab", base_conv=lab_w)
        elif self.cfg.preserve_color == "hsv":
            out = self.put_luminance(white_bgr, fused_L, method="hsv")
        else:
            out = self.put_luminance(white_bgr, fused_L, method="gray")

        return out

    # -------------------- Wavelet-based Fusion --------------------
    def fuse_wavelet_dwt(self, white_bgr: np.ndarray, als_bgr_aligned: np.ndarray) -> np.ndarray:
        """Fuse images using Discrete Wavelet Transform."""
        if self.cfg.progress_callback:
            self.cfg.progress_callback("Computing wavelet decomposition...", 0.2)

        # Convert to luminance
        Lw, lab_w = self.to_luminance_lab(white_bgr)
        Lals = self.als_pseudo_luminance(als_bgr_aligned)

        # Ensure images have even dimensions for wavelet transform
        h, w = Lw.shape
        if h % 2 != 0:
            Lw = Lw[:-1, :]
            Lals = Lals[:-1, :]
        if w % 2 != 0:
            Lw = Lw[:, :-1]
            Lals = Lals[:, :-1]

        # Multi-level wavelet decomposition
        coeffs_white = pywt.wavedec2(Lw, self.cfg.wavelet_type, level=self.cfg.wavelet_levels)
        coeffs_als = pywt.wavedec2(Lals, self.cfg.wavelet_type, level=self.cfg.wavelet_levels)

        if self.cfg.progress_callback:
            self.cfg.progress_callback("Fusing wavelet coefficients...", 0.6)

        # Fuse coefficients
        fused_coeffs = []
        for i, (cw, ca) in enumerate(zip(coeffs_white, coeffs_als)):
            if i == 0:  # Approximation coefficients (low frequency)
                fused_coeffs.append(0.5 * cw + 0.5 * ca)
            else:  # Detail coefficients (high frequency)
                # Fuse based on local variance for each subband
                if isinstance(cw, tuple):  # Detail coefficients are tuples (LH, HL, HH)
                    fused_detail = []
                    for j, (dw, da) in enumerate(zip(cw, ca)):
                        # Calculate local variance
                        var_white = cv2.GaussianBlur(dw**2, (5, 5), 1.0) - cv2.GaussianBlur(dw, (5, 5), 1.0)**2
                        var_als = cv2.GaussianBlur(da**2, (5, 5), 1.0) - cv2.GaussianBlur(da, (5, 5), 1.0)**2

                        # Create fusion weights based on variance
                        total_var = var_white + var_als + 1e-6
                        weight_white = var_white / total_var
                        weight_als = var_als / total_var

                        # Boost ALS contribution for better bruise visibility
                        weight_als = weight_als * 1.3
                        weight_white = 1.0 - weight_als

                        fused_detail.append(weight_white * dw + weight_als * da)
                    fused_coeffs.append(tuple(fused_detail))
                else:
                    fused_coeffs.append(0.5 * cw + 0.5 * ca)

        if self.cfg.progress_callback:
            self.cfg.progress_callback("Reconstructing from wavelets...", 0.8)

        # Reconstruct fused image
        fused_L = pywt.waverec2(fused_coeffs, self.cfg.wavelet_type)
        fused_L = np.clip(fused_L, 0, 255).astype(np.uint8)

        # Resize back to original dimensions if needed
        if fused_L.shape != (h, w):
            fused_L = cv2.resize(fused_L, (w, h), interpolation=cv2.INTER_LINEAR)

        # Restore color
        if self.cfg.preserve_color == "lab":
            out = self.put_luminance(white_bgr, fused_L, method="lab", base_conv=lab_w)
        elif self.cfg.preserve_color == "hsv":
            out = self.put_luminance(white_bgr, fused_L, method="hsv")
        else:
            out = self.put_luminance(white_bgr, fused_L, method="gray")

        return out

    # -------------------- Gradient-based Fusion --------------------
    def compute_gradient_magnitude(self, img: np.ndarray) -> np.ndarray:
        """Compute gradient magnitude using Sobel operators."""
        grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        return magnitude

    def fuse_gradient_based(self, white_bgr: np.ndarray, als_bgr_aligned: np.ndarray) -> np.ndarray:
        """Fuse images based on gradient information and edge strength."""
        if self.cfg.progress_callback:
            self.cfg.progress_callback("Computing gradients...", 0.2)

        # Convert to luminance
        Lw, lab_w = self.to_luminance_lab(white_bgr)
        Lals = self.als_pseudo_luminance(als_bgr_aligned)

        # Smooth images slightly to reduce noise
        Lw_smooth = cv2.GaussianBlur(Lw, (3, 3), self.cfg.gradient_sigma)
        Lals_smooth = cv2.GaussianBlur(Lals, (3, 3), self.cfg.gradient_sigma)

        # Compute gradient magnitudes
        grad_white = self.compute_gradient_magnitude(Lw_smooth)
        grad_als = self.compute_gradient_magnitude(Lals_smooth)

        if self.cfg.progress_callback:
            self.cfg.progress_callback("Creating fusion weights...", 0.5)

        # Create edge maps
        edge_white = grad_white > (self.cfg.edge_threshold * np.max(grad_white))
        edge_als = grad_als > (self.cfg.edge_threshold * np.max(grad_als))

        # Create fusion weights based on gradient strength
        total_grad = grad_white + grad_als + 1e-6
        weight_white = grad_white / total_grad
        weight_als = grad_als / total_grad

        # Apply morphological operations to smooth weight maps
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        weight_white = cv2.morphologyEx(weight_white.astype(np.float32), cv2.MORPH_CLOSE, kernel)
        weight_als = cv2.morphologyEx(weight_als.astype(np.float32), cv2.MORPH_CLOSE, kernel)

        # Normalize weights
        total_weight = weight_white + weight_als + 1e-6
        weight_white = weight_white / total_weight
        weight_als = weight_als / total_weight

        # Boost ALS contribution in edge regions for better bruise visibility
        edge_boost = np.where(edge_als, 1.4, 1.0)
        weight_als = weight_als * edge_boost
        weight_white = 1.0 - weight_als

        if self.cfg.progress_callback:
            self.cfg.progress_callback("Blending images...", 0.8)

        # Fuse images
        fused_L = weight_white * Lw + weight_als * Lals
        fused_L = np.clip(fused_L, 0, 255).astype(np.uint8)

        # Apply edge enhancement
        kernel_sharpen = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        fused_L = cv2.filter2D(fused_L, -1, kernel_sharpen * 0.1) + fused_L * 0.9
        fused_L = np.clip(fused_L, 0, 255).astype(np.uint8)

        # Restore color
        if self.cfg.preserve_color == "lab":
            out = self.put_luminance(white_bgr, fused_L, method="lab", base_conv=lab_w)
        elif self.cfg.preserve_color == "hsv":
            out = self.put_luminance(white_bgr, fused_L, method="hsv")
        else:
            out = self.put_luminance(white_bgr, fused_L, method="gray")

        return out

    # -------------------- Hybrid Adaptive Fusion --------------------
    def analyze_local_properties(self, img: np.ndarray, window_size: int) -> Dict[str, np.ndarray]:
        """Analyze local image properties for adaptive fusion."""
        # Local variance
        kernel = np.ones((window_size, window_size), np.float32) / (window_size * window_size)
        mean = cv2.filter2D(img, -1, kernel)
        variance = cv2.filter2D(img**2, -1, kernel) - mean**2

        # Local gradient
        grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        gradient = np.sqrt(grad_x**2 + grad_y**2)

        # Local contrast (standard deviation)
        contrast = np.sqrt(np.maximum(variance, 0))

        return {
            'variance': variance,
            'gradient': gradient,
            'contrast': contrast,
            'mean': mean
        }

    def fuse_hybrid_adaptive(self, white_bgr: np.ndarray, als_bgr_aligned: np.ndarray) -> np.ndarray:
        """Hybrid adaptive fusion combining multiple techniques."""
        if self.cfg.progress_callback:
            self.cfg.progress_callback("Analyzing image properties...", 0.1)

        # Convert to luminance
        Lw, lab_w = self.to_luminance_lab(white_bgr)
        Lals = self.als_pseudo_luminance(als_bgr_aligned)

        # Analyze local properties
        props_white = self.analyze_local_properties(Lw, self.cfg.local_window_size)
        props_als = self.analyze_local_properties(Lals, self.cfg.local_window_size)

        if self.cfg.progress_callback:
            self.cfg.progress_callback("Computing multi-scale decomposition...", 0.3)

        # Multi-scale decomposition (simplified Laplacian pyramid)
        scales = 3
        white_scales = [Lw]
        als_scales = [Lals]

        for i in range(scales - 1):
            white_down = cv2.pyrDown(white_scales[-1])
            als_down = cv2.pyrDown(als_scales[-1])
            white_scales.append(white_down)
            als_scales.append(als_down)

        if self.cfg.progress_callback:
            self.cfg.progress_callback("Adaptive fusion at multiple scales...", 0.6)

        # Fuse at each scale with different strategies
        fused_scales = []
        for i, (ws, als_s) in enumerate(zip(white_scales, als_scales)):
            if i == 0:  # Full resolution - use gradient-based fusion
                grad_w = self.compute_gradient_magnitude(ws)
                grad_als = self.compute_gradient_magnitude(als_s)

                # Adaptive weights based on local contrast and gradient
                contrast_w = props_white['contrast']
                contrast_als = props_als['contrast']

                # Resize contrast maps to match current scale
                if contrast_w.shape != ws.shape:
                    contrast_w = cv2.resize(contrast_w, (ws.shape[1], ws.shape[0]))
                    contrast_als = cv2.resize(contrast_als, (als_s.shape[1], als_s.shape[0]))

                # Combined weight based on gradient and contrast
                weight_w = (grad_w + contrast_w) / 2
                weight_als = (grad_als + contrast_als) / 2

                # Normalize and boost ALS in high-contrast regions
                total_weight = weight_w + weight_als + 1e-6
                weight_w = weight_w / total_weight
                weight_als = weight_als / total_weight

                # Boost ALS where contrast is high (likely bruise regions)
                high_contrast_mask = contrast_als > self.cfg.contrast_threshold * np.max(contrast_als)
                weight_als = np.where(high_contrast_mask, weight_als * 1.5, weight_als)
                weight_w = 1.0 - weight_als

                fused_scale = weight_w * ws + weight_als * als_s

            else:  # Lower resolutions - use simpler averaging with slight ALS bias
                fused_scale = 0.4 * ws + 0.6 * als_s

            fused_scales.append(fused_scale)

        if self.cfg.progress_callback:
            self.cfg.progress_callback("Reconstructing final image...", 0.8)

        # Reconstruct from scales (simple upsampling and blending)
        fused_L = fused_scales[0]  # Start with full resolution

        for i in range(1, len(fused_scales)):
            # Upsample lower resolution and blend
            upsampled = fused_scales[i]
            for _ in range(i):
                upsampled = cv2.pyrUp(upsampled)

            # Resize to match if needed
            if upsampled.shape != fused_L.shape:
                upsampled = cv2.resize(upsampled, (fused_L.shape[1], fused_L.shape[0]))

            # Blend with decreasing weight for lower resolutions
            weight = 0.3 / i
            fused_L = (1 - weight) * fused_L + weight * upsampled

        # Final enhancement
        fused_L = np.clip(fused_L, 0, 255).astype(np.uint8)
        fused_L = self.clahe(fused_L, clip=2.0)

        # Restore color
        if self.cfg.preserve_color == "lab":
            out = self.put_luminance(white_bgr, fused_L, method="lab", base_conv=lab_w)
        elif self.cfg.preserve_color == "hsv":
            out = self.put_luminance(white_bgr, fused_L, method="hsv")
        else:
            out = self.put_luminance(white_bgr, fused_L, method="gray")

        return out

    # -------------------- Quality Metrics --------------------
    def compute_quality_metrics(self, img1: np.ndarray, img2: np.ndarray, fused: np.ndarray) -> Dict[str, float]:
        """Compute quality assessment metrics for fusion evaluation."""
        # Convert to grayscale for metrics computation
        if len(img1.shape) == 3:
            img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        else:
            img1_gray = img1

        if len(img2.shape) == 3:
            img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        else:
            img2_gray = img2

        if len(fused.shape) == 3:
            fused_gray = cv2.cvtColor(fused, cv2.COLOR_BGR2GRAY)
        else:
            fused_gray = fused

        metrics = {}

        try:
            # SSIM with respect to both input images
            ssim1 = ssim(img1_gray, fused_gray, data_range=255)
            ssim2 = ssim(img2_gray, fused_gray, data_range=255)
            metrics['ssim_white'] = float(ssim1)
            metrics['ssim_als'] = float(ssim2)
            metrics['ssim_avg'] = float((ssim1 + ssim2) / 2)

            # PSNR with respect to both input images
            psnr1 = psnr(img1_gray, fused_gray, data_range=255)
            psnr2 = psnr(img2_gray, fused_gray, data_range=255)
            metrics['psnr_white'] = float(psnr1)
            metrics['psnr_als'] = float(psnr2)
            metrics['psnr_avg'] = float((psnr1 + psnr2) / 2)

            # Mutual Information (simplified approximation)
            def mutual_info_approx(x, y):
                # Simplified MI using histogram correlation
                hist_x = cv2.calcHist([x], [0], None, [256], [0, 256]).flatten()
                hist_y = cv2.calcHist([y], [0], None, [256], [0, 256]).flatten()
                hist_x = hist_x / (np.sum(hist_x) + 1e-6)
                hist_y = hist_y / (np.sum(hist_y) + 1e-6)
                return np.corrcoef(hist_x, hist_y)[0, 1]

            mi1 = mutual_info_approx(img1_gray, fused_gray)
            mi2 = mutual_info_approx(img2_gray, fused_gray)
            metrics['mi_white'] = float(mi1) if not np.isnan(mi1) else 0.0
            metrics['mi_als'] = float(mi2) if not np.isnan(mi2) else 0.0
            metrics['mi_avg'] = float((mi1 + mi2) / 2) if not (np.isnan(mi1) or np.isnan(mi2)) else 0.0

            # Edge preservation metric
            def edge_preservation(ref, fused):
                # Compute edge strength
                edges_ref = cv2.Canny(ref, 50, 150)
                edges_fused = cv2.Canny(fused, 50, 150)

                # Calculate preservation ratio
                ref_edges = np.sum(edges_ref > 0)
                fused_edges = np.sum(edges_fused > 0)

                if ref_edges > 0:
                    return min(fused_edges / ref_edges, 1.0)
                return 0.0

            ep1 = edge_preservation(img1_gray, fused_gray)
            ep2 = edge_preservation(img2_gray, fused_gray)
            metrics['edge_preservation_white'] = float(ep1)
            metrics['edge_preservation_als'] = float(ep2)
            metrics['edge_preservation_avg'] = float((ep1 + ep2) / 2)

        except Exception as e:
            print(f"Warning: Error computing quality metrics: {e}")
            # Return default values if computation fails
            for key in ['ssim_white', 'ssim_als', 'ssim_avg', 'psnr_white', 'psnr_als', 'psnr_avg',
                       'mi_white', 'mi_als', 'mi_avg', 'edge_preservation_white', 'edge_preservation_als', 'edge_preservation_avg']:
                metrics[key] = 0.0

        return metrics

    def fuse(self, white_bgr: np.ndarray, als_bgr_aligned: np.ndarray) -> np.ndarray:
        """Main fusion method that dispatches to the selected fusion algorithm."""
        if self.cfg.progress_callback:
            self.cfg.progress_callback("Starting fusion process...", 0.0)

        # Dispatch to the appropriate fusion method
        if self.cfg.method == FusionMethod.FREQUENCY_DOMAIN:
            result = self.fuse_frequency_domain(white_bgr, als_bgr_aligned)
        elif self.cfg.method == FusionMethod.LAPLACIAN_PYRAMID:
            result = self.fuse_laplacian_pyramid(white_bgr, als_bgr_aligned)
        elif self.cfg.method == FusionMethod.WAVELET_DWT:
            result = self.fuse_wavelet_dwt(white_bgr, als_bgr_aligned)
        elif self.cfg.method == FusionMethod.GRADIENT_BASED:
            result = self.fuse_gradient_based(white_bgr, als_bgr_aligned)
        elif self.cfg.method == FusionMethod.HYBRID_ADAPTIVE:
            result = self.fuse_hybrid_adaptive(white_bgr, als_bgr_aligned)
        else:
            # Default to frequency domain
            result = self.fuse_frequency_domain(white_bgr, als_bgr_aligned)

        # Compute quality metrics if requested
        if self.cfg.compute_metrics:
            if self.cfg.progress_callback:
                self.cfg.progress_callback("Computing quality metrics...", 0.9)
            self.quality_metrics = self.compute_quality_metrics(white_bgr, als_bgr_aligned, result)

        if self.cfg.progress_callback:
            self.cfg.progress_callback("Fusion complete!", 1.0)

        return result

    def fuse_frequency_domain(self, white_bgr: np.ndarray, als_bgr_aligned: np.ndarray) -> np.ndarray:
        """Enhanced fusion with adaptive blending and better detail preservation."""
        # 1) Luminance from white
        Lw, lab_w = self.to_luminance_lab(white_bgr)

        # 2) Enhanced ALS pseudo-luminance
        Lals = self.als_pseudo_luminance(als_bgr_aligned)

        # 3) Frequency split with enhanced parameters
        lp_w = self.lowpass(Lw, self.cfg.sigma_low)
        hp_als = self.highpass(Lals, self.cfg.sigma_high)

        # 4) Calculate adaptive blending weights
        w_low_adaptive, w_high_adaptive = self.adaptive_blend_weights(lp_w, hp_als)

        if self.cfg.debug_dir is not None:
            cv2.imwrite(str(self.cfg.debug_dir / "03_L_white.jpg"), self.normalize_to_uint8(Lw))
            cv2.imwrite(str(self.cfg.debug_dir / "04_L_als_pseudo.jpg"), self.normalize_to_uint8(Lals))
            cv2.imwrite(str(self.cfg.debug_dir / "05_lowpass_white.jpg"), self.normalize_to_uint8(lp_w))
            cv2.imwrite(str(self.cfg.debug_dir / "06_highpass_als.jpg"), self.normalize_to_uint8(hp_als))
            cv2.imwrite(str(self.cfg.debug_dir / "06a_adaptive_weights_low.jpg"), self.normalize_to_uint8(w_low_adaptive * 255))
            cv2.imwrite(str(self.cfg.debug_dir / "06b_adaptive_weights_high.jpg"), self.normalize_to_uint8(w_high_adaptive * 255))

        # 5) Enhanced adaptive blending
        fused_L = (w_low_adaptive * lp_w + w_high_adaptive * hp_als)

        # Apply additional contrast enhancement to preserve bruise visibility
        fused_L = cv2.convertScaleAbs(fused_L, alpha=1.1, beta=5)
        fused_L = np.clip(fused_L, 0, 255).astype(np.uint8)

        # Apply adaptive histogram equalization with stronger parameters
        fused_L = self.clahe(fused_L, clip=2.5)

        if self.cfg.debug_dir is not None:
            cv2.imwrite(str(self.cfg.debug_dir / "07_fused_L.jpg"), fused_L)

        # 6) Restore color with enhanced luminance
        if self.cfg.preserve_color == "lab":
            out = self.put_luminance(white_bgr, fused_L, method="lab", base_conv=lab_w)
        elif self.cfg.preserve_color == "hsv":
            out = self.put_luminance(white_bgr, fused_L, method="hsv")
        else:
            out = self.put_luminance(white_bgr, fused_L, method="gray")

        return out

    # -------------------- Pipeline --------------------
    def run(self, white_path: os.PathLike | str, als_path: os.PathLike | str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run full pipeline. Returns (white_r, als_aligned, fused_bgr)."""
        white = self.imread_color(white_path)
        als = self.imread_color(als_path)
        white_r, _ = self.resize_max_side(white, self.cfg.max_size)
        als_r, _ = self.resize_max_side(als, self.cfg.max_size)
        als_aligned, H = self.align_als_to_white(als_r, white_r)
        if self.cfg.try_ecc:
            als_aligned = self.ecc_refine(als_aligned, white_r)
        fused = self.fuse(white_r, als_aligned)
        if self.cfg.debug_dir is not None:
            # side-by-side
            h = max(white_r.shape[0], als_aligned.shape[0], fused.shape[0])
            def pad(img):
                if img.shape[0] == h:
                    return img
                pad_h = h - img.shape[0]
                return cv2.copyMakeBorder(img, 0, pad_h, 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
            sbs = np.hstack([pad(white_r), pad(als_aligned), pad(fused)])
            cv2.imwrite(str(self.cfg.debug_dir / "08_side_by_side.jpg"), sbs)
        return white_r, als_aligned, fused


# -------------------- CLI --------------------
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Fuse white-light and ALS JPEGs via frequency blending.")
    p.add_argument("--white", required=True, help="Path to white-light JPEG")
    p.add_argument("--als", required=True, help="Path to ALS JPEG")
    p.add_argument("--out", required=True, help="Output fused JPEG path")
    p.add_argument("--sigma_low", type=float, default=6.0)
    p.add_argument("--sigma_high", type=float, default=3.0)
    p.add_argument("--w_low", type=float, default=0.6)
    p.add_argument("--w_high", type=float, default=0.8)
    p.add_argument("--preserve_color", choices=["lab", "hsv", "gray"], default="lab")
    p.add_argument("--max_size", type=int, default=2200)
    p.add_argument("--try_ecc", action="store_true")
    p.add_argument("--debug_dir", default=None)
    args = p.parse_args()

    cfg = FusionConfig(
        max_size       = args.max_size,
        try_ecc        = args.try_ecc,
        sigma_low      = args.sigma_low,
        sigma_high     = args.sigma_high,
        w_low          = args.w_low,
        w_high         = args.w_high,
        preserve_color = args.preserve_color,
        debug_dir      = Path(args.debug_dir) if args.debug_dir else None,
    )

    engine = AdvancedBruiseFusion(cfg)
    white_r, als_aligned, fused = engine.run(args.white, args.als)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".jpg", fused, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        raise RuntimeError("Failed to encode output JPEG.")
    out_path.write_bytes(buf.tobytes())

    print(f"Saved fused image: {out_path}")
    if cfg.debug_dir:
        print(f"Diagnostics saved to: {cfg.debug_dir}")
