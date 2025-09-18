"""
Modular class-based frequency fusion of white-light and ALS JPEG images
----------------------------------------------------------------------

This module fuses a white-light image (low-pass structure) with an ALS image
(high-pass detail). Because you only have RGB JPEGs (not multispectral bands),
we cannot impose a literal 415 nm spectral cutoff in post-processing. Instead,
we bias the ALS contribution toward the blue channel (which correlates with
~415 nm illumination) while doing spatial-frequency fusion.

Key steps:
 1) Alignment: ORB + RANSAC homography (ALS -> White), optional ECC refinement
 2) ALS pseudo-luminance: 0.6*B + 0.2*G + 0.2*R
 3) Frequency split: Low-pass (white L), High-pass (ALS pseudo-L)
 4) Blend luminance: fused_L = w_low*LP_white + w_high*HP_ALS
 5) Restore color: put fused_L into Lab L of white-light for natural tones

Usage (CLI):
    python bruise_fusion_modular.py \
        --white /path/to/white.jpg --als /path/to/als.jpg --out /path/to/fused.jpg \
        --sigma_low 6.0 --sigma_high 3.0 --w_low 0.6 --w_high 0.8 \
        --preserve_color lab --try_ecc --debug_dir /tmp/bruise_debug

Author: ChatGPT
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import cv2
import numpy as np
from typing import Optional, Tuple


@dataclass
class FusionConfig:
    # Alignment & pre-processing
    max_size: int = 2200             # resize longest side before processing
    try_ecc: bool = False            # run ECC refinement after homography

    # Frequency fusion
    sigma_low: float = 6.0           # Gaussian sigma for low-pass (white)
    sigma_high: float = 3.0          # Gaussian sigma for high-pass (ALS)
    w_low: float = 0.6               # weight for low-pass white component
    w_high: float = 0.8              # weight for high-pass ALS component

    # Color handling: 'lab' (replace L), 'hsv' (replace V), or 'gray'
    preserve_color: str = "lab"

    # Diagnostics
    debug_dir: Optional[Path] = None # directory to save intermediates


class BruiseFusion:
    """Fuse white-light and ALS images using spatial-frequency blending.

    Methods compose a pipeline but can be used individually for experimentation.
    """
    def __init__(self, config: FusionConfig):
        self.cfg = config
        if self.cfg.debug_dir is not None:
            self.cfg.debug_dir = Path(self.cfg.debug_dir)
            self.cfg.debug_dir.mkdir(parents=True, exist_ok=True)

    # -------------------- IO helpers --------------------
    @staticmethod
    def imread_color(path: os.PathLike | str) -> np.ndarray:
        arr = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        return img

    @staticmethod
    def resize_max_side(img: np.ndarray, max_side: int) -> Tuple[np.ndarray, float]:
        h, w = img.shape[:2]
        if max(h, w) <= max_side:
            return img, 1.0
        scale = max_side / float(max(h, w))
        out = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        return out, scale

    # -------------------- Color space utils --------------------
    @staticmethod
    def to_luminance_lab(img_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        L = lab[:, :, 0].astype(np.float32)
        return L, lab

    @staticmethod
    def to_luminance_hsv(img_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        V = hsv[:, :, 2].astype(np.float32)
        return V, hsv

    @staticmethod
    def put_luminance(base_bgr: np.ndarray, new_L: np.ndarray, method: str = "lab", base_conv: Optional[np.ndarray] = None) -> np.ndarray:
        new_L = np.clip(new_L, 0, 255).astype(np.uint8)
        if method == "lab":
            if base_conv is None:
                base_conv = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2LAB)
            lab = base_conv.copy()
            lab[:, :, 0] = new_L
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        elif method == "hsv":
            if base_conv is None:
                base_conv = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2HSV)
            hsv = base_conv.copy()
            hsv[:, :, 2] = new_L
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        else:  # gray
            return cv2.cvtColor(new_L, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def normalize_to_uint8(x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32)
        x -= x.min()
        denom = (x.max() - x.min())
        if denom < 1e-6:
            return np.zeros_like(x, dtype=np.uint8)
        x = x / denom
        return (x * 255.0).clip(0, 255).astype(np.uint8)

    @staticmethod
    def clahe(img8: np.ndarray, clip: float = 2.0) -> np.ndarray:
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
        return clahe.apply(img8)

    # -------------------- Alignment --------------------
    def align_als_to_white(self, als_bgr: np.ndarray, white_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Align als_bgr to white_bgr using ORB+RANSAC homography. Returns warped ALS and H."""
        orb = cv2.ORB_create(5000)
        g1 = cv2.cvtColor(als_bgr, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(white_bgr, cv2.COLOR_BGR2GRAY)
        kp1, des1 = orb.detectAndCompute(g1, None)
        kp2, des2 = orb.detectAndCompute(g2, None)
        if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
            raise RuntimeError("Not enough ORB features to compute alignment. Try larger max_size or clearer images.")
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(des1, des2, k=2)
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]
        if len(good) < 8:
            raise RuntimeError(f"Too few good ORB matches ({len(good)}) to estimate homography.")
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4.0)
        if H is None:
            raise RuntimeError("Homography estimation failed.")
        h, w = white_bgr.shape[:2]
        warped = cv2.warpPerspective(als_bgr, H, (w, h), flags=cv2.INTER_LINEAR)
        if self.cfg.debug_dir is not None:
            vis = cv2.drawMatches(g1, kp1, g2, kp2, [m for i, m in enumerate(good) if mask.ravel()[i] == 1], None,
                                  matchesMask=mask.ravel().tolist(), flags=2)
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
        b, g, r = cv2.split(als_bgr.astype(np.float32))
        als_like = 0.6 * b + 0.2 * g + 0.2 * r  # bias towards blue (approx 415 nm illum)
        return self.normalize_to_uint8(als_like).astype(np.float32)

    def lowpass(self, L: np.ndarray, sigma: float) -> np.ndarray:
        return cv2.GaussianBlur(L, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma).astype(np.float32)

    def highpass(self, L: np.ndarray, sigma: float) -> np.ndarray:
        blur = cv2.GaussianBlur(L, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma).astype(np.float32)
        hp = L - blur
        # Normalize
        hp_mean = float(np.mean(hp))
        hp_std = float(np.std(hp)) + 1e-6
        hp_norm = (hp - hp_mean) / hp_std
        # scale to 0..255
        hp_norm -= hp_norm.min()
        hp_norm /= (hp_norm.max() - hp_norm.min() + 1e-6)
        hp_norm *= 255.0
        return hp_norm

    def fuse(self, white_bgr: np.ndarray, als_bgr_aligned: np.ndarray) -> np.ndarray:
        # 1) Luminance from white
        Lw, lab_w = self.to_luminance_lab(white_bgr)
        # 2) ALS pseudo-luminance
        Lals = self.als_pseudo_luminance(als_bgr_aligned)
        # 3) Frequency split
        lp_w = self.lowpass(Lw, self.cfg.sigma_low)
        hp_als = self.highpass(Lals, self.cfg.sigma_high)
        if self.cfg.debug_dir is not None:
            cv2.imwrite(str(self.cfg.debug_dir / "03_L_white.jpg"), self.normalize_to_uint8(Lw))
            cv2.imwrite(str(self.cfg.debug_dir / "04_L_als_pseudo.jpg"), self.normalize_to_uint8(Lals))
            cv2.imwrite(str(self.cfg.debug_dir / "05_lowpass_white.jpg"), self.normalize_to_uint8(lp_w))
            cv2.imwrite(str(self.cfg.debug_dir / "06_highpass_als.jpg"), self.normalize_to_uint8(hp_als))
        # 4) Blend luminance
        fused_L = (self.cfg.w_low * lp_w + self.cfg.w_high * hp_als)
        fused_L = np.clip(fused_L, 0, 255).astype(np.uint8)
        fused_L = self.clahe(fused_L, clip=2.0)
        if self.cfg.debug_dir is not None:
            cv2.imwrite(str(self.cfg.debug_dir / "07_fused_L.jpg"), fused_L)
        # 5) Restore color
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

    engine = BruiseFusion(cfg)
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
