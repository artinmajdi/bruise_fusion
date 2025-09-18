"""
Streamlit Dashboard for Bruise Fusion
====================================

Interactive web interface for fusing white-light and ALS images using spatial-frequency blending.
Integrates with the BruiseFusion class from core2.py.

Usage:
    streamlit run dashboard.py

Author: AI Assistant
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

from src.utils import AdvancedBruiseFusion, FusionConfig, FusionMethod
from src.settings import logger

# TODO: Check if the deployed streamlit app is running: https://imagefuse.streamlit.app/


class BruiseFusionDashboard:
    """Enhanced dashboard for the bruise fusion application."""

    def __init__(self) -> None:
        """Initialize the dashboard component."""
        self.fusion_engine: Optional[AdvancedBruiseFusion] = None

        # Session state initialization
        if 'white_image' not in st.session_state:
            st.session_state.white_image = None
        if 'als_image' not in st.session_state:
            st.session_state.als_image = None
        if 'processing_complete' not in st.session_state:
            st.session_state.processing_complete = False
        if 'results' not in st.session_state:
            st.session_state.results = None
        if 'progress_status' not in st.session_state:
            st.session_state.progress_status = ""
        if 'progress_value' not in st.session_state:
            st.session_state.progress_value = 0.0

    def _apply_custom_css(self) -> None:
        """Apply custom CSS styling to the dashboard."""
        st.markdown("""
        <style>
        .main-header {
            text-align: center;
            padding: 1rem 0;
            border-bottom: 2px solid #f0f2f6;
            margin-bottom: 2rem;
        }
        .parameter-section {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }
        .result-container {
            border: 1px solid #e0e0e0;
            border-radius: 0.5rem;
            padding: 1rem;
            margin: 1rem 0;
        }
        </style>
        """, unsafe_allow_html=True)

    def _render_header(self) -> None:
        """Display the main header of the dashboard."""
        st.markdown('<div class="main-header">', unsafe_allow_html=True)
        st.title("🔬 Bruise Fusion Dashboard")
        st.markdown("**Fuse white-light and ALS images using spatial-frequency blending**")
        st.markdown('</div>', unsafe_allow_html=True)

    def _render_sidebar(self) -> Dict[str, Any]:
        """Render the sidebar with fusion parameters."""
        st.sidebar.header("Fusion Parameters")

        # Fusion method selection
        st.sidebar.subheader("🔬 Fusion Method")
        fusion_method: Tuple[str, FusionMethod] = st.sidebar.selectbox(
            "Select Fusion Algorithm",
            options=[
                ("Frequency Domain", FusionMethod.FREQUENCY_DOMAIN),
                ("Laplacian Pyramid", FusionMethod.LAPLACIAN_PYRAMID),
                ("Wavelet (DWT)", FusionMethod.WAVELET_DWT),
                ("Gradient-Based", FusionMethod.GRADIENT_BASED),
                ("Hybrid Adaptive", FusionMethod.HYBRID_ADAPTIVE)
            ],
            format_func=lambda x: x[0],
            index=0,
            help="Choose the fusion algorithm to use"
        )

        # Image processing parameters
        st.sidebar.subheader("📐 Image Processing")
        max_size: int = st.sidebar.slider("Max Image Size", 1000, 3000, 2200, 100,
                                    help="Resize longest side before processing")

        # Alignment method selection
        st.sidebar.subheader("🎯 Alignment Method")
        alignment_method: str = st.sidebar.selectbox(
            "Alignment Algorithm",
            options=[
                ("Auto-Select", "auto"),
                ("ORB + RANSAC (Fast)", "orb"),
                ("SIFT + RANSAC (High Quality)", "sift"),
                ("Multi-Scale (Robust)", "multiscale"),
                ("Hybrid SIFT+ECC (Premium)", "hybrid")
            ],
            format_func=lambda x: x[0],
            index=0,
            help="Choose alignment method: Auto-Select analyzes images and picks optimal method"
        )

        try_ecc: bool = st.sidebar.checkbox("Try ECC Refinement", False, help="Run ECC refinement after homography alignment")

        # Method-specific parameters
        method_value: FusionMethod = fusion_method[1]

        # Initialize all variables with default values first
        sigma_low         : float = 8.0
        sigma_high        : float = 2.0
        w_low             : float = 0.5
        w_high            : float = 1.0
        pyramid_levels    : int   = 5
        pyramid_sigma     : float = 1.0
        wavelet_type      : str   = 'db4'
        wavelet_levels    : int   = 4
        gradient_sigma    : float = 1.0
        edge_threshold    : float = 0.1
        local_window_size : int   = 15
        contrast_threshold: float = 0.2

        if method_value == FusionMethod.FREQUENCY_DOMAIN:
            st.sidebar.subheader("🌊 Frequency Parameters")
            sigma_low = st.sidebar.slider("Low-pass Sigma (White)", 1.0, 15.0, 8.0, 0.5, help="Gaussian sigma for low-pass filtering of white image")

            sigma_high = st.sidebar.slider("High-pass Sigma (ALS)", 1.0, 10.0, 2.0, 0.5, help="Gaussian sigma for high-pass filtering of ALS image")

            w_low = st.sidebar.slider("Low-pass Weight", 0.0, 1.0, 0.5, 0.1, help="Weight for low-pass white component")

            w_high = st.sidebar.slider("High-pass Weight", 0.0, 2.0, 1.0, 0.1, help="Weight for high-pass ALS component")

        elif method_value == FusionMethod.LAPLACIAN_PYRAMID:
            st.sidebar.subheader("🏔️ Pyramid Parameters")
            pyramid_levels = st.sidebar.slider("Pyramid Levels", 3, 8, 5, 1, help="Number of pyramid levels")
            pyramid_sigma = st.sidebar.slider("Pyramid Sigma", 0.5, 3.0, 1.0, 0.1, help="Gaussian sigma for pyramid construction")

        elif method_value == FusionMethod.WAVELET_DWT:
            st.sidebar.subheader("🌊 Wavelet Parameters")
            wavelet_type = st.sidebar.selectbox("Wavelet Type", ['db1', 'db4', 'db8', 'haar', 'bior2.2', 'coif2'], index=1, help="Type of wavelet to use")
            wavelet_levels = st.sidebar.slider("Decomposition Levels", 2, 6, 4, 1, help="Number of wavelet decomposition levels")

        elif method_value == FusionMethod.GRADIENT_BASED:
            st.sidebar.subheader("📈 Gradient Parameters")
            gradient_sigma = st.sidebar.slider("Gradient Sigma", 0.5, 3.0, 1.0, 0.1, help="Sigma for gradient computation")
            edge_threshold = st.sidebar.slider("Edge Threshold", 0.05, 0.5, 0.1, 0.01, help="Threshold for edge detection")

        elif method_value == FusionMethod.HYBRID_ADAPTIVE:
            st.sidebar.subheader("🔄 Adaptive Parameters")
            local_window_size = st.sidebar.slider("Local Window Size", 5, 25, 15, 2, help="Window size for local analysis")

            contrast_threshold = st.sidebar.slider("Contrast Threshold", 0.1, 0.5, 0.2, 0.05, help="Threshold for contrast-based decisions")

        # Color preservation method
        st.sidebar.subheader("🎨 Color Options")
        preserve_color: str = st.sidebar.selectbox(
            "Color Preservation Method",
            ["lab", "hsv", "gray"],
            index=0,
            help="Method to preserve color information"
        )

        # Quality metrics
        compute_metrics: bool = st.sidebar.checkbox("Compute Quality Metrics", True,
                                            help="Calculate SSIM, PSNR, and other quality metrics")

        # Debug options
        st.sidebar.subheader("🔧 Debug Options")
        save_debug: bool = st.sidebar.checkbox("Save Debug Images", False,
                                        help="Save intermediate processing steps")

        return {
            'method'            : method_value,
            'alignment_method'  : alignment_method[1],  # Extract the method code
            'max_size'          : max_size,
            'try_ecc'           : try_ecc,
            'sigma_low'         : sigma_low,
            'sigma_high'        : sigma_high,
            'w_low'             : w_low,
            'w_high'            : w_high,
            'pyramid_levels'    : pyramid_levels,
            'pyramid_sigma'     : pyramid_sigma,
            'wavelet_type'      : wavelet_type,
            'wavelet_levels'    : wavelet_levels,
            'gradient_sigma'    : gradient_sigma,
            'edge_threshold'    : edge_threshold,
            'local_window_size' : local_window_size,
            'contrast_threshold': contrast_threshold,
            'preserve_color'    : preserve_color,
            'compute_metrics'   : compute_metrics,
            'save_debug'        : save_debug
        }

    def _render_image_upload(self) -> Tuple[Optional[Any], Optional[Any]]:
        """Render the image upload interface."""
        col1: Any
        col2: Any
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📸 White-Light Image")
            white_file: Optional[Any] = st.file_uploader(
                "Upload white-light image",
                type=['jpg', 'jpeg', 'png', 'nef'],
                key="white"
            )

            if white_file is not None:
                white_image: Image.Image = Image.open(white_file)
                st.image(white_image, caption="White-Light Image", width='stretch')
                st.session_state.white_image = white_image

        with col2:
            st.subheader("🔵 ALS Image")
            als_file: Optional[Any] = st.file_uploader(
                "Upload ALS image",
                type=['jpg', 'jpeg', 'png', 'nef'],
                key="als"
            )

            if als_file is not None:
                als_image: Image.Image = Image.open(als_file)
                st.image(als_image, caption="ALS Image", width='stretch')
                st.session_state.als_image = als_image

        return white_file, als_file

    def _process_images(self, white_file: Optional[Any], als_file: Optional[Any], params: Dict[str, Any]) -> bool:
        """Process the uploaded images with the given parameters."""
        if white_file is None or als_file is None:
            st.error("Please upload both white-light and ALS images!")
            return False

        # try:
        with st.spinner("Processing images... This may take a few moments."):
            # Create temporary files for processing
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_dir_path: Path = Path(temp_dir)

                # Save uploaded files to temporary directory
                white_path: Path = temp_dir_path / "white_temp.jpg"
                als_path: Path = temp_dir_path / "als_temp.jpg"

                # Convert PIL images to OpenCV format and save
                white_cv: np.ndarray = cv2.cvtColor(np.array(st.session_state.white_image), cv2.COLOR_RGB2BGR)
                als_cv: np.ndarray = cv2.cvtColor(np.array(st.session_state.als_image), cv2.COLOR_RGB2BGR)

                cv2.imwrite(str(white_path), white_cv)
                cv2.imwrite(str(als_path), als_cv)

                # Configure fusion parameters
                debug_dir: Optional[Path] = temp_dir_path / "debug" if params['save_debug'] else None

                # Progress callback function
                def progress_callback(message: str, progress: float) -> None:
                    st.session_state.progress_value = progress
                    st.session_state.progress_status = message

                config: FusionConfig = FusionConfig(
                    method             = params['method'],
                    max_size           = params['max_size'],
                    alignment_method   = params['alignment_method'],
                    try_ecc            = params['try_ecc'],
                    sigma_low          = params['sigma_low'],
                    sigma_high         = params['sigma_high'],
                    w_low              = params['w_low'],
                    w_high             = params['w_high'],
                    pyramid_levels     = params['pyramid_levels'],
                    pyramid_sigma      = params['pyramid_sigma'],
                    wavelet_type       = params['wavelet_type'],
                    wavelet_levels     = params['wavelet_levels'],
                    gradient_sigma     = params['gradient_sigma'],
                    edge_threshold     = params['edge_threshold'],
                    local_window_size  = params['local_window_size'],
                    contrast_threshold = params['contrast_threshold'],
                    preserve_color     = params['preserve_color'],
                    compute_metrics    = params['compute_metrics'],
                    debug_dir          = debug_dir,
                    progress_callback  = progress_callback
                )

                # Initialize progress
                st.session_state.progress_value = 0.0
                st.session_state.progress_status = "Starting fusion process..."

                # Create progress bar
                progress_bar: Any = st.progress(0.0)
                status_text: Any = st.empty()

                # Run fusion
                self.fusion_engine = AdvancedBruiseFusion(config)
                white_resized: np.ndarray
                als_aligned: np.ndarray
                fused_result: np.ndarray
                white_resized, als_aligned, fused_result = self.fusion_engine.run(white_path, als_path)

                # Update progress to completion
                progress_bar.progress(1.0)
                status_text.text("✅ Fusion completed!")

                # Store results in session state
                st.session_state.results = {
                    'white_resized': cv2.cvtColor(white_resized, cv2.COLOR_BGR2RGB),
                    'als_aligned'  : cv2.cvtColor(als_aligned, cv2.COLOR_BGR2RGB),
                    'fused_result' : cv2.cvtColor(fused_result, cv2.COLOR_BGR2RGB),
                    'debug_dir'    : debug_dir,
                    'params'       : params
                }
                st.session_state.processing_complete = True

        st.success("✅ Processing completed successfully!")
        return True

        # except Exception as e:
        #     st.error(f"❌ An error occurred during processing: {str(e)}")
        #     st.exception(e)
        #     return False

    def _display_results(self) -> None:
        """Display the fusion results and quality metrics."""
        if 'results' not in st.session_state or not st.session_state.get('processing_complete', False):
            return

        result: Dict[str, Any] = st.session_state.results

        st.header("🎯 Fusion Results")

        # Display main result
        col1: Any
        col2: Any
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Fused Image")
            # Convert BGR to RGB for display
            result_rgb: np.ndarray = result['fused_result']
            st.image(result_rgb, caption="Fused Result", width='stretch')

        with col2:
            # Display quality metrics if available
            if 'metrics' in result and result['metrics']:
                st.subheader("📊 Quality Metrics")
                metrics: Dict[str, Any] = result['metrics']

                if 'ssim' in metrics:
                    st.metric("SSIM", f"{metrics['ssim']:.4f}")
                if 'psnr' in metrics:
                    st.metric("PSNR", f"{metrics['psnr']:.2f} dB")
                if 'mse' in metrics:
                    st.metric("MSE", f"{metrics['mse']:.2f}")
                if 'entropy' in metrics:
                    st.metric("Entropy", f"{metrics['entropy']:.3f}")
                if 'gradient_magnitude' in metrics:
                    st.metric("Gradient Mag.", f"{metrics['gradient_magnitude']:.2f}")

            # Download button for result
            st.subheader("💾 Download")
            # Convert RGB back to BGR for encoding
            result_bgr: np.ndarray = cv2.cvtColor(result['fused_result'], cv2.COLOR_RGB2BGR)
            result_bytes: bytes = cv2.imencode('.jpg', result_bgr)[1].tobytes()
            st.download_button(
                label     = "Download Fused Image",
                data      = result_bytes,
                file_name = "fused_result.jpg",
                mime      = "image/jpeg"
            )

        # Display input images for comparison
        st.subheader("📷 Input Images")
        col1, col2 = st.columns(2)

        with col1:
            if 'white_resized' in result:
                white_rgb: np.ndarray = result['white_resized']
                st.image(white_rgb, caption="White-light (Resized)", width='stretch')

        with col2:
            if 'als_aligned' in result:
                als_rgb: np.ndarray = result['als_aligned']
                st.image(als_rgb, caption="ALS (Aligned)", width='stretch')

        # Debug images section
        if st.session_state.get('debug_dir') and os.path.exists(st.session_state.debug_dir):
            with st.expander("🔧 Debug Images", expanded=False):
                debug_files: List[str] = [f for f in os.listdir(st.session_state.debug_dir)
                                if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

                if debug_files:
                    debug_cols: List[Any] = st.columns(min(3, len(debug_files)))
                    for i, debug_file in enumerate(debug_files):
                        with debug_cols[i % 3]:
                            debug_path: str = os.path.join(st.session_state.debug_dir, debug_file)
                            debug_img: Optional[np.ndarray] = cv2.imread(debug_path)
                            if debug_img is not None:
                                debug_rgb: np.ndarray = cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB)
                                st.image(debug_rgb, caption=debug_file, width='stretch')
                else:
                    st.info("No debug images available.")

        # Processing parameters
        if 'results' in st.session_state and 'params' in st.session_state.results:
            with st.expander("⚙️ Processing Parameters", expanded=False):
                params: Dict[str, Any] = st.session_state.results['params']

                # Display method and key parameters
                st.write(f"**Fusion Method:** {params['method'].value}")
                st.write(f"**Max Size:** {params['max_size']}")
                st.write(f"**ECC Refinement:** {params['try_ecc']}")
                st.write(f"**Color Preservation:** {params['preserve_color']}")

                # Method-specific parameters
                if params['method'] == FusionMethod.FREQUENCY_DOMAIN:
                    st.write(f"**Sigma Low:** {params['sigma_low']}")
                    st.write(f"**Sigma High:** {params['sigma_high']}")
                    st.write(f"**Weight Low:** {params['w_low']}")
                    st.write(f"**Weight High:** {params['w_high']}")
                elif params['method'] == FusionMethod.LAPLACIAN_PYRAMID:
                    st.write(f"**Pyramid Levels:** {params['pyramid_levels']}")
                    st.write(f"**Pyramid Sigma:** {params['pyramid_sigma']}")
                elif params['method'] == FusionMethod.WAVELET_DWT:
                    st.write(f"**Wavelet Type:** {params['wavelet_type']}")
                    st.write(f"**Wavelet Levels:** {params['wavelet_levels']}")
                elif params['method'] == FusionMethod.GRADIENT_BASED:
                    st.write(f"**Gradient Sigma:** {params['gradient_sigma']}")
                    st.write(f"**Edge Threshold:** {params['edge_threshold']}")
                elif params['method'] == FusionMethod.HYBRID_ADAPTIVE:
                    st.write(f"**Window Size:** {params['local_window_size']}")
                    st.write(f"**Contrast Threshold:** {params['contrast_threshold']}")

    def _render_about_section(self) -> None:
        """Render the about section with tool information."""
        st.header("ℹ️ About This Tool")

        st.markdown("""
        ### Advanced Bruise Fusion System

        This tool provides **training-free image fusion** for combining white-light and ALS (Alternate Light Source)
        images to enhance bruise visibility and forensic analysis.

        #### 🔬 Available Fusion Methods:

        **1. Frequency Domain Fusion**
        - Combines low-frequency components from white-light images with high-frequency details from ALS images
        - Best for: General purpose fusion with good detail preservation

        **2. Laplacian Pyramid Fusion**
        - Multi-scale decomposition using Gaussian and Laplacian pyramids
        - Best for: Sharp edge preservation and multi-scale feature fusion

        **3. Wavelet (DWT) Fusion**
        - Discrete Wavelet Transform-based fusion with various wavelet types
        - Best for: Texture preservation and noise reduction

        **4. Gradient-Based Fusion**
        - Focuses on gradient magnitude and edge information
        - Best for: Edge enhancement and structural detail preservation

        **5. Hybrid Adaptive Fusion**
        - Combines multiple methods based on local image characteristics
        - Best for: Optimal results across diverse image regions

        #### 📊 Quality Metrics:
        - **SSIM**: Structural Similarity Index (higher is better)
        - **PSNR**: Peak Signal-to-Noise Ratio in dB (higher is better)
        - **MSE**: Mean Squared Error (lower is better)
        - **Entropy**: Information content measure
        - **Gradient Magnitude**: Edge strength measure

        #### 🎨 Color Preservation:
        - **LAB**: Preserves luminance while maintaining color accuracy
        - **HSV**: Maintains hue and saturation information
        - **Gray**: Grayscale processing for maximum detail

        #### 🔧 Advanced Features:
        - Real-time progress feedback during processing
        - Debug image output for analysis
        - Multiple image format support
        - ECC-based image alignment refinement
        - Automatic quality assessment
        """)

        with st.expander("📋 Parameter Guidelines", expanded=False):
            st.markdown("""
            #### Frequency Domain Parameters:
            - **Sigma Low/High**: Controls the frequency separation (lower = more detail)
            - **Weights**: Balance between white-light and ALS contributions

            #### Pyramid Parameters:
            - **Levels**: More levels = finer detail analysis (3-8 recommended)
            - **Sigma**: Gaussian blur amount for pyramid construction

            #### Wavelet Parameters:
            - **Type**: Different wavelets for different image characteristics
            - **Levels**: Decomposition depth (2-6 recommended)

            #### Gradient Parameters:
            - **Sigma**: Smoothing before gradient computation
            - **Threshold**: Sensitivity to edge detection

            #### Adaptive Parameters:
            - **Window Size**: Local analysis region size
            - **Contrast Threshold**: Decision boundary for method selection
            """)

        st.info("💡 **Tip**: Start with Frequency Domain fusion for general use, then experiment with other methods for specific image characteristics.")

        st.markdown("---")
        st.markdown("**Developed for forensic image analysis and bruise documentation.**")

    def run(self) -> None:
        """Render the dashboard."""
        # Set up the Streamlit page configuration
        st.set_page_config(
            page_title="Bruise Fusion Dashboard",
            page_icon="🔬",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        # Apply custom CSS
        self._apply_custom_css()

        # Display header
        self._render_header()

        # Sidebar navigation and parameters
        params: Dict[str, Any] = self._render_sidebar()

        # Main content area - image upload
        white_file: Optional[Any]
        als_file: Optional[Any]
        white_file, als_file = self._render_image_upload()

        # Process button
        if st.button("🚀 Process Images", type="primary", width='stretch'):
            self._process_images(white_file, als_file, params)

        # Display results if processing is complete
        self._display_results()

        # Information section
        self._render_about_section()


def main() -> None:
    """Entry point for the dashboard application."""
    import subprocess
    import sys
    import os

    # Get the path to the current dashboard.py file
    dashboard_path = os.path.abspath(__file__)

    # Launch streamlit with the dashboard file
    subprocess.run([sys.executable, "-m", "streamlit", "run", dashboard_path])


if __name__ == "__main__":
    # Create and render the dashboard
    dashboard = BruiseFusionDashboard()
    dashboard.run()