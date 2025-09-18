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
import io
import logging
from typing import Optional

# Import the BruiseFusion class
from core import BruiseFusion, FusionConfig


class BruiseFusionDashboard:
    """Enhanced dashboard for the bruise fusion application."""

    def __init__(self):
        """Initialize the dashboard component."""
        self.fusion_engine: Optional[BruiseFusion] = None

        # Session state initialization
        if 'white_image' not in st.session_state:
            st.session_state.white_image = None
        if 'als_image' not in st.session_state:
            st.session_state.als_image = None
        if 'processing_complete' not in st.session_state:
            st.session_state.processing_complete = False
        if 'results' not in st.session_state:
            st.session_state.results = None

    def _apply_custom_css(self):
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

    def _render_header(self):
        """Display the main header of the dashboard."""
        st.markdown('<div class="main-header">', unsafe_allow_html=True)
        st.title("🔬 Bruise Fusion Dashboard")
        st.markdown("**Fuse white-light and ALS images using spatial-frequency blending**")
        st.markdown('</div>', unsafe_allow_html=True)

    def _render_sidebar(self):
        """Render the sidebar with fusion parameters."""
        st.sidebar.header("Fusion Parameters")

        # Image processing parameters
        max_size = st.sidebar.slider("Max Image Size", 1000, 3000, 2200, 100,
                                    help="Resize longest side before processing")

        try_ecc = st.sidebar.checkbox("Try ECC Refinement", False,
                                     help="Run ECC refinement after homography alignment")

        # Frequency fusion parameters
        st.sidebar.subheader("Frequency Parameters")
        sigma_low = st.sidebar.slider("Low-pass Sigma (White)", 1.0, 15.0, 6.0, 0.5,
                                     help="Gaussian sigma for low-pass filtering of white image")

        sigma_high = st.sidebar.slider("High-pass Sigma (ALS)", 1.0, 10.0, 3.0, 0.5,
                                      help="Gaussian sigma for high-pass filtering of ALS image")

        w_low = st.sidebar.slider("Low-pass Weight", 0.0, 1.0, 0.6, 0.1,
                                 help="Weight for low-pass white component")

        w_high = st.sidebar.slider("High-pass Weight", 0.0, 2.0, 0.8, 0.1,
                                  help="Weight for high-pass ALS component")

        # Color preservation method
        preserve_color = st.sidebar.selectbox(
            "Color Preservation Method",
            ["lab", "hsv", "gray"],
            index=0,
            help="Method to preserve color information"
        )

        # Debug options
        st.sidebar.subheader("Debug Options")
        save_debug = st.sidebar.checkbox("Save Debug Images", False,
                                        help="Save intermediate processing steps")

        return {
            'max_size': max_size,
            'try_ecc': try_ecc,
            'sigma_low': sigma_low,
            'sigma_high': sigma_high,
            'w_low': w_low,
            'w_high': w_high,
            'preserve_color': preserve_color,
            'save_debug': save_debug
        }

    def _render_image_upload(self):
        """Render the image upload interface."""
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📸 White-Light Image")
            white_file = st.file_uploader(
                "Upload white-light image",
                type=['jpg', 'jpeg', 'png'],
                key="white"
            )

            if white_file is not None:
                white_image = Image.open(white_file)
                st.image(white_image, caption="White-Light Image", width='stretch')
                st.session_state.white_image = white_image

        with col2:
            st.subheader("🔵 ALS Image")
            als_file = st.file_uploader(
                "Upload ALS image",
                type=['jpg', 'jpeg', 'png'],
                key="als"
            )

            if als_file is not None:
                als_image = Image.open(als_file)
                st.image(als_image, caption="ALS Image", width='stretch')
                st.session_state.als_image = als_image

        return white_file, als_file

    def _process_images(self, white_file, als_file, params):
        """Process the uploaded images with the given parameters."""
        if white_file is None or als_file is None:
            st.error("Please upload both white-light and ALS images!")
            return False

        try:
            with st.spinner("Processing images... This may take a few moments."):
                # Create temporary files for processing
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_dir_path = Path(temp_dir)

                    # Save uploaded files to temporary directory
                    white_path = temp_dir_path / "white_temp.jpg"
                    als_path = temp_dir_path / "als_temp.jpg"

                    # Convert PIL images to OpenCV format and save
                    white_cv = cv2.cvtColor(np.array(st.session_state.white_image), cv2.COLOR_RGB2BGR)
                    als_cv = cv2.cvtColor(np.array(st.session_state.als_image), cv2.COLOR_RGB2BGR)

                    cv2.imwrite(str(white_path), white_cv)
                    cv2.imwrite(str(als_path), als_cv)

                    # Configure fusion parameters
                    debug_dir = temp_dir_path / "debug" if params['save_debug'] else None

                    config = FusionConfig(
                        max_size=params['max_size'],
                        try_ecc=params['try_ecc'],
                        sigma_low=params['sigma_low'],
                        sigma_high=params['sigma_high'],
                        w_low=params['w_low'],
                        w_high=params['w_high'],
                        preserve_color=params['preserve_color'],
                        debug_dir=debug_dir
                    )

                    # Run fusion
                    self.fusion_engine = BruiseFusion(config)
                    white_resized, als_aligned, fused_result = self.fusion_engine.run(white_path, als_path)

                    # Store results in session state
                    st.session_state.results = {
                        'white_resized': cv2.cvtColor(white_resized, cv2.COLOR_BGR2RGB),
                        'als_aligned': cv2.cvtColor(als_aligned, cv2.COLOR_BGR2RGB),
                        'fused_result': cv2.cvtColor(fused_result, cv2.COLOR_BGR2RGB),
                        'debug_dir': debug_dir,
                        'params': params
                    }
                    st.session_state.processing_complete = True

            st.success("✅ Processing completed successfully!")
            return True

        except Exception as e:
            st.error(f"❌ An error occurred during processing: {str(e)}")
            st.exception(e)
            return False

    def _display_results(self):
        """Display the processing results."""
        if not st.session_state.processing_complete or st.session_state.results is None:
            return

        results = st.session_state.results

        # Display results in columns
        st.subheader("📊 Results")

        result_col1, result_col2, result_col3 = st.columns(3)

        with result_col1:
            st.image(results['white_resized'], caption="White-Light (Resized)", width='stretch')

        with result_col2:
            st.image(results['als_aligned'], caption="ALS (Aligned)", width='stretch')

        with result_col3:
            st.image(results['fused_result'], caption="Fused Result", width='stretch')

        # Download button for fused result
        st.subheader("💾 Download Result")

        # Convert to bytes for download
        fused_pil = Image.fromarray(results['fused_result'])
        buf = io.BytesIO()
        fused_pil.save(buf, format='JPEG', quality=95)
        buf.seek(0)

        st.download_button(
            label="📥 Download Fused Image",
            data=buf.getvalue(),
            file_name="fused_result.jpg",
            mime="image/jpeg",
            use_container_width=True
        )

        # Display debug images if enabled
        if results['params']['save_debug'] and results['debug_dir'] and results['debug_dir'].exists():
            st.subheader("🔍 Debug Images")

            debug_files = list(results['debug_dir'].glob("*.jpg"))
            if debug_files:
                debug_cols = st.columns(min(3, len(debug_files)))

                for i, debug_file in enumerate(sorted(debug_files)):
                    col_idx = i % len(debug_cols)
                    with debug_cols[col_idx]:
                        debug_img = Image.open(debug_file)
                        st.image(debug_img, caption=debug_file.name, width='stretch')

        # Display processing parameters used
        with st.expander("ℹ️ Processing Parameters Used"):
            st.json(results['params'])

    def _render_about_section(self):
        """Render the about section with information about the tool."""
        with st.expander("ℹ️ About This Tool"):
            st.markdown("""
            ### How it works:

            1. **Alignment**: The ALS image is aligned to the white-light image using ORB feature matching and RANSAC homography
            2. **Frequency Separation**:
               - White-light image provides low-frequency structure (smooth variations)
               - ALS image provides high-frequency details (fine textures)
            3. **ALS Processing**: Creates pseudo-luminance biased toward blue channel (simulating ~415nm illumination)
            4. **Fusion**: Combines low-pass white structure with high-pass ALS details
            5. **Color Restoration**: Maintains natural color tones from the white-light image

            ### Parameters:
            - **Sigma Low/High**: Control the frequency separation (lower = more detail preserved)
            - **Weights**: Balance between white-light structure and ALS details
            - **Color Method**: How to preserve color information (LAB usually works best)
            - **ECC Refinement**: Additional alignment step for better registration
            """)

    def run(self):
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
        params = self._render_sidebar()

        # Main content area - image upload
        white_file, als_file = self._render_image_upload()

        # Process button
        if st.button("🚀 Process Images", type="primary", width='stretch'):
            self._process_images(white_file, als_file, params)

        # Display results if processing is complete
        self._display_results()

        # Information section
        self._render_about_section()


def main():
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