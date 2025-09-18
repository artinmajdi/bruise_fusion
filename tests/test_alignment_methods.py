#!/usr/bin/env python3
"""
Comprehensive Evaluation Script for Advanced Image Alignment Methods
===================================================================

This script evaluates the performance of different alignment techniques
implemented in the ImageAligner class, providing detailed metrics and
comparisons to help select the best method for specific use cases.

Usage:
    python test_alignment_methods.py --input_dir /path/to/test/images
    python test_alignment_methods.py --single_test image1.jpg image2.jpg
"""

import argparse
import cv2
import numpy as np
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
from dataclasses import dataclass

from src.utils import ImageAligner, FusionConfig


@dataclass
class AlignmentTestResult:
    """Results from testing an alignment method."""
    method: str
    execution_time: float
    quality_metrics: Dict[str, Any]
    success: bool
    error_message: str = ""


class AlignmentEvaluator:
    """Comprehensive evaluator for image alignment methods."""
    
    def __init__(self, debug_dir: Path = None):
        self.debug_dir = debug_dir or Path("alignment_test_results")
        self.debug_dir.mkdir(exist_ok=True)
        
        # Initialize aligner with debug enabled
        self.config = FusionConfig(debug_dir=self.debug_dir)
        self.aligner = ImageAligner(self.config)
        
        # Test methods to evaluate
        self.test_methods = ['orb', 'sift', 'multiscale', 'hybrid']
        
    def load_test_images(self, img1_path: str, img2_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load and validate test images."""
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        if img1 is None or img2 is None:
            raise ValueError(f"Could not load images: {img1_path}, {img2_path}")
            
        return img1, img2
    
    def compute_alignment_metrics(self, img1: np.ndarray, img2: np.ndarray, 
                                aligned: np.ndarray) -> Dict[str, float]:
        """Compute comprehensive alignment quality metrics."""
        from skimage.metrics import structural_similarity as ssim
        from skimage.metrics import peak_signal_noise_ratio as psnr
        
        # Convert to grayscale for some metrics
        gray1 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        gray_aligned = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
        
        metrics = {}
        
        try:
            # SSIM - Structural Similarity Index
            metrics['ssim'] = ssim(gray1, gray_aligned, data_range=255)
            
            # PSNR - Peak Signal-to-Noise Ratio
            metrics['psnr'] = psnr(gray1, gray_aligned, data_range=255)
            
            # Mean Squared Error
            mse = np.mean((gray1.astype(float) - gray_aligned.astype(float)) ** 2)
            metrics['mse'] = mse
            
            # Root Mean Squared Error
            metrics['rmse'] = np.sqrt(mse)
            
            # Normalized Cross Correlation
            ncc = cv2.matchTemplate(gray1, gray_aligned, cv2.TM_CCORR_NORMED)[0, 0]
            metrics['ncc'] = ncc
            
            # Mutual Information (simplified approximation)
            hist_2d, _, _ = np.histogram2d(gray1.ravel(), gray_aligned.ravel(), bins=256)
            pxy = hist_2d / float(np.sum(hist_2d))
            px = np.sum(pxy, axis=1)
            py = np.sum(pxy, axis=0)
            
            # Avoid log(0)
            px_py = px[:, None] * py[None, :]
            nzs = pxy > 0
            mi = np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))
            metrics['mutual_info'] = mi
            
        except Exception as e:
            print(f"Warning: Could not compute some metrics: {e}")
            
        return metrics
    
    def test_alignment_method(self, img1: np.ndarray, img2: np.ndarray, 
                            method: str) -> AlignmentTestResult:
        """Test a specific alignment method."""
        print(f"Testing {method} alignment...")
        
        start_time = time.time()
        
        try:
            # Force the aligner to use specific method
            original_method = getattr(self.aligner, '_force_method', None)
            self.aligner._force_method = method
            
            # Perform alignment
            aligned = self.aligner.align_als_to_white(img1, img2)
            
            execution_time = time.time() - start_time
            
            # Compute quality metrics
            quality_metrics = self.compute_alignment_metrics(img1, img2, aligned)
            
            # Get aligner's internal metrics
            aligner_metrics = self.aligner.get_alignment_quality_report()
            quality_metrics.update(aligner_metrics['metrics'])
            
            # Save debug images
            method_dir = self.debug_dir / method
            method_dir.mkdir(exist_ok=True)
            cv2.imwrite(str(method_dir / "aligned_result.jpg"), aligned)
            
            # Reset force method
            if hasattr(self.aligner, '_force_method'):
                delattr(self.aligner, '_force_method')
            
            return AlignmentTestResult(
                method=method,
                execution_time=execution_time,
                quality_metrics=quality_metrics,
                success=True
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"Error testing {method}: {e}")
            
            return AlignmentTestResult(
                method=method,
                execution_time=execution_time,
                quality_metrics={},
                success=False,
                error_message=str(e)
            )
    
    def run_comprehensive_test(self, img1_path: str, img2_path: str) -> List[AlignmentTestResult]:
        """Run comprehensive test on all alignment methods."""
        print(f"Loading test images: {img1_path}, {img2_path}")
        
        try:
            img1, img2 = self.load_test_images(img1_path, img2_path)
        except Exception as e:
            print(f"Error loading images: {e}")
            return []
        
        # Save original images for reference
        cv2.imwrite(str(self.debug_dir / "original_img1.jpg"), img1)
        cv2.imwrite(str(self.debug_dir / "original_img2.jpg"), img2)
        
        results = []
        
        for method in self.test_methods:
            result = self.test_alignment_method(img1, img2, method)
            results.append(result)
            
            # Brief summary
            if result.success:
                ssim_val = result.quality_metrics.get('ssim', 0)
                time_val = result.execution_time
                print(f"  {method}: SSIM={ssim_val:.3f}, Time={time_val:.2f}s")
            else:
                print(f"  {method}: FAILED - {result.error_message}")
        
        return results
    
    def generate_report(self, results: List[AlignmentTestResult], 
                       output_path: str = None) -> Dict[str, Any]:
        """Generate comprehensive evaluation report."""
        if not output_path:
            output_path = str(self.debug_dir / "alignment_evaluation_report.json")
        
        report = {
            'test_summary': {
                'total_methods': len(results),
                'successful_methods': sum(1 for r in results if r.success),
                'failed_methods': sum(1 for r in results if not r.success)
            },
            'method_results': {},
            'rankings': {},
            'recommendations': []
        }
        
        successful_results = [r for r in results if r.success]
        
        if not successful_results:
            report['recommendations'].append("No methods succeeded - check input images")
            return report
        
        # Detailed results for each method
        for result in results:
            report['method_results'][result.method] = {
                'success': result.success,
                'execution_time': result.execution_time,
                'quality_metrics': result.quality_metrics,
                'error_message': result.error_message
            }
        
        # Rankings by different criteria
        if successful_results:
            # Rank by SSIM (higher is better)
            ssim_ranking = sorted(successful_results, 
                                key=lambda x: x.quality_metrics.get('ssim', 0), 
                                reverse=True)
            report['rankings']['by_ssim'] = [r.method for r in ssim_ranking]
            
            # Rank by execution time (lower is better)
            time_ranking = sorted(successful_results, 
                                key=lambda x: x.execution_time)
            report['rankings']['by_speed'] = [r.method for r in time_ranking]
            
            # Rank by PSNR (higher is better)
            psnr_ranking = sorted(successful_results,
                                key=lambda x: x.quality_metrics.get('psnr', 0),
                                reverse=True)
            report['rankings']['by_psnr'] = [r.method for r in psnr_ranking]
            
            # Overall recommendation based on balanced criteria
            best_method = self._select_best_method(successful_results)
            report['recommendations'].append(f"Best overall method: {best_method}")
            
            # Performance insights
            fastest_method = time_ranking[0].method
            highest_quality = ssim_ranking[0].method
            
            report['recommendations'].append(f"Fastest method: {fastest_method}")
            report['recommendations'].append(f"Highest quality: {highest_quality}")
            
            if fastest_method != highest_quality:
                report['recommendations'].append(
                    "Consider speed vs quality trade-off based on your use case"
                )
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nDetailed report saved to: {output_path}")
        return report
    
    def _select_best_method(self, results: List[AlignmentTestResult]) -> str:
        """Select best method based on balanced criteria."""
        scores = {}
        
        for result in results:
            method = result.method
            metrics = result.quality_metrics
            
            # Normalized scoring (0-1 scale)
            ssim_score = metrics.get('ssim', 0)
            psnr_score = min(metrics.get('psnr', 0) / 50.0, 1.0)  # Normalize PSNR
            speed_score = max(0, 1.0 - result.execution_time / 10.0)  # Penalize slow methods
            
            # Weighted combination
            total_score = (ssim_score * 0.4 + psnr_score * 0.4 + speed_score * 0.2)
            scores[method] = total_score
        
        return max(scores.keys(), key=lambda k: scores[k])
    
    def create_visual_comparison(self, results: List[AlignmentTestResult]):
        """Create visual comparison plots."""
        successful_results = [r for r in results if r.success]
        
        if len(successful_results) < 2:
            print("Not enough successful results for visual comparison")
            return
        
        methods = [r.method for r in successful_results]
        ssim_values = [r.quality_metrics.get('ssim', 0) for r in successful_results]
        psnr_values = [r.quality_metrics.get('psnr', 0) for r in successful_results]
        times = [r.execution_time for r in successful_results]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # SSIM comparison
        ax1.bar(methods, ssim_values, color='skyblue')
        ax1.set_title('SSIM Comparison (Higher is Better)')
        ax1.set_ylabel('SSIM')
        ax1.tick_params(axis='x', rotation=45)
        
        # PSNR comparison
        ax2.bar(methods, psnr_values, color='lightgreen')
        ax2.set_title('PSNR Comparison (Higher is Better)')
        ax2.set_ylabel('PSNR (dB)')
        ax2.tick_params(axis='x', rotation=45)
        
        # Execution time comparison
        ax3.bar(methods, times, color='salmon')
        ax3.set_title('Execution Time (Lower is Better)')
        ax3.set_ylabel('Time (seconds)')
        ax3.tick_params(axis='x', rotation=45)
        
        # Quality vs Speed scatter plot
        ax4.scatter(times, ssim_values, s=100, alpha=0.7)
        for i, method in enumerate(methods):
            ax4.annotate(method, (times[i], ssim_values[i]), 
                        xytext=(5, 5), textcoords='offset points')
        ax4.set_xlabel('Execution Time (seconds)')
        ax4.set_ylabel('SSIM')
        ax4.set_title('Quality vs Speed Trade-off')
        
        plt.tight_layout()
        plt.savefig(str(self.debug_dir / "alignment_comparison.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visual comparison saved to: {self.debug_dir / 'alignment_comparison.png'}")


def main():
    parser = argparse.ArgumentParser(description='Test image alignment methods')
    parser.add_argument('--img1', required=True, help='Path to first image (ALS)')
    parser.add_argument('--img2', required=True, help='Path to second image (white light)')
    parser.add_argument('--output_dir', default='alignment_test_results', 
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = AlignmentEvaluator(debug_dir=Path(args.output_dir))
    
    # Run comprehensive test
    print("Starting comprehensive alignment method evaluation...")
    results = evaluator.run_comprehensive_test(args.img1, args.img2)
    
    if not results:
        print("No results to analyze")
        return
    
    # Generate report
    print("\nGenerating evaluation report...")
    report = evaluator.generate_report(results)
    
    # Create visual comparison
    print("Creating visual comparison...")
    evaluator.create_visual_comparison(results)
    
    # Print summary
    print("\n" + "="*60)
    print("ALIGNMENT METHOD EVALUATION SUMMARY")
    print("="*60)
    
    successful_results = [r for r in results if r.success]
    if successful_results:
        print(f"Successfully tested {len(successful_results)} methods:")
        
        for result in successful_results:
            metrics = result.quality_metrics
            print(f"\n{result.method.upper()}:")
            print(f"  SSIM: {metrics.get('ssim', 0):.4f}")
            print(f"  PSNR: {metrics.get('psnr', 0):.2f} dB")
            print(f"  Time: {result.execution_time:.2f} seconds")
        
        print(f"\nRecommendations:")
        for rec in report['recommendations']:
            print(f"  • {rec}")
    else:
        print("No methods succeeded. Check your input images and try again.")
    
    print(f"\nDetailed results saved in: {args.output_dir}")


if __name__ == "__main__":
    main()