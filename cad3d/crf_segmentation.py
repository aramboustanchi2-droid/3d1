"""
Conditional Random Fields (CRF) for CAD Segmentation
Markov Random Fields برای تقسیم‌بندی دقیق نقشه‌های CAD

این ماژول از CRF برای segmentation دقیق استفاده می‌کند:
- مرزبندی دیوارها
- تشخیص خطوط دقیق
- تفکیک عناصر مختلف
- بهبود خروجی CNN/U-Net

استفاده برای:
- ساختمان‌سازی (Building Construction)
- پل‌سازی (Bridge Engineering)
- جاده‌سازی (Road Construction)
- سدسازی (Dam Construction)
- تونل‌سازی (Tunnel Construction)
- کارخانه‌های صنعتی (Industrial Factories)
- ماشین‌سازی (Machinery Manufacturing)
- هر صنعتی با نیاز به نقشه فنی

Theory:
CRF (Conditional Random Field) یک مدل احتمالاتی است که:
1. روابط بین پیکسل‌های مجاور را در نظر می‌گیرد
2. از context استفاده می‌کند (پیکسل‌های اطراف)
3. مرزها را صاف و دقیق می‌کند
4. خروجی CNN را بهبود می‌دهد

مزایا:
- Boundary refinement: مرزهای دقیق‌تر
- Spatial consistency: consistency فضایی بالا
- Context-aware: از context استفاده می‌کند
- Post-processing: بهبود نتایج CNN/U-Net
"""

from typing import Optional, Tuple, Dict, Any, List
import numpy as np
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️  PyTorch not available. CRF will not work.")

try:
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral, create_pairwise_gaussian
    PYDENSECRF_AVAILABLE = True
except ImportError:
    PYDENSECRF_AVAILABLE = False
    print("⚠️  pydensecrf not available. Install: pip install git+https://github.com/lucasb-eyer/pydensecrf.git")


if TORCH_AVAILABLE:
    
    class LinearChainCRF(nn.Module):
        """
        Linear-Chain CRF for sequential labeling
        
        مناسب برای:
        - تشخیص خطوط متوالی
        - دنبال کردن مرزها
        - Sequence labeling
        """
        
        def __init__(self, num_tags: int):
            super().__init__()
            
            self.num_tags = num_tags
            
            # Transition matrix: [num_tags, num_tags]
            # transitions[i, j] = score of transitioning from tag i to tag j
            self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
            
            # Start and end transitions
            self.start_transitions = nn.Parameter(torch.randn(num_tags))
            self.end_transitions = nn.Parameter(torch.randn(num_tags))
        
        def forward(
            self,
            emissions: Tensor,  # [batch, seq_len, num_tags]
            tags: Tensor,  # [batch, seq_len]
            mask: Optional[Tensor] = None  # [batch, seq_len]
        ) -> Tensor:
            """
            Compute negative log-likelihood loss
            
            Args:
                emissions: Emission scores from neural network
                tags: Ground truth tags
                mask: Mask for variable length sequences
            
            Returns:
                Negative log-likelihood loss
            """
            if mask is None:
                mask = torch.ones_like(tags, dtype=torch.bool)
            
            # Compute log partition function (forward algorithm)
            log_partition = self._forward_algorithm(emissions, mask)
            
            # Compute score of gold sequence
            gold_score = self._score_sequence(emissions, tags, mask)
            
            # NLL loss
            return (log_partition - gold_score).mean()
        
        def decode(
            self,
            emissions: Tensor,  # [batch, seq_len, num_tags]
            mask: Optional[Tensor] = None  # [batch, seq_len]
        ) -> List[List[int]]:
            """
            Viterbi decoding to find best tag sequence
            
            Args:
                emissions: Emission scores
                mask: Mask for variable length
            
            Returns:
                List of best tag sequences
            """
            if mask is None:
                mask = torch.ones(emissions.shape[:2], dtype=torch.bool, device=emissions.device)
            
            return self._viterbi_decode(emissions, mask)
        
        def _forward_algorithm(self, emissions: Tensor, mask: Tensor) -> Tensor:
            """Forward algorithm to compute partition function"""
            batch_size, seq_len, num_tags = emissions.shape
            
            # Initialize with start transitions
            alpha = self.start_transitions + emissions[:, 0]  # [batch, num_tags]
            
            for t in range(1, seq_len):
                # Broadcast and add transitions
                # [batch, num_tags, 1] + [num_tags, num_tags] + [batch, 1, num_tags]
                emit_scores = emissions[:, t].unsqueeze(1)  # [batch, 1, num_tags]
                trans_scores = self.transitions.unsqueeze(0)  # [1, num_tags, num_tags]
                alpha_broadcast = alpha.unsqueeze(2)  # [batch, num_tags, 1]
                
                next_alpha = torch.logsumexp(
                    alpha_broadcast + trans_scores + emit_scores,
                    dim=1
                )  # [batch, num_tags]
                
                # Apply mask
                alpha = torch.where(
                    mask[:, t].unsqueeze(1),
                    next_alpha,
                    alpha
                )
            
            # Add end transitions
            alpha = alpha + self.end_transitions
            
            # Sum over all possible end tags
            return torch.logsumexp(alpha, dim=1)
        
        def _score_sequence(self, emissions: Tensor, tags: Tensor, mask: Tensor) -> Tensor:
            """Compute score of a given tag sequence"""
            batch_size, seq_len = tags.shape
            
            # Start transition score
            score = self.start_transitions[tags[:, 0]]
            
            # Emission scores
            score += emissions[:, 0].gather(1, tags[:, 0].unsqueeze(1)).squeeze(1)
            
            for t in range(1, seq_len):
                # Transition score
                trans_score = self.transitions[tags[:, t-1], tags[:, t]]
                
                # Emission score
                emit_score = emissions[:, t].gather(1, tags[:, t].unsqueeze(1)).squeeze(1)
                
                # Add scores with mask
                score += torch.where(
                    mask[:, t],
                    trans_score + emit_score,
                    torch.zeros_like(score)
                )
            
            # End transition score
            last_tag_indices = mask.sum(1) - 1
            last_tags = tags.gather(1, last_tag_indices.unsqueeze(1).long()).squeeze(1)
            score += self.end_transitions[last_tags]
            
            return score
        
        def _viterbi_decode(self, emissions: Tensor, mask: Tensor) -> List[List[int]]:
            """Viterbi algorithm for finding best path"""
            batch_size, seq_len, num_tags = emissions.shape
            
            # Initialize
            viterbi = self.start_transitions + emissions[:, 0]
            backpointers = []
            
            for t in range(1, seq_len):
                # Compute scores for all possible previous tags
                broadcast_viterbi = viterbi.unsqueeze(2)  # [batch, num_tags, 1]
                broadcast_transitions = self.transitions.unsqueeze(0)  # [1, num_tags, num_tags]
                broadcast_emissions = emissions[:, t].unsqueeze(1)  # [batch, 1, num_tags]
                
                next_scores = broadcast_viterbi + broadcast_transitions + broadcast_emissions
                
                # Find best previous tag
                next_viterbi, next_backpointer = next_scores.max(dim=1)
                
                backpointers.append(next_backpointer)
                viterbi = next_viterbi
            
            # Add end transitions
            viterbi += self.end_transitions
            
            # Backtrack to find best path
            best_paths = []
            for b in range(batch_size):
                # Find best last tag
                best_last_tag = viterbi[b].argmax().item()
                
                path = [best_last_tag]
                for backpointer in reversed(backpointers):
                    best_last_tag = backpointer[b, best_last_tag].item()
                    path.append(best_last_tag)
                
                path.reverse()
                best_paths.append(path)
            
            return best_paths


    class DenseCRF2D:
        """
        Dense CRF for 2D image segmentation
        
        مناسب برای:
        - Segmentation نقشه‌های CAD
        - مرزبندی دقیق دیوارها
        - Post-processing CNN/U-Net output
        
        این کلاس از pydensecrf استفاده می‌کند که خیلی سریع‌تر از PyTorch است.
        """
        
        def __init__(
            self,
            num_classes: int,
            sxy_gaussian: float = 3.0,  # Spatial std for Gaussian kernel
            compat_gaussian: float = 3.0,  # Compatibility for Gaussian
            sxy_bilateral: float = 80.0,  # Spatial std for bilateral kernel
            srgb_bilateral: float = 13.0,  # Color std for bilateral kernel
            compat_bilateral: float = 10.0,  # Compatibility for bilateral
            num_iterations: int = 5  # Number of mean-field iterations
        ):
            """
            Initialize Dense CRF
            
            Args:
                num_classes: تعداد کلاس‌ها (مثلاً: background, wall, column, beam, ...)
                sxy_gaussian: فاصله فضایی برای Gaussian kernel (پیکسل)
                compat_gaussian: وزن Gaussian pairwise term
                sxy_bilateral: فاصله فضایی برای bilateral kernel
                srgb_bilateral: فاصله رنگی برای bilateral kernel
                compat_bilateral: وزن bilateral pairwise term
                num_iterations: تعداد تکرار mean-field inference
            """
            if not PYDENSECRF_AVAILABLE:
                raise ImportError("pydensecrf required. Install: pip install git+https://github.com/lucasb-eyer/pydensecrf.git")
            
            self.num_classes = num_classes
            self.sxy_gaussian = sxy_gaussian
            self.compat_gaussian = compat_gaussian
            self.sxy_bilateral = sxy_bilateral
            self.srgb_bilateral = srgb_bilateral
            self.compat_bilateral = compat_bilateral
            self.num_iterations = num_iterations
        
        def refine_segmentation(
            self,
            image: np.ndarray,  # [H, W, 3] - RGB image
            probs: np.ndarray,  # [num_classes, H, W] - Softmax probabilities from CNN
            return_probs: bool = False
        ) -> np.ndarray:
            """
            بهبود segmentation با CRF
            
            Args:
                image: تصویر اصلی RGB (برای bilateral term)
                probs: احتمالات softmax از CNN/U-Net
                return_probs: اگر True باشد احتمالات برمی‌گرداند، وگرنه labels
            
            Returns:
                Refined segmentation map or probabilities
            """
            h, w = image.shape[:2]
            
            # Create CRF
            d = dcrf.DenseCRF2D(w, h, self.num_classes)
            
            # Unary potential from CNN output
            unary = unary_from_softmax(probs)
            d.setUnaryEnergy(unary)
            
            # Pairwise potential: Gaussian (appearance kernel)
            # This encourages nearby pixels with similar appearance to have same label
            d.addPairwiseGaussian(
                sxy=(self.sxy_gaussian, self.sxy_gaussian),
                compat=self.compat_gaussian,
                kernel=dcrf.DIAG_KERNEL,
                normalization=dcrf.NORMALIZE_SYMMETRIC
            )
            
            # Pairwise potential: Bilateral (smoothness kernel)
            # This encourages nearby pixels with similar color to have same label
            d.addPairwiseBilateral(
                sxy=(self.sxy_bilateral, self.sxy_bilateral),
                srgb=(self.srgb_bilateral, self.srgb_bilateral, self.srgb_bilateral),
                rgbim=image.astype(np.uint8),
                compat=self.compat_bilateral,
                kernel=dcrf.DIAG_KERNEL,
                normalization=dcrf.NORMALIZE_SYMMETRIC
            )
            
            # Inference
            Q = d.inference(self.num_iterations)
            Q = np.array(Q).reshape((self.num_classes, h, w))
            
            if return_probs:
                return Q
            else:
                return np.argmax(Q, axis=0).astype(np.uint8)
        
        def refine_batch(
            self,
            images: np.ndarray,  # [batch, H, W, 3]
            probs: np.ndarray,  # [batch, num_classes, H, W]
            return_probs: bool = False
        ) -> np.ndarray:
            """
            Batch processing for multiple images
            
            Args:
                images: Batch of RGB images
                probs: Batch of probability maps
                return_probs: Whether to return probabilities or labels
            
            Returns:
                Refined segmentation maps
            """
            results = []
            
            for i in range(len(images)):
                refined = self.refine_segmentation(
                    images[i],
                    probs[i],
                    return_probs=return_probs
                )
                results.append(refined)
            
            return np.stack(results, axis=0)


    class CRFEnhancedSegmentation(nn.Module):
        """
        CNN/U-Net با CRF post-processing
        
        این مدل ترکیبی است که:
        1. CNN/U-Net برای segmentation اولیه
        2. CRF برای بهبود مرزها
        
        برای استفاده در تمام صنایع:
        - ساختمان‌سازی: تشخیص دیوار، ستون، تیر
        - پل‌سازی: تشخیص اجزای پل
        - جاده‌سازی: تشخیص لاین‌ها، مرزها
        - سدسازی: تشخیص بدنه سد، پی
        - تونل‌سازی: تشخیص مقطع تونل
        - کارخانه: تشخیص تجهیزات
        - ماشین‌سازی: تشخیص قطعات
        """
        
        def __init__(
            self,
            backbone: nn.Module,  # CNN or U-Net
            num_classes: int,
            use_crf: bool = True,
            crf_params: Optional[Dict[str, Any]] = None
        ):
            """
            Initialize CRF-Enhanced Segmentation
            
            Args:
                backbone: CNN or U-Net model for initial segmentation
                num_classes: Number of segmentation classes
                use_crf: Whether to use CRF post-processing
                crf_params: Parameters for CRF (optional)
            """
            super().__init__()
            
            self.backbone = backbone
            self.num_classes = num_classes
            self.use_crf = use_crf
            
            if use_crf and PYDENSECRF_AVAILABLE:
                crf_params = crf_params or {}
                self.crf = DenseCRF2D(num_classes=num_classes, **crf_params)
            else:
                self.crf = None
        
        def forward(self, x: Tensor) -> Tensor:
            """
            Forward pass (only CNN, no CRF in training)
            
            Args:
                x: Input image [batch, 3, H, W]
            
            Returns:
                Logits [batch, num_classes, H, W]
            """
            return self.backbone(x)
        
        def predict(
            self,
            x: Tensor,  # [batch, 3, H, W]
            images_rgb: Optional[np.ndarray] = None,  # For CRF
            use_crf: Optional[bool] = None
        ) -> Tensor:
            """
            Prediction with optional CRF refinement
            
            Args:
                x: Input tensor
                images_rgb: RGB images for CRF (numpy array)
                use_crf: Override default CRF usage
            
            Returns:
                Predicted segmentation map
            """
            use_crf = use_crf if use_crf is not None else self.use_crf
            
            # CNN/U-Net forward
            with torch.no_grad():
                logits = self.backbone(x)
                probs = F.softmax(logits, dim=1)
            
            if use_crf and self.crf is not None and images_rgb is not None:
                # Convert to numpy
                probs_np = probs.cpu().numpy()
                
                # Apply CRF
                refined = self.crf.refine_batch(
                    images=images_rgb,
                    probs=probs_np,
                    return_probs=False
                )
                
                return torch.from_numpy(refined).to(x.device)
            else:
                # No CRF, just argmax
                return torch.argmax(probs, dim=1)


else:
    # Placeholder if PyTorch not available
    class LinearChainCRF:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch required for CRF")
    
    class DenseCRF2D:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch and pydensecrf required")
    
    class CRFEnhancedSegmentation:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch required")


# ============================================================================
# Helper Functions
# ============================================================================

def create_simple_unet(num_classes: int, in_channels: int = 3) -> nn.Module:
    """
    ساخت یک U-Net ساده برای segmentation
    
    برای استفاده در:
    - تشخیص دیوارها در نقشه ساختمان
    - تشخیص خطوط در نقشه جاده
    - تشخیص اجزای مکانیکی
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required")
    
    class SimpleUNet(nn.Module):
        def __init__(self):
            super().__init__()
            
            # Encoder
            self.enc1 = self._conv_block(in_channels, 64)
            self.enc2 = self._conv_block(64, 128)
            self.enc3 = self._conv_block(128, 256)
            self.enc4 = self._conv_block(256, 512)
            
            # Bottleneck
            self.bottleneck = self._conv_block(512, 1024)
            
            # Decoder
            self.dec4 = self._upconv_block(1024, 512)
            self.dec3 = self._upconv_block(512, 256)
            self.dec2 = self._upconv_block(256, 128)
            self.dec1 = self._upconv_block(128, 64)
            
            # Output
            self.out = nn.Conv2d(64, num_classes, kernel_size=1)
            
            self.pool = nn.MaxPool2d(2)
        
        def _conv_block(self, in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
        
        def _upconv_block(self, in_ch, out_ch):
            return nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
        
        def forward(self, x):
            # Encoder
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            e3 = self.enc3(self.pool(e2))
            e4 = self.enc4(self.pool(e3))
            
            # Bottleneck
            b = self.bottleneck(self.pool(e4))
            
            # Decoder with skip connections
            d4 = self.dec4(b) + e4
            d3 = self.dec3(d4) + e3
            d2 = self.dec2(d3) + e2
            d1 = self.dec1(d2) + e1
            
            return self.out(d1)
    
    return SimpleUNet()


def visualize_segmentation(
    image: np.ndarray,
    segmentation: np.ndarray,
    num_classes: int,
    alpha: float = 0.5
) -> np.ndarray:
    """
    تصویرسازی نتیجه segmentation
    
    Args:
        image: تصویر اصلی [H, W, 3]
        segmentation: نقشه segmentation [H, W]
        num_classes: تعداد کلاس‌ها
        alpha: شفافیت overlay
    
    Returns:
        تصویر overlay شده
    """
    # Create color map
    colors = np.random.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]  # Background = black
    
    # Map segmentation to colors
    colored = colors[segmentation]
    
    # Blend with original image
    blended = (alpha * image + (1 - alpha) * colored).astype(np.uint8)
    
    return blended


# ============================================================================
# Example Usage
# ============================================================================

def main():
    """مثال استفاده از CRF"""
    
    if not TORCH_AVAILABLE:
        print("❌ PyTorch required")
        return
    
    print("="*70)
    print("CRF-Enhanced Segmentation Demo")
    print("="*70)
    
    # Parameters
    num_classes = 10  # مثلاً: background, wall, column, beam, door, window, ...
    img_size = 256
    
    # 1. Create U-Net backbone
    print("\n1️⃣ Creating U-Net backbone...")
    backbone = create_simple_unet(num_classes=num_classes)
    print(f"   ✓ U-Net created ({num_classes} classes)")
    
    # 2. Create CRF-enhanced model
    print("\n2️⃣ Creating CRF-enhanced model...")
    model = CRFEnhancedSegmentation(
        backbone=backbone,
        num_classes=num_classes,
        use_crf=PYDENSECRF_AVAILABLE,
        crf_params={
            'sxy_gaussian': 3.0,
            'compat_gaussian': 3.0,
            'sxy_bilateral': 80.0,
            'srgb_bilateral': 13.0,
            'compat_bilateral': 10.0,
            'num_iterations': 5
        }
    )
    print(f"   ✓ Model created (CRF: {PYDENSECRF_AVAILABLE})")
    
    # 3. Test with random data
    print("\n3️⃣ Testing with synthetic data...")
    x = torch.randn(2, 3, img_size, img_size)
    
    # Forward pass (training mode)
    logits = model(x)
    print(f"   ✓ Forward pass: {logits.shape}")
    
    # Prediction (inference mode)
    if PYDENSECRF_AVAILABLE:
        images_rgb = (torch.rand(2, img_size, img_size, 3) * 255).numpy().astype(np.uint8)
        pred = model.predict(x, images_rgb=images_rgb, use_crf=True)
        print(f"   ✓ Prediction with CRF: {pred.shape}")
    else:
        pred = model.predict(x, use_crf=False)
        print(f"   ✓ Prediction without CRF: {pred.shape}")
    
    print("\n" + "="*70)
    print("✅ Demo complete!")
    print("="*70)
    
    print("\n💡 Use cases:")
    print("   🏗️  Building: Detect walls, columns, beams")
    print("   🌉 Bridge: Segment bridge components")
    print("   🛣️  Road: Detect lanes, boundaries")
    print("   🚧 Dam: Identify dam body, foundation")
    print("   🚇 Tunnel: Segment tunnel sections")
    print("   🏭 Factory: Detect equipment, machinery")
    print("   ⚙️  Machinery: Identify parts, components")


if __name__ == "__main__":
    main()
