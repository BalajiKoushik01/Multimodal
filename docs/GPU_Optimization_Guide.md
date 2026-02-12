# GPU Optimization Guide for GTX 1650 (4GB VRAM)

## Current Optimized Configuration

Your training has been automatically optimized for your **NVIDIA GeForce GTX 1650 (4.29 GB VRAM)**:

### ✅ **Optimizations Applied**

**1. Model Architecture**:
- **Backbone**: ResNet50 (optimal for 4GB VRAM)
- **Batch Size**: 12 per step
- **Gradient Accumulation**: 4 steps → **Effective Batch Size: 48**
- **Mixed Precision (AMP)**: Enabled (reduces memory by ~50%)

**2. Data Loading Optimizations**:
- **Num Workers**: 4 (parallel data loading)
- **Pin Memory**: Enabled (faster CPU→GPU transfer)
- **Persistent Workers**: Enabled (workers stay alive between epochs)
- **Prefetch Factor**: 2 (prefetch 2 batches ahead)
- **Image Caching**: First 1000 images cached in RAM

**3. CUDA Optimizations**:
```python
torch.backends.cudnn.benchmark = True  # Auto-tune convolutions
torch.backends.cuda.matmul.allow_tf32 = True  # TF32 for matmul
torch.backends.cudnn.allow_tf32 = True  # TF32 for convolutions
```

**4. Training Strategy**:
- **SSL Strategy**: Multimodal contrastive learning
- **Epochs**: 100 (full training)
- **Learning Rate**: 1e-3 with cosine annealing
- **Warmup**: 10 epochs
- **Checkpointing**: Every 5 epochs + best model

---

## Performance Expectations

### Training Speed (GTX 1650):
- **~3-5 seconds/batch** (batch size 12)
- **~200 batches/epoch** (11,193 training images)
- **~15-20 minutes/epoch**
- **Total training time**: ~25-33 hours for 100 epochs

### GPU Utilization:
- **Memory Usage**: ~3.5-3.8 GB / 4.29 GB (~85%)
- **GPU Utilization**: 95-100% during training
- **Power**: Near maximum TDP

---

## Why ResNet50 Instead of ViT?

For 4GB VRAM, ResNet50 is the optimal choice:

| Model | Memory (FP16) | Batch Size | Performance |
|-------|---------------|------------|-------------|
| **ResNet50** | ~3.5 GB | 12 | ✅ Optimal |
| ViT-Small | ~4.2 GB | 8 | ⚠️ Tight fit |
| ViT-Base | ~6.5 GB | 4 | ❌ OOM risk |

**ResNet50 advantages**:
- Proven performance on medical imaging
- Efficient memory usage
- Faster training per epoch
- More stable with mixed precision

---

## Upgrade Path for Better Performance

If you want to use ViT-Base for world-class performance, consider:

**Option 1: Cloud GPU** (Recommended)
- Google Colab Pro: Tesla T4 (16GB) - $10/month
- Paperspace: RTX 4000 (8GB) - $0.51/hour
- AWS EC2: g4dn.xlarge (16GB) - $0.526/hour

**Option 2: Local GPU Upgrade**
- RTX 3060 (12GB): ~$300 - Can run ViT-Base with batch size 16
- RTX 4060 Ti (16GB): ~$500 - Can run ViT-Base with batch size 24
- RTX 4070 (12GB): ~$600 - Can run ViT-Base with batch size 20

**Option 3: Optimize Current Setup**
- Use gradient checkpointing (trade compute for memory)
- Reduce image size to 192×192
- Use ViT-Tiny or ViT-Small instead

---

## Monitoring Training

**Check GPU Usage**:
```bash
nvidia-smi -l 1  # Update every second
```

**Monitor Training**:
- Checkpoints: `outputs/ssl_worldclass/checkpoints/`
- Logs: `outputs/ssl_worldclass/logs/metrics.json`
- Best model: `outputs/ssl_worldclass/checkpoints/best_model.pt`

**Expected Loss Curve**:
- Epoch 0-10: Loss decreases rapidly (5.0 → 2.0)
- Epoch 10-50: Steady decrease (2.0 → 1.0)
- Epoch 50-100: Fine-tuning (1.0 → 0.5-0.7)

---

## Next Steps After SSL Pre-Training

1. **Evaluate Representations** (Week 5-6):
   - Linear probing on labeled data
   - Visualize learned features
   - Check attention maps

2. **Supervised Fine-Tuning** (Week 7-8):
   - Load best SSL checkpoint
   - Fine-tune on ~1,300 labeled images
   - Multi-task learning (malignancy, severity, progression)

3. **Evaluation** (Week 9-10):
   - Test on held-out set
   - Cross-dataset validation
   - Clinical expert review

---

## Current Training Status

**Configuration**:
- GPU: NVIDIA GeForce GTX 1650 (4.29 GB)
- Backbone: ResNet50
- Batch Size: 12 (effective: 48)
- SSL Strategy: Multimodal contrastive
- Epochs: 100
- Mixed Precision: Enabled

**Optimizations**:
- ✅ CUDA optimizations (TF32, cudnn.benchmark)
- ✅ Data prefetching and caching
- ✅ Persistent workers
- ✅ Gradient accumulation
- ✅ Mixed precision training

**Expected Performance**:
- Training time: ~25-33 hours
- GPU utilization: 95-100%
- Memory usage: ~85% (3.5-3.8 GB)

---

## Tips for Maximum Performance

1. **Close Other Applications**: Free up GPU memory
2. **Monitor Temperature**: Ensure good cooling (keep <80°C)
3. **Use SSD**: Faster data loading from SSD vs HDD
4. **Stable Power**: Avoid power interruptions during training
5. **Let it Run**: Don't interrupt training unnecessarily

---

## Troubleshooting

**Out of Memory (OOM)**:
- Reduce batch size to 10 or 8
- Disable image caching
- Reduce num_workers to 2

**Slow Training**:
- Check if GPU is being used: `nvidia-smi`
- Ensure CUDA is properly installed
- Close background GPU applications

**Training Crashes**:
- Check GPU temperature
- Update NVIDIA drivers
- Verify CUDA/PyTorch compatibility

---

**Your training is optimized for maximum performance on GTX 1650!** 🚀
