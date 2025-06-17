# VGGT Experimenter

A comprehensive testing script for VGGT (Visual Geometry Grounded Transformer) with multi-view 3D reconstruction capabilities.

## Overview

VGGT is a state-of-the-art neural network for multi-view 3D reconstruction presented at CVPR 2025 by Oxford and Meta providing:

- **Camera pose estimation** (direct output, no PnP needed)
- **Depth maps** with confidence scores
- **3D point clouds** (both from depth and direct 3D prediction)
- **COLMAP format export** for integration with other tools
- **Comprehensive timing analysis**


## Installation

### 1. Download VGGT

```bash
# Clone VGGT repository
git clone https://github.com/facebookresearch/vggt.git
cd vggt
```

### 2. Install Dependencies

Follow instrunctions found at [VGGT GitHub Repository](https://github.com/facebookresearch/vggt)

### 3. Model Download

The VGGT-1B model (5GB) will be automatically downloaded from Hugging Face on first use and cached locally. Subsequent runs use the cached model.

## Usage

### Basic Usage

```bash
python vggt_experimenter.py \
    --images /path/to/scene/images \
    --workspace /path/to/output \
    --conf_threshold 65.0
```

### Full Options

```bash
python vggt_experimenter.py \
    --images /path/to/scene/images \
    --workspace /path/to/output \
    --size 512 \
    --dtype fp16 \
    --conf_threshold 65.0 \
    --model_size 1B
```

### Batch Processing

```bash
python vggt_experimenter.py \
    --dataset_root /path/to/multiple/scenes \
    --workspace /path/to/output \
    --conf_threshold 65.0
```

## Parameters

| Parameter | Description | Default | Recommended Values |
|-----------|-------------|---------|-------------------|
| `--images` | Path to folder containing scene images | Required | - |
| `--dataset_root` | Path to parent folder for batch processing | Alternative to --images | - |
| `--workspace` | Output directory for results | Required | - |
| `--size` | Resize longer edge (pixels) | 512 | 224-512 (smaller = less memory) |
| `--dtype` | Inference precision | fp16 | fp16, bf16, fp32 |
| `--conf_threshold` | Confidence percentile (0-100%) | 50.0 | See guidelines below |
| `--model_size` | VGGT model variant | 1B | Currently only 1B available |

## Confidence Threshold Guidelines

The confidence threshold is specified as a **percentile** (0-100%), where higher values keep only the most confident points:

### Recommended Values by Scene Type

| Scene Type | Confidence Threshold | Reasoning |
|------------|---------------------|-----------|
| **Large outdoor scenes** (Tanks & Temples) | **65-80%** | Complex geometry, occlusions, varying lighting |
| **Object-centric scenes** | **40-60%** | Simpler geometry, controlled conditions |
| **Indoor scenes** | **50-70%** | Moderate complexity, some occlusions |
| **Simple objects** | **30-50%** | Clean geometry, minimal noise |

### Examples

```bash
# Large outdoor scene (Tanks & Temples style)
--conf_threshold 75.0

# Simple object reconstruction  
--conf_threshold 45.0

# Mixed indoor/outdoor scene
--conf_threshold 60.0
```

## Output Structure

The script generates comprehensive outputs in the workspace directory:

```
workspace/
├── cameras_extrinsic.npy          # Camera poses [S, 4, 4]
├── cameras_intrinsic.npy          # Camera intrinsics [S, 3, 3]
├── pointcloud_from_depth.ply      # Point cloud from depth maps
├── pointcloud_from_points.ply     # Point cloud from direct 3D prediction
├── timings.csv                    # Performance timing breakdown
├── colmap/                        # COLMAP format export
│   ├── cameras.txt
│   ├── images.txt
│   └── points3D.bin
└── depth_maps/                    # Depth and confidence visualizations
    ├── depth_000.png
    ├── conf_000.png
    └── ...
```

## Memory Management

VGGT-1B requires significant GPU memory. For RTX 4090 (24GB):

| Image Count | Resolution | Memory Usage | Recommendation |
|-------------|------------|--------------|----------------|
| 10-20 images | 512x512 | ~8-12GB | Safe |
| 50 images | 224x224 | ~12-16GB | Works |
| 100+ images | 224x224 | >24GB | OOM on RTX 4090 |

Note: On RTX 4090, you will get OOM with 100 images at 224x224 resolution.



## Links

- [VGGT GitHub Repository](https://github.com/facebookresearch/vggt)
- [VGGT Project Page](https://vgg-t.github.io/)
- [Paper (CVPR 2025)](https://jytime.github.io/data/VGGT_CVPR25.pdf)
