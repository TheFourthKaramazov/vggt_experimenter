#!/usr/bin/env python3
"""
vggt_experimenter.py   •   17-Jun-2025
---------------------------------------
VGGT runner with timings, camera poses, depth maps, point clouds, and tracking.

VGGT uses DINOv2 encoder with alternating attention mechanism for multi-view 3D reconstruction.

Single scene
------------
python vggt_experimenter.py \
    --images            /path/to/scene/images \
    --workspace         /results/scene \
    --size              512 \
    --dtype             fp16 \
    --conf_threshold    65.0 \
    --model_size        1B

Batch
-----
python vggt_experimenter.py \
    --dataset_root      /data/multi_scenes \
    --workspace         /results/all \
    --size              512 \
    --dtype             bf16 \
    --conf_threshold    65.0 \
    --model_size        1B
"""

import argparse
import csv
import time
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import open3d as o3d
import torch

# Add VGGT to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external" / "vggt"))

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map


def run_timed(func, *args, **kwargs):
    """Time a function execution."""
    t0 = time.perf_counter()
    out = func(*args, **kwargs)
    return out, time.perf_counter() - t0


def save_ply(xyz: np.ndarray, rgb: np.ndarray, path: Path):
    """Save point cloud as PLY file."""
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(xyz)
    if rgb is not None and rgb.shape[0] > 0:
        pc.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64) / 255.0)
    o3d.io.write_point_cloud(str(path), pc, compressed=True)


def save_cameras_colmap_format(cameras, output_dir: Path):
    """Save camera parameters in COLMAP format."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save intrinsics (cameras.txt)
    with open(output_dir / "cameras.txt", "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for i, K in enumerate(cameras['intrinsic']):
            # Assume PINHOLE model: fx, fy, cx, cy
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            f.write(f"{i} PINHOLE 512 512 {fx} {fy} {cx} {cy}\n")
    
    # Save poses (images.txt)
    with open(output_dir / "images.txt", "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for i, ext in enumerate(cameras['extrinsic']):
            # Convert extrinsic matrix to quaternion + translation
            R = ext[:3, :3]
            t = ext[:3, 3]
            
            # Convert rotation matrix to quaternion (simplified)
            trace = R[0,0] + R[1,1] + R[2,2]
            if trace > 0:
                s = np.sqrt(trace + 1.0) * 2
                qw = 0.25 * s
                qx = (R[2,1] - R[1,2]) / s
                qy = (R[0,2] - R[2,0]) / s
                qz = (R[1,0] - R[0,1]) / s
            else:
                qw, qx, qy, qz = 1, 0, 0, 0  # Fallback
            
            f.write(f"{i} {qw} {qx} {qy} {qz} {t[0]} {t[1]} {t[2]} {i} image_{i:03d}.jpg\n")
            f.write("\n")  # Empty line for points


def process_scene(
    img_dir: Path,
    ws_dir: Path,
    size: int,
    device,
    dtype,
    model_size: str,
    conf_threshold: float,  # percentile (0-100%), keeps top X% most confident points
):
    """Process a single scene with VGGT."""
    ws_dir.mkdir(parents=True, exist_ok=True)
    timings = {}
    T0 = time.perf_counter()

    # 1) Load VGGT model
    print(f"Loading VGGT-{model_size} model...")
    model, timings["load_model"] = run_timed(
        VGGT.from_pretrained, f"facebook/VGGT-{model_size}"
    )
    model.to(device).eval()

    # 2) Gather images
    files = sorted(p for p in img_dir.rglob("*")
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not files:
        print(f"⚠  No images found in {img_dir}")
        return
    
    print(f"Found {len(files)} images")
    orig_imgs = [Image.open(fp).convert("RGB") for fp in files]

    # 3) Preprocess images for VGGT
    image_paths = [str(p) for p in files]
    images, timings["load_imgs"] = run_timed(
        load_and_preprocess_images, image_paths
    )
    images = images.to(device)

    # 4) VGGT Inference - predict all outputs
    print("Running VGGT inference...")
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=dtype):
            predictions, timings["inference"] = run_timed(
                model, images
            )

    # 5) Convert pose encoding to camera matrices
    print("Converting pose encoding to camera matrices...")
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    
    # 6) Extract predictions with correct keys and remove batch dimension
    print("Extracting predictions...")
    
    # Camera parameters  
    extrinsic = extrinsic[0].cpu().numpy()  # Remove batch dim: [S, 4, 4]
    intrinsic = intrinsic[0].cpu().numpy()  # Remove batch dim: [S, 3, 3]
    
    # Depth maps and confidence
    depth_maps = predictions['depth'][0].cpu().numpy()      # Remove batch dim: [S, H, W, 1] -> [S, H, W]
    depth_conf = predictions['depth_conf'][0].cpu().numpy() # Remove batch dim: [S, H, W]
    
    # Remove singleton dimension from depth if present
    if depth_maps.ndim == 4 and depth_maps.shape[-1] == 1:
        depth_maps = depth_maps.squeeze(-1)  # [S, H, W, 1] -> [S, H, W]
    
    # Point maps and confidence  
    point_maps = predictions['world_points'][0].cpu().numpy()      # Remove batch dim: [S, H, W, 3]
    point_conf = predictions['world_points_conf'][0].cpu().numpy() # Remove batch dim: [S, H, W]

    # 7) Save camera parameters
    cameras = {
        'extrinsic': extrinsic,
        'intrinsic': intrinsic
    }
    np.save(ws_dir / "cameras_extrinsic.npy", extrinsic)
    np.save(ws_dir / "cameras_intrinsic.npy", intrinsic)
    
    # Save in COLMAP format
    save_cameras_colmap_format(cameras, ws_dir / "colmap")

    # 8) Generate point clouds from both depth and point predictions
    t_fuse = time.perf_counter()
    
    # From depth maps (usually more accurate according to VGGT paper)
    depth_xyz_list, depth_rgb_list = [], []
    # From point maps (direct 3D prediction)
    point_xyz_list, point_rgb_list = [], []
    
    for i, (orig_img, depth, d_conf, points, p_conf, K, ext) in enumerate(
        zip(orig_imgs, depth_maps, depth_conf, point_maps, point_conf, intrinsic, extrinsic)
    ):
        H, W = depth.shape
        
        # Percentile-based confidence filtering (VGGT approach)
        # Calculate confidence thresholds as percentiles
        depth_conf_flat = d_conf[np.isfinite(d_conf)]
        point_conf_flat = p_conf[np.isfinite(p_conf)]
        
        if conf_threshold == 0.0:
            depth_threshold = 0.0
            point_threshold = 0.0
        else:
            depth_threshold = np.percentile(depth_conf_flat, conf_threshold) if depth_conf_flat.size > 0 else 0.0
            point_threshold = np.percentile(point_conf_flat, conf_threshold) if point_conf_flat.size > 0 else 0.0
        
        # Apply percentile-based filtering
        depth_mask = (d_conf >= depth_threshold) & (d_conf > 1e-5)  # VGGT uses 1e-5 minimum
        point_mask = (p_conf >= point_threshold) & (p_conf > 1e-5)
        
        # Check for valid depth/points
        depth_valid = np.isfinite(depth) & (depth > 0)
        point_valid = np.isfinite(points).all(-1)
        
        depth_mask &= depth_valid
        point_mask &= point_valid
        
        if not depth_mask.any() and not point_mask.any():
            continue
            
        # Resize original image to match prediction resolution
        rgb = np.asarray(orig_img.resize((W, H), Image.LANCZOS))
        
        # From depth maps - unproject to 3D
        if depth_mask.any():
            # Create pixel coordinates
            y, x = np.mgrid[0:H, 0:W]
            pixels = np.stack([x, y, np.ones_like(x)], axis=-1).astype(np.float32)
            
            # Unproject using camera intrinsics
            K_inv = np.linalg.inv(K)
            rays = (pixels @ K_inv.T) * depth[..., None]  # [H, W, 3]
            
            # Transform to world coordinates
            world_points = rays @ ext[:3, :3].T + ext[:3, 3]  # [H, W, 3]
            
            depth_xyz_list.append(world_points[depth_mask])
            depth_rgb_list.append(rgb[depth_mask])
        
        # From point maps - direct 3D points
        if point_mask.any():
            point_xyz_list.append(points[point_mask])
            point_rgb_list.append(rgb[point_mask])

    # Combine all points
    if depth_xyz_list:
        depth_xyz_all = np.concatenate(depth_xyz_list, 0)
        depth_rgb_all = np.concatenate(depth_rgb_list, 0)
        save_ply(depth_xyz_all, depth_rgb_all, ws_dir / "pointcloud_from_depth.ply")
    else:
        depth_xyz_all = np.zeros((0, 3))
        
    if point_xyz_list:
        point_xyz_all = np.concatenate(point_xyz_list, 0)
        point_rgb_all = np.concatenate(point_rgb_list, 0)
        save_ply(point_xyz_all, point_rgb_all, ws_dir / "pointcloud_from_points.ply")
    else:
        point_xyz_all = np.zeros((0, 3))

    timings["fuse_pts"] = time.perf_counter() - t_fuse

    # 9) Save depth maps as images for visualization
    depth_dir = ws_dir / "depth_maps"
    depth_dir.mkdir(exist_ok=True)
    for i, (depth, conf) in enumerate(zip(depth_maps, depth_conf)):
        # Normalize depth for visualization
        valid_depth = depth[np.isfinite(depth) & (depth > 0)]
        if valid_depth.size > 0:
            depth_norm = (depth - valid_depth.min()) / (valid_depth.max() - valid_depth.min())
            depth_norm = np.clip(depth_norm * 255, 0, 255).astype(np.uint8)
            Image.fromarray(depth_norm).save(depth_dir / f"depth_{i:03d}.png")
            
            # Save confidence map
            conf_norm = np.clip(conf * 255, 0, 255).astype(np.uint8)
            Image.fromarray(conf_norm).save(depth_dir / f"conf_{i:03d}.png")

    timings["total"] = time.perf_counter() - T0

    # 10) Write CSV + console summary
    with (ws_dir / "timings.csv").open("w", newline="") as f:
        csv.writer(f).writerows([["stage", "seconds"], *timings.items()])

    print(f"\n=== {ws_dir.name} – VGGT Results ===")
    for k, v in timings.items():
        print(f"{k:<15}: {v:7.2f}s")
    print("=== Statistics ===")
    print(f"views          : {len(extrinsic)}")
    print(f"points (depth) : {depth_xyz_all.shape[0]}")
    print(f"points (direct): {point_xyz_all.shape[0]}")
    if depth_xyz_all.size:
        print(f"mean |XYZ| (depth): {np.linalg.norm(depth_xyz_all, axis=1).mean():.2f} units")
    if point_xyz_all.size:
        print(f"mean |XYZ| (direct): {np.linalg.norm(point_xyz_all, axis=1).mean():.2f} units")


def main():
    ap = argparse.ArgumentParser(
        description="VGGT experimenter with comprehensive 3D reconstruction analysis"
    )
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--images", help="folder of images for ONE scene")
    g.add_argument("--dataset_root", help="batch: parent folder of scenes")
    ap.add_argument(
        "--workspace", required=True,
        help="where to write outputs"
    )
    ap.add_argument(
        "--size", type=int, default=512,
        help="resize longer edge (default 512)"
    )
    ap.add_argument(
        "--dtype", choices=("fp32", "fp16", "bf16"), default="fp16",
        help="data type for inference (default fp16)"
    )
    ap.add_argument(
        "--model_size", choices=("1B",), default="1B",
        help="VGGT model size (default 1B)"
    )
    ap.add_argument(
        "--conf_threshold", type=float, default=50.0,
        help="confidence threshold as percentile (0-100%%, default 50%% - keeps top 50%% most confident points)"
    )
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.dtype == "fp32":
        dtype = torch.float32
    elif args.dtype == "fp16":
        dtype = torch.float16
    else:  # bf16
        dtype = torch.bfloat16

    ws = Path(args.workspace).expanduser().resolve()
    ws.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Dtype: {dtype}")
    print(f"Model: VGGT-{args.model_size}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")

    if args.images:
        process_scene(
            Path(args.images), ws,
            size=args.size,
            device=device,
            dtype=dtype,
            model_size=args.model_size,
            conf_threshold=args.conf_threshold
        )
    else:
        for scene in sorted(Path(args.dataset_root).iterdir()):
            if scene.is_dir():
                print(f"\nProcessing scene: {scene.name}")
                process_scene(
                    scene, ws / scene.name,
                    size=args.size,
                    device=device,
                    dtype=dtype,
                    model_size=args.model_size,
                    conf_threshold=args.conf_threshold
                )


if __name__ == "__main__":
    main()
