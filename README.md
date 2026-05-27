# Shadow Art

Interactive explorer for shadow/haze detection using bright channel and dark channel priors. Implements algorithms from Panagopoulos et al. and He et al. with Felzenszwalb segmentation, soft matting, and guided filter refinement.

## Run locally

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11+.

```
uv run python app.py
```

Open http://localhost:5555

## What it does

- **Haze mode**: Dark channel prior (He et al. CVPR 2009) for depth estimation and dehazing
- **Shadow mode**: Bright channel cue (Panagopoulos et al.) for shadow detection
- **Segmentation**: TPAMI shadow detection via Felzenszwalb segmentation + GMM histogram confidence
- **Soft matting**: Levin et al. closed-form matting Laplacian for sharp edge-preserving refinement

Drop images onto the page or use the upload button. Adjust parameters with the sliders. Export renders at full resolution.
