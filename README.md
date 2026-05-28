This is an intereactive explorer for some image processing algorithms related to shadow and haze detection. You can drop an image on the page and play around. 

## How to download and use

Click the green button above ↗ , download zip.

Unzip, then right-click the folder in Finder and select “new terminal at folder”. 

Paste the below command into the terminal. It's to install [uv](https://docs.astral.sh/uv/), the project's only external dependency.

```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

The paste this command to run the app:

```
uv run python app.py
```


In any browser, go to http://localhost:5555


## What it does

- **Haze mode**: Dark channel prior for depth estimation and dehazing
- **Shadow mode**: Bright channel cue for shadow detection
- **Segmentation**: Shadow detection via Felzenszwalb segmentation + GMM histogram confidence
- **Soft matting**: Closed-form matting Laplacian for sharp edge-preserving refinement

Drop images onto the page or use the upload button. Adjust parameters with the sliders. Export renders at full resolution.

## Papers

- He, Sun, Tang. [Single Image Haze Removal Using Dark Channel Prior](https://ieeexplore.ieee.org/document/5567108) (CVPR 2009, TPAMI 2011)
- Panagopoulos, Wang, Samaras, Paragios. [Estimating Shadows with the Bright Channel Cue](https://link.springer.com/chapter/10.1007/978-3-642-17277-9_1) (ECCV 2010 Workshop)
- Panagopoulos, Wang, Samaras, Paragios. [Simultaneous Cast Shadows, Illumination and Geometry Inference Using Hypergraphs](https://ieeexplore.ieee.org/document/6197293) (TPAMI 2013)
- Levin, Lischinski, Weiss. [A Closed-Form Solution to Natural Image Matting](https://ieeexplore.ieee.org/document/4359322) (TPAMI 2008)
