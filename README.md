<img width="444"  alt="084894F7-48AE-491D-9C79-AD8497D6A0CE_1_102_o_1_haze_k1_b0 09_g1 1_r9_e0 0006_transmission" src="https://github.com/user-attachments/assets/469e7ef6-ed8d-415d-b31f-85d06413830c" />



Hi! This is an intereactive explorer for some image processing algorithms related to shadow and haze detection. The app runs in the browser and allows you to drop an image on the page and play around. 


## Quick start

Click the green button at the top of the page and "download zip".

Unzip the file in Finder, then right-click the folder and select “new terminal at folder”. 

<img width="444"  alt="Screenshot 2026-05-27 at 10 53 39 PM" src="https://github.com/user-attachments/assets/9ea0378e-5030-4566-a02d-4959ed618803" />


In the terminal, paste this command into the terminal to install [uv](https://docs.astral.sh/uv/), the project's only external dependency.

```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

When `uv` is downloaded, you'll need to close and re-open the terminal to actually be able to use it. When you've done this (make sure new terminal is in the same location), this command will run the app:

```
uv run python app.py
```

Now, in any browser, go to http://localhost:5555


Have fun!

### What it does (technical)

- **Haze mode**: Dark channel prior for depth estimation and dehazing
- **Shadow mode**: Bright channel cue for shadow detection
- **Segmentation**: Shadow detection via Felzenszwalb segmentation + GMM histogram confidence
- **Soft matting**: Closed-form matting Laplacian for sharp edge-preserving refinement


### Papers

- He, Sun, Tang. [Single Image Haze Removal Using Dark Channel Prior](https://ieeexplore.ieee.org/document/5567108) (CVPR 2009, TPAMI 2011)
- Panagopoulos, Wang, Samaras, Paragios. [Estimating Shadows with the Bright Channel Cue](https://link.springer.com/chapter/10.1007/978-3-642-17277-9_1) (ECCV 2010 Workshop)
- Panagopoulos, Wang, Samaras, Paragios. [Simultaneous Cast Shadows, Illumination and Geometry Inference Using Hypergraphs](https://ieeexplore.ieee.org/document/6197293) (TPAMI 2013)
- Levin, Lischinski, Weiss. [A Closed-Form Solution to Natural Image Matting](https://ieeexplore.ieee.org/document/4359322) (TPAMI 2008)
