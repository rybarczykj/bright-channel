
Hi! This app is an intereactive explorer for some image processing algorithms related to shadow and haze detection. The app runs in the browser and allows you to drop an image on the page and play around. 

<img width="444"  alt="image" src="https://github.com/user-attachments/assets/ca4048c9-32ea-4a12-8f6b-3d58edf78a22" />



### Quick start

#### 1. Download this project

Click the green button at the top of the page and "download zip".

  <img width="333" alt="Screenshot 2026-05-27 at 11 02 23 PM" src="https://github.com/user-attachments/assets/fc1983f0-bbcf-4cd8-9016-cb54304cf0e0" />



#### 2. Open the project in a terminal


In Finder, right-click the (unzipped) folder and select “new terminal at folder”. 

<img width="333"  alt="Screenshot 2026-05-27 at 10 53 39 PM" src="https://github.com/user-attachments/assets/9ea0378e-5030-4566-a02d-4959ed618803" />


#### 3. Install required dependency

[uv](https://docs.astral.sh/uv/) needs to be installed to run this. It's very common and can be trusted. To install, paste this command into the terminal:

```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

After `uv` is downloaded, you'll need to close and re-open the terminal to actually use it. 

#### 4. Run the app

Once you've re-opened the terminal, this command will run the app:

```
uv run python app.py
```

Now, in your web browser, go to http://localhost:5555

Have fun!




--- 
#### Terms 

- **Haze mode**: Dark channel prior for depth estimation and dehazing
- **Shadow mode**: Bright channel cue for shadow detection
- **Segmentation**: Shadow detection via Felzenszwalb segmentation + GMM histogram confidence
- **Soft matting**: Closed-form matting Laplacian for sharp edge-preserving refinement

#### Papers

- He, Sun, Tang. [Single Image Haze Removal Using Dark Channel Prior](https://projectsweb.cs.washington.edu/research/insects/CVPR2009/award/hazeremv_drkchnl.pdf) (CVPR 2009, TPAMI 2011)
- Panagopoulos, Wang, Samaras, Paragios. [Estimating Shadows with the Bright Channel Cue](https://link.springer.com/chapter/10.1007/978-3-642-35740-4_1) (ECCV 2010 Workshop)
- Panagopoulos, Wang, Samaras, Paragios. [Simultaneous Cast Shadows, Illumination and Geometry Inference Using Hypergraphs](https://www3.cs.stonybrook.edu/~cvl/content/papers/2013/alex_pami2013.pdf) (TPAMI 2013)
- Levin, Lischinski, Weiss. [A Closed-Form Solution to Natural Image Matting](https://people.csail.mit.edu/alevin/papers/Matting-Levin-Lischinski-Weiss-CVPR06.pdf) (TPAMI 2008)
