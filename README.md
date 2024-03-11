# CG-SENSE recon practice
This repo contains code for model-based MRI reconstruction. Specifically, CG-SENSE like recon is implemented using matlab's pcg(), which iteratively solves the problem Ah*A*x = Ah*y to recover the underlying image x given k-space data y and system model A. Note that this repo is for education purposes and NOT meant to be a high performance solution for MRI reconstruction. Although intially implemented in matlab, I plan to also write julia and python equivalents so this repo can be cloned and used for free.

# Data
For simplicity, a 256x256 Shepp-Logan phantom is chosen to be the ground truth, which is Fourier transformed into k-space and retrospectively undersampled. Artifical phase has also been added make the ground truth complex.
![ground truth image](https://github.com/rextlfung/cg-recon/blob/main/matlab/figs/img_gt.png)

# Model 1: Only Fourier transform
As sanity check, we begin with the simplest model possible which is just a Fourier transform (FT). As expected, the image can be perfectly reconstructed from k-space as the FT is a reversible or information preserving operation.
![Model 1](https://github.com/rextlfung/cg-recon/blob/main/matlab/figs/model1.png)
MSE = 1.1578e-30 + 1.8984e-31i
SSIM = 1.0000

# Model 2: FT and 2x undersampling in phase encoding (PE) or k_y direction
Now we add 2x undersampling to our system model. Observe the aliasing artifacts caused by undersampling. In other words, the problem Ax = y is now ill-posed.
![Model 2](https://github.com/rextlfung/cg-recon/blob/main/matlab/figs/model2.png)
![Model 2 Sampling Mask](https://github.com/rextlfung/cg-recon/blob/main/matlab/figs/model2mask.png)
MSE = 0.0184 - 0.0022i
SSIM = 0.7086

# Model 3: Multicoil, FT, and 2x k_y undersampling
How can we make the now ill-posed problem Ax = b more well-posed? One way is to add parallel imaging, where the underlying object (x) is measured by multiple receive coil arrays, which is the standard in most MRI systems nowadays. Assuming that each coil sees x in a linearly independent manner, now we effectively made A Ncoils times taller. This give us more equations for solving the unknowns in x, thus making the problem more well-posed. Read more at: https://mriquestions.com/what-is-pi.html. As shown below, with 8 synthetic coil sensitivity maps, a 2x undersampled image is now nearly perfectly reconstructed.
![Model 3](https://github.com/rextlfung/cg-recon/blob/main/matlab/figs/model3.png)
MSE = 1.3040e-14 + 4.3697e-14i
SSIM = 1.0000

# Model 3a: Model 3 but symmetric undersampling about k_y
Here I wanted to see how the recon is affected by instead making the sampling mask symmetric about k_y, which appears to be worse. One possible explanation could be that each missing k-space location no longer has an acquired data point at its conjugate location, thus the model cannot leverage conjugate symmetry (implicitly) to estimate the missing k-space.
![Model 3a](https://github.com/rextlfung/cg-recon/blob/main/matlab/figs/model3a.png)
![Model 3a sampling mask](https://github.com/rextlfung/cg-recon/blob/main/matlab/figs/model3amask.png)
MSE = -1.2969e-06 - 1.7984e-07i
SSIM = 0.9911

# Model 3b: Model 3 but 3x undersampling
Here I wanted to see how the recon would look like with 3x undersampling. It is not good.
![Model 3b](https://github.com/rextlfung/cg-recon/blob/main/matlab/figs/model3b.png)
![Model 3b sampling mask](https://github.com/rextlfung/cg-recon/blob/main/matlab/figs/model3bmask.png)
MSE = -0.0019 + 0.0014i
SSIM = 0.6528

# Model 3c: Model 3 but 6/8 partial Fourier (pf) in k_y
Adding 6/8 partial Fourier, which is a standard implementation in most MRI acquistion sequences. Read more at: https://mriquestions.com/partial-fourier.html
![Model 3c](https://github.com/rextlfung/cg-recon/blob/main/matlab/figs/model3c.png)
![Model 3c sampling mask](https://github.com/rextlfung/cg-recon/blob/main/matlab/figs/model3cmask.png)
MSE = -7.3333e-09 + 1.5834e-07i
SSIM = 0.9134

# Model 3d: Model 3 but 6/8 pf in k_x and k_y
![Model 3d](https://github.com/rextlfung/cg-recon/blob/main/matlab/figs/model3d.png)
![Model 3d sampling mask](https://github.com/rextlfung/cg-recon/blob/main/matlab/figs/model3dmask.png)
MSE = 3.4181e-04 - 6.5832e-05i
SSIM = 0.8238

# Model 3e: Model 3 but grid (CAIPI) like sampling
Here I decided to spread out the sampling pattern to see if it can aid the recon. Observe that the aliased images now appear in the diagonal direction, which leads to less overlapping with the ground truth image which has an oval shape. This technique is known as CAIPIRINHA or CAIPI. Read more at: https://mriquestions.com/caipirinha.html. In practice, this idea of evenly spreading out the undersampling is generally onl applied in the phase encoding directions as undersampling in the frequency encoding direction generally does not save much time.
![Model 3e](https://github.com/rextlfung/cg-recon/blob/main/matlab/figs/model3e.png)
![Model 3e sampling mask](https://github.com/rextlfung/cg-recon/blob/main/matlab/figs/model3emask.png)
MSE = -4.9529e-17 + 1.4571e-16i
SSIM = 1.0000

# Model 3f: Model 3 but CAIPI like sampling and pf in k_y
![Model 3f](https://github.com/rextlfung/cg-recon/blob/main/matlab/figs/model3f.png)
![Model 3f smapling mask](https://github.com/rextlfung/cg-recon/blob/main/matlab/figs/model3fmask.png)
MSE = -9.9820e-09 - 3.7154e-09i
SSIM = 0.9212

# Model 3g: Model 3 but CAIPI like sampling and pf in k_x and k_y
![Model 3g](https://github.com/rextlfung/cg-recon/blob/main/matlab/figs/model3g.png)
![Model 3g smapling mask](https://github.com/rextlfung/cg-recon/blob/main/matlab/figs/model3gmask.png)
MSE = 3.2983e-04 - 6.3412e-05i
SSIM = 0.8281
