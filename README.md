# Specular-optimization
We aim at recovering specular albedo and shininess in a photometric stereo procedure.

The adopted camera model is pinhole.

The image model for $I$ is $I = I_{\text{diffuse}} + I_{\text{specular}}$, where we have $I_{\text{specular}}=\rho_s(c+2)*\max(0, h \cdot n)^c \max(0, s\cdot n)\phi$, where $\rho_s$ is the specular albedo, $c$ the shininess, $h$ the half-way vector at the surface point, $n$ the normal vector, $s$ is the source (normalized) direction, $\phi$ its intensity.

We run an alternating optimization procedure, aiming at solving:

$$\text{given } I,I_{\text{diffuse}}, h, n, s, \phi, \quad \text{find } \rho_s, c \text{ with}\quad I - I_{\text{diffuse}}=\rho_s(c+2)*\max(0, h \cdot n)^c \max(0, s\cdot n)\phi $$

Applying the logarithm on both sides, this is equivalent to solving:

$$\delta -(\eta + c \alpha) = 0 $$

where $\delta = \log( I - I_{\text{diffuse}})-\beta$, $\beta = \log(\max(0, s\cdot n)\phi)$, $\alpha = \log(\max(0, h \cdot n))$, $\eta= \log(c+2)\rho_s$.

To the $\max$ terms, a spherical gaussian approximation can be applied, yielding a modification of this standard Phong BRDF.

Moreover, calling the residual $r:=\delta -(\eta + c \alpha)$, we can use the theory of robust estimators to develop an alternatively reweighted least squares algorithm. In fact our model, can be cast in a regression-like form:

$$s=h(b)+\epsilon$$

where abstractly, $s$ is data, $b$ are the parameters to be estimated from the knowledge of $s$, $\epsilon$ is noise. For $\Phi$ a robust estimator, and $r(b)=\log(s)-\log(h(b))$, the logarthmic optimization reads:

$$\min_b \Phi(s(1-e^{-r(b)}))$$

The corresponding weight for reweighted optimization is:

$$w(r) = \frac{\Phi'(s(1-e^{-r(b)}))se^{-r}}{r}$$

so that for a fixed $w_i=w(r_i^-)$, we aim at solving:

$$\min_b\sum_i w(r_i^-)r_i(b)^2$$

The best performing robust estimator is the Cauchy one.

## Limitations

- wherever a part of the image is free from observations of specularities, there the optimization is ill posed, so that the specular albedo explodes

## Improvements

- connect this purely specular optimization in an alternating fashion to a full photometric stereo pipeline

## Notes

- the algorithm performance can vary substantially according to the choice of the initial specular albedo, and the variance parameter in the Cauchy estimator

# Code

It is in Matlab, and mainly contained in `launch.m`, which the user can run (just by pressing the "Run" button in Matlab) to perform the optimization. The various options for the optimization are described at the beginning of the script.

Two ready datasets are provided in the folder `Datasets`. Best parameters configurations for both of them are also indicated in `launch.m`.

These are generated using OpenGL and the script `Datasets\opengl2matlab.m`.

# Data

The user should generate his/her data with the script `opengl2matlab.m`, that can be just run using the "Run" button of Matlab.

In turn, this file also needs some input before being run: we now explain how to generate such input.

Beforehands, we explain how to setup the OpenGL data generation, with Microsoft Visual Studio 2022. The instructions work on Windows 10. We are all the time working in the folder `OpenGL`.

## Configuring the Visual Studio Project
- set up glad, glfw, assimp as in `https://learnopengl.com/Getting-started/Creating-a-window`, `https://learnopengl.com/Model-Loading/Assimp`. Remember to put `assimp-vc143-mtd.lib` under the `lib`
- create a new project in the folder `OpenGL`, call it `data_generation`
- copy `src, lib, include, data` in `OpenGL`
- add `glad.c` as a source file for the `data_generation` solution
- add as new element the file `rgbd_generator.cpp`
- copy/paste the source code inside `rgbd_generator.cpp`
- add the `lib` and `include` directories in the `solution properties > configuration properties > VC++ directories > general > library directory` and `inclusion directories`. The two folders must be in front of the other
- prepend `assimp-vc143-mtd.lib;opengl32.lib;glfw3.lib;` to `solution properties > configuration properties > linker > additional dependencies`
- add the `lib`folder to `solution properties > configuration properties > linker > general > additional library directories`
- add the `bin` path to `solution properties > configuration properties > debug > environment`: use the syntax `PATH=C:\path\to\bin;%PATH%`
- add `mesh.h, conf.h, model.h` in the header files of the solution

## Running the C++ code
- the user has to provide a textured geometric model. For the two aforementioned examples, we provide all the needed files. It is important to set up the `.mtl` file correctly, a guide on how to do this is avaiable here: https://www.youtube.com/watch?v=4DQquG_o-Ac&ab_channel=Code%2CTech%2CandTutorials. The model folder has to be specified in `OpenGL\conf.h` through the string `model`
- with reference to `OpenGL\conf.h`, manually the user has specify an `out_folder`, which must exist in the system. Everything will be saved there
- then, run the main file `OpenGL\depth_map.cpp`. A window will pop up. The navigation is locked, press `u` or `l` to unlock or lock the navigation. With the keys `a,w,s,d` the user can translate the camera, with the mouse position the orientation can be changed, a zoom effect is available with the mouse wheel. With the keys, the light can be modified, just for visualization purposes. Once `enter` is pressed, a series of pictures of the object are rendered, see `OpenGL\conf.h` for more details. Press `esc` to exit
- the shaders can be changed in `OpenGL\conf.h`. There the user can implement any BRDF function. Phong is the default one

What is happening down the line: the object is illuminated at different angles with directional lighting, and HDR RGBD images (with normals etc) are saved. HDR is crucial in specular optimization, because specular observation sometimes ecceed the usual range 0-255.



