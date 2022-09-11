# Specular-optimization
We aim at recovering specular albedo and shininess in a photometric stereo procedure.

The adopted camera model is pinhole.

The image model for $I$ is $I = I_{\text{diffuse}} + I_{\text{specular}}$, where we have $I_{\text{specular}}=\rho_s(c+2)*\max(0, h \cdot n)^c \max(0, s\cdot n)\phi$, where $\rho_s$ is the specular albedo, $c$ the shininess, $h$ the half-way vector at the surface point, $n$ the normal vector, $s$ is the source (normalized) direction, $\phi$ its intensity.

We run an alternating optimization procedure, aiming at solving:

$$\text{given } I,I_{\text{diffuse}}, h, n, s, \phi, \quad \text{find } \rho_s, c \text{ with}\quad I - I_{\text{diffuse}}=\rho_s(c+2)*\max(0, h \cdot n)^c \max(0, s\cdot n)\phi $$

Applying the logarithm on both sides, this is equivalent to solving:

$$\delta -(\eta + c \alpha) = 0 $$

where $\delta = \log( I - I_{\text{diffuse}})-\beta$, $\beta = \max(0, s\cdot n)\phi$, $\alpha = \log(\max(0, h \cdot n))$, $\eta= \log(c+2)\rho_s$.

To the $\max$ term, a spherical gaussian approximation can be applied.

Calling the residual $r:=\delta -(\eta + c \alpha)$, we can use the theory of robust estimators to developed an alternatively reweighted least squares algorithm. In fact our model can be cast in a regression-like form:

$$s=h(b)+\epsilon$$

where $s$ is data, $b$ is to estimate, $\epsilon$ is noise. For $\Phi$ a robust estimator, and $r$ as above (in this notation, $r(b)=\log(s)-\log(h(b))$, the logarthmic optimization reads:

$$\min_b \Phi(s(1-e^{-r(b)}))$$

The corresponding weight for reweighted optimization is:

$$w(r) = \frac{\Phi'(s(1-e^{-r(b)}))se^{-r}}{r}$$

## Limitations

- wherever a part of the image is free from observations of specularities, there the optimization is ill posed, so that the specular albedo explodes
