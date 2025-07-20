## Geodesic Light: photons in a black-hole spacetime

Python code to integrate the null (or light-like) **geodesics** of a **black-hole spacetime** in General Relativity. This means looking at the [**photons**](https://en.wikipedia.org/wiki/Photon) orbiting the (possibly) rotating black hole.

Here we follow closely the method presented in [Pu et al. (2016)](https://ui.adsabs.harvard.edu/abs/2016ApJ...820..105P/abstract). This software derives from the tutorials of the *Black Holes and Neutron Stars* course (Dr. C. Fromm and Dr. R. Gold) at the ITP of the Goethe-Universität (Frankfurt am Main).

## Relativistic Astrophysics background (geeky section 🤓)

Using geometrised units, $c=G=1$, the [Kerr metric](http://www.roma1.infn.it/teongrav/leonardo/bh/bhcap3.pdf) in Boyer-Lindquist coordinates is written as
$$\mathrm{d}s^{2}=-\left(1-\frac{2Mr}{\Sigma}\right) \mathrm{d}t^{2}-\frac{4aMr\sin^{2}\theta}{\Sigma} \mathrm{d}t \mathrm{d}\phi+\frac{\Sigma}{\Delta} \mathrm{d}r^{2}+\Sigma \mathrm{d}\theta^{2}+\left(r^{2}+a^{2}+\frac{2a^{2}Mr\sin^{2}\theta}{\Sigma}\right) \sin^{2}\theta \mathrm{d}\phi^{2},$$
where $\Sigma\equiv r^2 + a^2 \cos^2 \theta$ and $\Delta\equiv r^2 - 2Mr + a^2$. For simplicity, we will set also $M=1$.