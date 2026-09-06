# Supplementary Material: Image segmentation with traveling waves in an exactly solvable recurrent neural network

**Luisa H. B. Liboni**<sup>1,2,3,4,*</sup>, **Roberto C. Budzinski**<sup>1,2,3,4,*</sup>, **Alexandra N. Busch**<sup>1,2,3,4,*</sup>, **Sindy Löwe**<sup>5</sup>, **Thomas A. Keller**<sup>6</sup>, **Max Welling**<sup>6</sup>, and **Lyle E. Muller**<sup>1,2,3,4</sup>

<sup>1</sup>Department of Mathematics, Western University, London, ON, Canada  
<sup>2</sup>Western Institute for Neuroscience, Western University, London, ON, Canada  
<sup>3</sup>Western Academy for Advanced Research, Western University, London, ON, Canada  
<sup>4</sup>Fields Lab for Network Science, Fields Institute, Toronto, ON, Canada  
<sup>5</sup>AMLab, University of Amsterdam, Amsterdam, Netherlands  
<sup>6</sup>UvA-Bosch Delta Lab, University of Amsterdam, Amsterdam, Netherlands

\* These authors contributed equally

> **Editorial provenance (not part of the supplement).** Cleaned from the supplied supplementary-text extraction; the supplementary PDF has not been re-downloaded or checked. The [published main article](01_paper.md) is available through [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11725882/) and [PNAS](https://doi.org/10.1073/pnas.2321319121).
>
> **Mathematical transcription notes.** Equations (S1)–(S4) write the frequency as $\omega_i$, making explicit the node-specific frequency in (S5); the extraction used $\omega$. Missing imaginary units in the exponentials of (S3), and the missing coupling factor $\epsilon$ in (S4), have been restored from the adjacent change-of-variable derivation. The modal coefficient in (S7) and (S8) includes **both** $\lambda_i^k$ and $\mathbf r_i^T\mathbf x(0)$. The equality in the source's truncated expansion (S8) is retained, but is generally an approximation when discarded modes contribute. The discrete recurrence (S6) is not asserted to be an exact time discretization of the continuous ODE (S5).
>
> **Source inconsistencies retained.** Section VI refers to main-text Figure 5 for eigenanalysis, whereas the published eigenanalysis illustration is Figure 6. The unresolved citation key `shi2000normalized` in section VII has no numbered reference in the supplied extraction. The differing parameter/window specifications in section X are preserved rather than reconciled by guesswork.

---

## I. DISCRETE-TIME RECURRENT COMPLEX-VALUED DYNAMICAL SYSTEM

We start from the continuous-time nonlinear complex-valued dynamical system described by:

$$\dot{\psi}_i(t) = \omega_i + \epsilon \sum_{j=1}^{N^2} a_{ij} \left[\sin(\psi_j(t) - \psi_i(t)) - i \cos(\psi_j(t) - \psi_i(t))\right], \quad \text{(S1)}$$

where $\psi_i(t) \in \mathbb{C}$ represents the state of a node, $\omega_i \in \mathbb{R}$ specifies the frequency at which the nodes' dynamics evolve. The elements $a_{ij} \in \mathbb{R}$ determine the connections between nodes $i$ and $j$ and are scaled by the coupling parameter $\epsilon \in \mathbb{R}$.

By Euler's identity, Eq. (S1) can be rewritten as:

$$\dot{\psi}_i(t) = \omega_i - i \epsilon e^{-i\psi_i(t)} \sum_{j=1}^{N^2} a_{ij} e^{i\psi_j(t)}. \quad \text{(S2)}$$

We now multiply Eq. (S2) by $i e^{i\psi_i}$ to obtain:

$$i e^{i\psi_i(t)} \dot{\psi}_i(t) = i \omega_i e^{i\psi_i(t)} + \epsilon \sum_{j=1}^{N^2} a_{ij} e^{i\psi_j(t)}, \quad \text{(S3)}$$

and define the change of variables:

$$x_i(t) = e^{i\psi_i(t)}, \quad \dot{x}_i = i e^{i\psi_i(t)} \dot{\psi}_i(t),$$

with which Eq. (S3) becomes:

$$\dot{x}_i(t) = i \omega_i x_i(t) + \epsilon \sum_{j=1}^{N^2} a_{ij} x_j(t). \quad \text{(S4)}$$

Finally, Eq. (S4) can be written in matrix form as:

$$\dot{\mathbf{x}}(t) = (\text{diag}(i\boldsymbol{\omega}) + \epsilon \mathbf{A}) \mathbf{x}(t). \quad \text{(S5)}$$

We then consider the discrete-time version for Eq. (S5), which leads to:

$$\mathbf{x}(k + 1) = \underbrace{(\text{diag}(i\boldsymbol{\omega}) + \epsilon \mathbf{A})}_{\mathbf{B}} \mathbf{x}(k), \quad \text{(S6)}$$

where $k \in \mathbb{Z}_+$ is a positive integer.

---

## II. CONSIDERATIONS ON AMPLITUDE DYNAMICS

Because we are considering a linear complex-valued network, the nodes' amplitude $|x_i(k)|$ in Eq. (S6) naturally diverges to infinity or decreases to zero in asymptotic time. However, it is essential to notice that, for transient time, the amplitude and phase dynamics of the cv-NN display sophisticated behavior, which we leverage to perform calculations, more specifically, object segmentation.

Modulating the eigenvalues of $\mathbf{B}$ in Eq. (S6) can extend the transient time, resulting in the phase dynamics evolving nontrivially while the amplitudes remain finite, which we call the *feasible period*. Therefore, in the feasible period, the amplitudes are still numerically bounded, and therefore, calculations are numerically feasible. The phase dynamics simulations depicted in Figure 2 of the main text are shown in Movie 1 and Movie 2, where it is clear that object-centric spatiotemporal patterns emerge and imprint each object, enabling further segmentation. Movie 3 further shows the amplitude dynamics for the same test case used in Movie 1.

---

## III. SEGMENTATION PERFORMANCE

Although performance for object segmentation is not the primary goal of this paper, the general segmentation algorithm we propose aims to help us better understand the dynamics of the complex-valued dynamical system and draw a baseline for the robustness of this object-centric representation. As mentioned in the main text, object-centric representations are obtained after segmenting the background in the first layer and without any training mechanism. The K-means algorithm is applied to the similarity low-dimensional representation to obtain clusters of nodes with similar dynamics.

By applying the segmentation approach for forty timesteps $T = 40$ of the dynamics, we obtain an accuracy of 93% of pixels on 1000 images drawn from the 2Shapes dataset. Furthermore, 86% of pixels were correctly clustered in 1000 images drawn from the 3Shapes dataset. Both analyses considered nonoverlapping configurations. Neighboring object configurations were also excluded from the testing dataset. These performances underscore the object-centric representation's remarkable efficiency, obtained without any training mechanism and with a very low computational burden.

![Published main-paper Figure 3: two-layer object segmentation](01_figures/pnas.2321319121fig03.jpg)

*Related illustration: published **main-paper Figure 3**, showing the two-layer algorithm evaluated here. This is not supplementary Figure S1.*

---

## IV. CONSIDERATIONS REGARDING OVERLAPPING CASES

As discussed in the main text (Figure 5), the nodes in the overlapping area co-participate in both spatiotemporal propagations that arise in the phase dynamics. This result shows that, in that region, the objects overlap, and one can retrieve the exact overlapping location from the pixel space. For naturalistic images, where different objects have slightly different pixel intensity values, the spatiotemporal patterns have been shown to be unique to each object.

We now extend our analysis to those images where different objects have the same pixel intensity. In Figure S1, we depict the similarity projection of the overlapping cases from the 2Shapes dataset, in which the foreground square and triangle intensity are given the value of 1, with different overlapping percentages. Therefore, the objects are nonseparable in the input space.

As one would expect, the more significant the percentage of overlap, the more difficult it is to cluster the spatiotemporal patterns into different object-specific representations. Given that no training mechanism is used, the spatiotemporal patterns obtained in these overlapping cases were remarkable.

**Figure S1 (Missing):** An input with non-disjoint foreground figures with a higher overlap percentage. The overlapping nodes can be identified as sharing two clusters.
*Note: The supplement figure for S1 is not available in the local figure directory. This depicts the similarity projection in three dimensions (labels, input dynamics, and projection).*

---

## V. MECHANISTIC INTERPRETATION OF THE CV-RNN AND THE ROLE OF THE HYPERPARAMETERS

The cv-RNN dynamics can be understood through the eigendecomposition:

$$\mathbf{x}(k) = \mathbf{B}^k \mathbf{x}(0) = \sum_{i=1}^{N^2} \underbrace{\lambda_i^k \left(\mathbf{r}_i^T \mathbf{x}(0)\right)}_{\mu_i(k)} \mathbf{v}_i, \quad \text{(S7)}$$

where $\lambda_i$ are the eigenvalues associated with eigenvectors $\mathbf{v}_i$ of $\mathbf{B}$, $\mathbf{r}_i^T$ are the rows of $[\mathbf{v}_1 \cdots \mathbf{v}_{N^2}]^{-1}$, and coefficients $\mu_i(k)$, which depend on the initial conditions and eigenvalue, weight the contribution of each eigenvector.

This complex-valued network is, in fact, a system of oscillators, a connection we have recently developed in depth [1]. In this previous work, we developed an operator-based approach where the dynamics of real-valued Kuramoto oscillators could be matched exactly by using two operators to evolve a complex-valued system: the first is a linear operator, which rotates the oscillators in the complex plane; the second is a nonlinear operator, which normalizes the amplitudes of the nodes, so that the dynamics in the argument of the complex-valued system precisely match individual trajectories of the original, real-valued Kuramoto system [2]. Here, we leverage the mathematical insights developed along this line of research to introduce a new network that can perform computations while also being exactly solvable. To do this, we now drop the second, nonlinear operator, to study a system with sophisticated dynamics governed fully by a linear operator. It is these linear dynamics, specifically when looking at the phase values of the oscillators, that underlie computation in the cv-RNN.

### A. Mechanistic Interpretation

We analyze the recurrent dynamics in the cv-RNN through the connection to our analysis of oscillator networks. Nodes in the cv-RNN are complex-valued oscillators whose natural frequencies are modulated by input images. These inputs interact with the recurrent dynamics specified by the topographic connectivity of the network to produce the spatiotemporal patterns used for image segmentation. In practice, nodes receiving input from pixels within delimited regions (similar pixel intensities and near to each other) tend to synchronize. Moreover, propagations will occur because of the phase differences between different delimited areas.

These dynamics can be understood through our eigendecomposition analysis of the composite matrix $\mathbf{B}$, which contains the input image in diagonal entries, and captures the recurrent connectivity scheme in off-diagonal entries. Eigendecomposition can thus reveal the mechanism underlying segmentation in the cv-RNN, by explaining the interaction between image inputs and the underlying recurrent dynamics of the system. Image inputs to the cv-RNN shift the spectrum of the composite matrix $\mathbf{B}$, changing the contribution of eigenmodes in the system, and thus shaping the phase patterns that emerge.

Consider for example a network with local connectivity (Fig. S2, left column). In this case, nodes are weakly affected by their neighbors, and the system does not present a prominent leading eigenvector. In practice, spatiotemporal dynamics occur with too fine a granularity to properly segment the objects (Fig. S2c, left). On the other hand, a network with stronger, global connectivity will quickly synchronize due to a leading eigenvalue with very large modulus. In this case, objects cannot be segmented because the network is fully synchronized (Fig. S2c, right).

However, when image inputs are combined with optimal recurrent connectivity, the eigenspectrum of the system shapes object-specific waves in the cv-RNN (Fig. S2b,c, middle). Eigendecomposition of the system thus provides a mechanistic understanding for how the interaction between input images and recurrent connections in the network drive the nodes to cluster into groups with separable, object-specific spatiotemporal dynamics.

**Figure S2 (Missing):** Example of the influence of network connectivity on segmentation. (a) We begin by inputting an image into layer 1 of our cv-RNN to obtain the mask for the three objects. We then manipulate the connectivity scheme in layer 2 to investigate the resulting segmentation. (b) Our cv-RNN (middle column) results in object-specific waves. When connectivity is too local (left column), the spatiotemporal patterns that appear are choppy and local in space. When connectivity is too global (right column), the system quickly synchronizes (corresponding to a very large modulus first eigenvalue). (c) The segmentation labels resulting from the spatiotemporal patterns in each network are plotted. The local network assigns pixels within each object to many different clusters, while the more global network assigns all objects to the same cluster. Both of these results are as expected from the type of dynamics displayed.
*Note: Figure S2 is not available in the local figure directory.*

### B. The Role of the Hyperparameters

The hyperparameters in the cv-RNN are $\epsilon$ and $\sigma$, which control the peak strength and spatial extent of recurrent connections, respectively. These two hyperparameters have a clear physical interpretation, which we can understand through the connection with our analysis of coupled oscillators. From this perspective, cv-RNN nodes with stronger coupling will tend to quickly synchronize in phase, within regions determined by the spatial extent of recurrent connections, and nodes with weaker coupling will tend to synchronize much more slowly. Consistent with our understanding of coupled oscillator dynamics, then, $\epsilon$ controls, in general, the time to synchronize in the network. $\sigma$, as well, has an equally straightforward physical interpretation, as it will control the spatial extent of recurrent interactions in the network. The specific point we find in $(\epsilon, \sigma)$ space, then, corresponds to a network that will generate traveling waves on the general spatial scale of objects in the image, during a sufficiently extended transient period. (See Fig. S2 for an example of how varying these parameters affects segmentation.)

### C. Tuning Hyperparameters

To tune $\epsilon$ and $\sigma$, we conducted a grid search using only two images from the dataset of images with two geometric shapes. Considering only two example images at this point renders even a systematic search over the two-dimensional parameter space fast and effective. Further, even considering only two images for the tuning, we find an impressive amount of generalization: the cv-RNN that segments these two images then generalizes directly to the rest of the datasets considered in this work, including images with two or three shapes, combinations of shapes and MNIST digits, and the handful of natural images we studied. The ease of fine-tuning the hyperparameters in our approach is supported by the simple interpretation of the cv-RNN as a network of coupled oscillators able to generate object-specific wave patterns.

---

## VI. LOW-RANK SPATIOTEMPORAL RECONSTRUCTION

It is well-known that the eigenvectors of linear dynamical systems shape the resulting dynamics. Therefore, the linear combination of eigenvectors generates the spatiotemporal patterns that arise during the transient dynamics of the cv-RNN. To show this, we use the eigenanalysis as exhibited in Figure 5 of the main text. As discussed, a finite number of eigenvalues of $\mathbf{B}^{(2)}$ have noticeably significant amplitudes with phases ranging from 1.20 to 1.57 rad, and the arguments of the eigenvectors are shaping bits and pieces of objects with a phase of $\pi$ rad. Therefore, a low-rank reconstruction of the dynamics in Eq. (S6) can be obtained:

$$\mathbf{x}(k) = \mathbf{B}^k \mathbf{x}(0) = \sum_{i=1}^{n} \underbrace{\lambda_i^k \left(\mathbf{r}_i^T \mathbf{x}(0)\right)}_{\mu_i(k)} \mathbf{v}_i, \quad \text{(S8)}$$

where the linear combination of the eigenvectors $\mathbf{v}_i$ is truncated for a low-rank order $n$, $\mathbf{r}_i^T$ are the rows of $[\mathbf{v}_1 \cdots \mathbf{v}_{N^2}]^{-1}$, and $\mu_i(k)$ denotes the contribution of each eigenvector.

The contributions are dependent on the initial condition and also on the eigenvalues. Each contribution will rotate the phases of the nodes at each timestep $k$, giving rise to the spatiotemporal dynamics.

![Published main-paper Figure 6: eigenmode interpretation](01_figures/pnas.2321319121fig06.jpg)

*Related illustration: published **main-paper Figure 6**. The supplied supplement retains its original figure references; this published image is not supplementary Figure S2 or S3.*

---

## VII. COMPUTATIONAL COMPLEXITY

The computational complexity of the cv-RNN in inference time is $O(T N^4)$ where $T$ is the number of timesteps of recurrent dynamics, and $N^2$ is the dimension of the state vector. The evolution of recurrent dynamics is followed by a clustering algorithm of complexity $O(L N^2)$, with $L$ accounting for linear factors related to the number of clusters, dimension of projection, and time to convergence. Image segmentation algorithms, such as normalized cuts (`shi2000normalized`, unresolved source citation), watershed approaches [3], graph cuts [4], and convolutional networks [5] operate directly on pixel values within a local neighborhood and have computational complexity $O(N^2)$ to $O(N^6)$. This range of computational complexity demonstrates that, in the current implementation, the cv-RNN algorithm falls within the middle of the range of computational complexities for other image segmentation approaches. It is important to note, in addition, that the cv-RNN requires tuning only two hyperparameters specifying recurrent connectivity in the network, reducing the overall computational burden compared to training deep learning architectures, which require tuning many hyperparameters for successful performance. Finally, since the cv-RNN (equation 5 of the main text) is a linear-time-invariant recurrence (as with SSMs), by using a parallel associative scan algorithm [6], it is possible to parallelize this over sequence length, to get $O(\log(T))$ complexity for sequence length, which would result in $O(\log(T) N^4)$. The linearity of the system is the feature enabling a scan like this, making the system easily scalable.

---

## VIII. EXTRA RESULTS

Figure S4 shows segmentation results for other naturalistic images selected from the BSDS500 dataset [7]. The images were resized to 64 × 64 pixels size and z-scored. It is clear that the object segmentation algorithm can remove significant portions of the background and make segmentation possible through its dynamics. For nonoverlapping objects in the naturalistic scenes, the network achieved the same very peculiar property: nodes corresponding to pixels of a particular object are locked to unique phase values during some timesteps of the feasible transient regime.

**Figure S3 (Missing):** (a) Each pixel in an input image projects to one node in the cv-RNN. The cv-RNN network is a 2D sheet with distance-dependent connectivity, which is uniform across nodes. Each node is an oscillator in the complex plane. Input images influence the natural frequency of the corresponding node. Image inputs interact with the recurrent dynamics of the cv-RNN, to produce spatiotemporal patterns of activity in the network that can be used to segment images. (b) A pooling layer can be added so that an input image may project to a smaller cv-RNN. The activity patterns in the cv-RNN can still be used to segment the image, by first clustering the cv-RNN nodes, then reversing the operation performed by the pooling layer. (c) An example of one image segmented by a 2-layer cv-RNN with half the number of nodes as the pixels in the image.
*Note: Figure S3 is not available in the local figure directory.*

**Figure S4 (Missing):** The original image (first column), input image (second column), and output (third column) of K-means clustering are plotted for 3 example naturalistic images.
*Note: Figure S4 is not available in the local figure directory. The paper demonstrates segmentation of naturalistic images from the BSDS500 dataset.*

---

## IX. MOVIES

The following eight movies accompany the main text and supplement, demonstrating various aspects of the cv-RNN dynamics and segmentation:

### A. Movie 1: Phase Dynamics for 2Shapes Dataset

In this movie, we use Eq. (S6) to simulate the complex-valued linear system with a visual input image from the 2Shapes dataset containing a triangle and a square (same input used in Figure 2a of the main text). The black pixels are assigned to −1 and the white pixels to 1. The movie shows the phases $\text{Arg}[\mathbf{x}(k)]$. At the beginning of the movie, we capture the phase's random initial condition $\text{Arg}[\mathbf{x}(0)]$. Closer nodes, which correspond to closer pixels, are strongly connected since the connectivity matrix is given by the decaying Gaussian rule in Eq. (4) from the main text. One can observe that the nodes corresponding to the background and the distinct foreground objects are being pushed by the $\text{diag}(i\boldsymbol{\omega})$ to oscillate at different intrinsic frequencies towards the end of the movie. This oscillation phenomenon agrees with what happens in continuous-time oscillators modeled by Eq. (S5). The rationale behind this correspondence is linked to the modification caused by the diagonal encoding of the visual input, which leads to changes in the intrinsic frequencies of individual nodes.

### B. Movie 2: Segmentation with MNIST and Shapes

Movie 2 shows the segmentation accomplished by the network's arguments for image input from the MNIST&Shapes dataset, where a triangle and a handwritten number three are combined, as shown in Figure 2b of the main text. We use the same connectivity parameters as the ones used in Movie 1. The digit is rendered in grayscale values to capture boundary effects. Note that the first pattern in the grid is being propagated through the nodes corresponding, in the input space, to the triangle. Another unique spatiotemporal pattern generated by the dynamics is propagating within the nodes related, in the input space, to the background, and, lastly, the third unique pattern that propagates is related to the nodes corresponding, in the input space, to the handwritten digit. Note that the different spatiotemporal patterns imprint the objects through their different progression in time.

### C. Movie 3: Amplitude Dynamics

Movie 3 shows the amplitude dynamics of the state vector $\mathbf{x}[k]$ reshaped into an $N \times N$ grid for the cv-RNN in the same conditions as considered in Movie 1. The initial value for the amplitude vector is $|\mathbf{x}(0)| = 1$.

Given that this system is not stable, the amplitudes are shown to grow with $k$. We can perform image segmentation with the phase dynamics. However, as shown in Movie 3, the amplitude dynamics of the cv-RNN are non-trivial during the transient time. Although we have not explored the amplitude dynamics in this paper, we expect to see more specialized algorithms arising in the future, which could, by applying a nonlinear operator, constrain the amplitudes and then leverage its information to improve object segmentation.

### D. Movie 4: Background Removal for Naturalistic Images

In this movie, we show the first step of the background removal for the coin naturalistic visual input. In this input, the coins are considered to have z-scored grayscale intensity. At $k = 60$, the background is removed by disregarding the nodes with higher frequency on the phase distribution and masking them so that they do not account for the second layer's system dynamics.

The second layer dynamics begins from a random phase condition for the unmasked nodes. Now, the phases of the nodes corresponding only to the foreground objects will progress. In the case of nonoverlapping objects, the nodes synchronize their phases to a unique value for each object while maintaining the background phase static. For completely binary inputs with a masked background, the rationale of why the nodes synchronize to a unique value for each object resides in the fact that the network is given a random starting point, which gives a stochastic difference for the phase synchronization for each object. In the case of grayscale ranges for the foreground objects and with the masked background, given that a continuum exists across the pixels within the same object, synchronization will happen in a small range of phases for each object.

### E. Movie 5: Overlapping Objects Dynamics

In the left panel of Movie 5, we show the spatiotemporal patterns generated for the case of overlapping objects from Figure 5 of the main text. In the right panel, we show the spatiotemporal patterns of the case of a 2Shapes image with a higher percentage of pixel overlap. We can see that the dynamical system cannot synchronize the nodes uniquely corresponding to each object because of the overlapping parts. However, the spatiotemporal patterns can again imprint each object, corroborating that the spatiotemporal patterns are unique propagations throughout the nodes mapped to each particular object and further separable. Because the inputs are not considered different objects in the raw image, the segmentation performance decreases with the amount of overlapping pixels.

### F. Movie 6: Low-Rank Eigenvector Reconstruction

In this movie, the same system used in Figure 6 of the main text is simulated through the linear combination of the six eigenvectors corresponding to the six most significant eigenvalues ($i \in [1, 6]$ in Eq. (S8)). We show that the segmentation can be achieved by using only a few numbers of eigenvectors to approximately reproduce the spatiotemporal pattern. These results demonstrate that the dynamics within a period of interest can be calculated very efficiently using low-rank approximations in their calculations. Therefore, note that by having matrix $\mathbf{B}^{(2)}$ obtained after masking $\mathbf{B}^{(1)}$, which is known a priori, one can easily calculate a few eigenvalues to generate approximately the spatiotemporal patterns by using just a few combinations of eigenvectors.

### G. Movie 7: Eigenvectors for Panel (a) of Figure 6

This movie corresponds to panel a of Figure 6 in the main text. The input image is depicted, followed by the eigenvectors corresponding to the ten most significant eigenvalues. The dynamics of the cv-RNN are then played.

### H. Movie 8: Eigenvectors for Panel (b) of Figure 6

This movie corresponds to panel b of Figure 6 in the main text. The input image is depicted, followed by the eigenvectors corresponding to the ten most significant eigenvalues. The dynamics of the cv-RNN are then played.

---

## X. COMPUTATIONAL DETAILS AND PARAMETERS

In all cases, the connection architecture of the cv-RNN is given by a distance-dependent lattice (Eq. (4) in the main paper) with $\epsilon = 1.0$, $\phi = 0$. The other relevant parameters can be found below:

### Main Text Parameters

| Figure | Connectivity | $\alpha$ | $\sigma$ | $T$ as transcribed |
| --- | --- | --- | --- | --- |
| 2a, 2b | $A$ | 0.2 | 0.0312 | Not specified |
| 3b | $A_1$ | 0.5 | 0.9 | 141–181 |
| 3c | $A_2$ | 0.5 | 0.0313 | 141–181 |
| 4a, 4b, 4c | $A_1$ / $A_2$ | 0.5 / 0.5 | 0.9 / 0.0313 | 141–181 |
| 5a, 5b, 5c | $A_1$ / $A_2$ | 0.5 / 0.5 | 0.9 / 0.0313 | 121–141 |
| 6 | $A_1$ / $A_2$ | 0.5 / 0.5 | 0.9 / 0.0313 | 121–141 |

### Supplementary Material Parameters

| Figure or movie | Connectivity | $\alpha$ | $\sigma$ | $T$ as transcribed |
| --- | --- | --- | --- | --- |
| S1 | $A_1$ / $A_2$ | 0.5 / 0.5 | 0.9 / 0.0313 | 141–181 |
| S2 | $A_1$ / $A_2$ | 0.5 / 0.5 | 0.9 / 0.0313 | 141–181 |
| Movies 1, 2, 3 | $A$ | 0.08 | 0.0313 | Not specified |
| Movies 4, 5 | $A_1$ / $A_2$ | 0.5 / 0.5 | 0.9 / 0.0313 | 141–181; also 121–141, with $\sigma=0.0313$ repeated |
| Movie 6 | $A$ | 0.08 | 0.0313 | Not specified |
| Movies 7, 8 | $A_1$ / $A_2$ | 0.5 / 0.5 | 0.9 / 0.0313 | Not specified |

*The original hyphen between a matrix name and $\alpha$ was a label separator, not subtraction. The Movies 4/5 row contains two windows without a clear assignment; no ordering between them is inferred.*

---

## XI. CODE AVAILABILITY

An open-source code repository for this work is available on GitHub: http://mullerlab.github.io

---

## References

[1] L. Muller, J. Mináč, and T. T. Nguyen, Algebraic approach to the Kuramoto model, *Phys. Rev. E* **104**, L022201 (2021).

[2] R. C. Budzinski, T. T. Nguyen, J. Doãn, J. Mináč, T. J. Sejnowski, and L. E. Muller, Geometry unites synchrony, chimeras, and waves in nonlinear oscillator networks, *Chaos* **32**, 031104 (2022).

[3] J. B. Roerdink and A. Meijster, The watershed transform: Definitions, algorithms and parallelization strategies, *Fundam. Informaticae* **41**, 187 (2000).

[4] Y. Boykov and V. Kolmogorov, An experimental comparison of min-cut/max-flow algorithms for energy minimization in vision, *IEEE Trans. Pattern Anal. Mach. Intell.* **26**, 1124 (2004).

[5] K. He and J. Sun, Convolutional neural networks at constrained time cost, in *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition* (2015), pp. 5353–5360.

[6] E. Martin and C. Cundy, Parallelizing linear recurrent neural nets over sequence length, arXiv preprint arXiv:1709.04057 (2017).

[7] D. Martin, C. Fowlkes, D. Tal, and J. Malik, A database of human segmented natural images and its application to evaluating segmentation algorithms and measuring ecological statistics, in *Proc. 8th Int'l Conf. Computer Vision*, Vol. 2 (2001), pp. 416–423.

---

**Supplementary artwork status:** Figures S1–S4 are not included in the local archive. S1 concerns overlap, S2 connectivity effects, S3 pooling, and S4 additional natural images. Their captions are retained above. The seven files in `01_figures/` are published main-paper figures, not substitutes for the missing supplementary artwork.
