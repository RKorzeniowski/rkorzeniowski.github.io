# Normalizing Flows

Normalizing flows have emerged as a powerful framework in deep generative modeling, allowing us to model complex data distributions by transforming a simple, known probability density (like a Gaussian) through a series of invertible transformations. In this post, we’ll dive deep into the mathematics behind normalizing flows, explore the statistical principles underpinning them, and compare them with other popular models such as Variational Autoencoders (VAEs) and Generative Adversarial Networks (GANs).

---

## 1. The Mathematical Basis of Normalizing Flows

### 1.1. Change-of-Variables Formula

At the heart of normalizing flows is the **change-of-variables formula** in probability theory. Suppose we have an invertible and differentiable function $f$ that maps a latent variable $\mathbf{z}$ (with a simple density $p_Z( \mathbf{z})$ ) to a data point $\mathbf{x}$ as follows:

$$
\mathbf{x} = f(\mathbf{z})
$$

The density $p_X(\mathbf{x})$ of $\mathbf{x}$ can be expressed by:

$$
p_X(\mathbf{x}) = p_Z(\mathbf{z}) \left| \det \frac{\partial f^{-1}(\mathbf{x})}{\partial \mathbf{x}} \right|
$$

Alternatively, using the inverse transformation:

$$
p_X(\mathbf{x}) = p_Z(f^{-1}(\mathbf{x})) \left| \det \frac{\partial f^{-1}(\mathbf{x})}{\partial \mathbf{x}} \right|
$$

In practice, we often use the forward transformation and compute the determinant of the Jacobian of $f$ itself:

$$
\log p_X(\mathbf{x}) = \log p_Z(\mathbf{z}) - \log \left| \det \frac{\partial f(\mathbf{z})}{\partial \mathbf{z}} \right|
$$

This formulation allows us to evaluate the exact likelihood of data points, which is a significant advantage over many other generative models.

### 1.2. Building a Flow

A normalizing flow is typically built by composing several simple transformations:

$$
f = f_K \circ f_{K-1} \circ \cdots \circ f_1
$$

Each transformation $f_k$ is chosen such that its Jacobian determinant is either tractable or efficiently computable. The overall log-likelihood becomes:

$$
\log p_X( \mathbf{x}) = \log p_Z(f_1^{-1} \circ \cdots \circ f_K^{-1}( \mathbf{x})) - \sum_{k=1}^K \log \left| \det \dfrac{\partial f_k( z_{k-1})}{\partial z_{k-1}} \right|
$$

where $\mathbf{z}_0 = \mathbf{z}$ and $\mathbf{z}_K = \mathbf{x}$.

### 1.3. Pseudocode Example

Below is a simplified pseudocode illustrating the forward pass of a normalizing flow:

```python
# Define a simple normalizing flow transformation
def normalizing_flow_forward(z, transformations):
    log_det_total = 0
    for f in transformations:
        z, log_det = f.forward(z)
        log_det_total += log_det
    return z, log_det_total

# Example transformation: Affine Coupling Layer
class AffineCouplingLayer:
    def __init__(self, scale_net, translate_net):
        self.scale_net = scale_net
        self.translate_net = translate_net

    def forward(self, z):
        # Split the input
        z1, z2 = split(z)
        s = self.scale_net(z1)  # scale factors
        t = self.translate_net(z1)  # translation factors

        # Compute transformed z2
        z2_transformed = z2 * exp(s) + t

        # Concatenate to form new z
        z_new = concatenate(z1, z2_transformed)

        # Log determinant of the Jacobian
        log_det = sum(s)  # Sum over dimensions of z2
        return z_new, log_det
```

This pseudocode highlights how each layer contributes both a transformed variable and a log-determinant, essential for computing the exact likelihood.

---

## 2. Statistical Insights

Normalizing flows rely on fundamental statistical principles:

- **Exact Likelihood Computation:** Unlike models that rely on variational approximations (such as VAEs), flows allow us to compute the exact log-likelihood, which can be directly optimized.
  
- **Invertibility:** The invertibility of the transformation ensures that no information is lost in the mapping process, preserving the statistical structure of the data.
  
- **Flexible Density Estimation:** By stacking multiple transformations, normalizing flows can model highly complex, multi-modal distributions while starting from a simple base distribution (e.g., a standard Gaussian).

- **Optimization and Stability:** The tractable likelihood and the deterministic nature of the transformations lead to more stable training dynamics compared to adversarial training in GANs.

---

## 3. Comparing Normalizing Flows with VAEs and GANs

### 3.1. Variational Autoencoders (VAEs)

- **Inference:** VAEs use an encoder to approximate the posterior distribution $q(\mathbf{z}|\mathbf{x})$ and a decoder to generate samples. They optimize a variational lower bound (ELBO) rather than the exact likelihood.
- **Latent Space:** The latent space in VAEs is often assumed to be a simple distribution (like a Gaussian), but the true posterior can be complex. Flows can be used to improve the flexibility of the VAE posterior (via *normalizing flow variational inference*).
- **Training:** Training VAEs involves stochastic gradient variational Bayes, which introduces some approximation error. In contrast, normalizing flows train using exact likelihood maximization.

### 3.2. Generative Adversarial Networks (GANs)

- **Adversarial Training:** GANs rely on a discriminator to guide the generator, which can lead to issues like mode collapse and training instability. Normalizing flows, by contrast, use direct likelihood-based training.
- **Likelihood Evaluation:** GANs do not provide a direct estimate of the likelihood, making it difficult to quantitatively evaluate their performance. Normalizing flows, with their exact likelihood, offer a transparent evaluation metric.
- **Invertibility:** Normalizing flows ensure that every generated sample has a corresponding latent representation, a property that GANs lack. This bidirectional mapping is useful for tasks such as representation learning and data manipulation.

---

## 4. Applications: Flow-TTS

One exciting application of normalizing flows is in the realm of text-to-speech synthesis. **Flow-TTS** leverages normalizing flows to generate high-quality audio from text by modeling the complex distribution of speech signals. The invertibility and exact likelihood estimation provided by normalizing flows help in capturing the nuances of speech, enabling more natural-sounding synthesis.

For those interested in diving deeper into this application, you can check my full implementation of Flow-TTS. Check it out here: [Full Flow-TTS Implementation](https://github.com/RKorzeniowski/FlowTTS/tree/main/Flow-TTS).

---

## 5. Conclusion

Normalizing flows offer a compelling alternative to traditional generative models by combining mathematical elegance with practical benefits. Their ability to provide exact likelihood estimates, invertible mappings, and flexible density modeling makes them a powerful tool for modern deep learning tasks. While VAEs and GANs have their own strengths, the statistical robustness and direct training methodology of normalizing flows continue to drive innovation, especially in challenging tasks such as high-fidelity audio synthesis with Flow-TTS.

This exploration reveals not only the theoretical foundations but also the practical considerations when choosing among different generative models. As research progresses, we can expect further integration of normalizing flows into hybrid models, pushing the boundaries of what we can achieve with generative deep learning.
