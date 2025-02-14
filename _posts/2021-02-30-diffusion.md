# Diffusion Models: Score Matching Perspective

Diffusion models have emerged as a powerful paradigm in generative modeling. By gradually corrupting data with noise and then reversing this process, these models can synthesize new, high-quality samples. In this post, we present a comprehensive look at diffusion models—covering the mathematical underpinnings of the forward diffusion process, an introduction to score matching (a technique that sidesteps normalization issues), and a detailed derivation of the reverse (generative) process starting from score matching theory.

---

## 1. Introduction to Diffusion Models

At a high level, a diffusion model transforms data into a simple noise distribution via a forward process and then learns to reverse this transformation. The forward process is typically designed so that an initial sample $x_0$ drawn from the true data distribution $p_0(x)$ is gradually “noised up” until it resembles a standard Gaussian. Once trained, the model generates new samples by inverting this process—starting from noise and gradually removing it until a sample from $p_0(x)$ emerges.

---

## 2. Mathematical Foundations

### 2.1. The Forward Diffusion Process

The forward process is defined via a stochastic differential equation (SDE):

$$
dx = f(x,t)\,dt + g(t)\,dw,
$$

with initial condition $x(0) \sim p_0(x)$. Here, $w$ denotes a standard Wiener process. A common choice in diffusion models is to set:

$$
f(x,t) = -\frac{1}{2}\beta(t)x \quad \text{and} \quad g(t) = \sqrt{\beta(t)},
$$

where $\beta(t)\) is a time-dependent variance schedule. Under this process, the probability density $p_t(x)$ evolves according to the Fokker–Planck equation:

$$
\partial_t p_t(x) = - \nabla \cdot \big(f(x,t)p_t(x)\big) + \frac{1}{2}g^2(t)\Delta p_t(x).
$$

Over time, as $t$ increases, the forward process gradually destroys the information contained in $x_0$, ultimately driving $p_t(x)$ to a simple Gaussian distribution.

### 2.2. The Reverse Diffusion Process

To generate samples, we need to “reverse” the forward diffusion. The theory of time reversal for SDEs tells us that the reverse-time process is given by:

$$
dx = \Big[f(x,t) - g^2(t)\, \nabla_x \ln p_t(x)\Big]\,dt + g(t)\,d\bar{w},
$$

where $d\bar{w}$ represents a reverse-time Wiener process. The additional term,

$$
-g^2(t)\, \nabla_x \ln p_t(x),
$$

acts as a corrective drift that “steers” the process back from noise toward regions of high probability in the data space. For our specific choices, the reverse SDE becomes

$$
dx = \Big[-\frac{1}{2}\beta(t)x - \beta(t)\, \nabla_x \ln p_t(x)\Big]\,dt + \sqrt{\beta(t)}\,d\bar{w}.
$$

---

3. Pseudocode for Diffusion Models

Below is a simplified pseudocode that outlines the training and sampling procedures for a diffusion model.
3.1. Training Procedure

```python
# Training loop for diffusion model
for epoch in range(num_epochs):
    for x0 in data_loader:
        # Sample a random timestep t
        t = random.randint(1, T)
        
        # Sample noise from standard Gaussian
        epsilon = sample_normal(shape=x0.shape)
        
        # Compute alpha_bar for timestep t
        alpha_bar_t = compute_alpha_bar(t)  # e.g., product of (1-beta) terms
        
        # Generate noisy data x_t
        x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * epsilon
        
        # Predict noise using the model
        predicted_epsilon = model(x_t, t)
        
        # Compute mean squared error loss between actual and predicted noise
        loss = mse_loss(epsilon, predicted_epsilon)
        
        # Backpropagation and optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

3.2. Sampling Procedure

```python
# Sampling from the diffusion model
# Start with pure Gaussian noise
x_T = sample_normal(shape=data_shape)

for t in reversed(range(1, T+1)):
    # Predict noise at current timestep
    predicted_epsilon = model(x_T, t)
    
    # Compute the reverse process mean (mu) based on predicted noise
    mu = compute_reverse_mean(x_T, t, predicted_epsilon)
    
    # Optionally, compute variance sigma for the reverse process
    sigma = compute_reverse_variance(t)
    
    # Sample the previous state x_{t-1}
    x_T = mu + sigma * sample_normal(shape=x_T.shape)

# Final generated sample
x0_generated = x_T
```

This pseudocode outlines the two critical stages of a diffusion model: the training phase where the model learns to predict noise, and the sampling phase where the learned reverse process is used to generate new data samples.


---

## 4. A Primer on Score Matching

Directly computing $\nabla_x \ln p_t(x)$ is typically intractable because it requires knowledge of the (often unnormalized) density $p_t(x)$. **Score matching** provides an elegant solution by focusing on learning the score function:

$$
s(x) \equiv \nabla_x \ln p(x).
$$

Rather than estimating $p(x)$ directly, we train a neural network $s_\theta(x)$ to approximate $\nabla_x \ln p(x)$ by minimizing the objective:

$$
L_{\mathrm{SM}}(\theta) = \frac{1}{2} \, E_{p(x)} \Big[ \| s_\theta(x) - \nabla_x \ln p(x) \|^2 \Big].
$$

Because the true score $\nabla_x \ln p(x)$ is unknown, integration by parts can be used to recast this objective in a form that avoids the explicit computation of the density’s normalizing constant (see citeturn0search3).

In practice, a variant called **denoising score matching (DSM)** is often used. Here, data is perturbed with Gaussian noise to yield noisy samples $x_t$, and a time-dependent score network $s_\theta(x,t)$ is trained so that:

$$
s_\theta(x,t) \approx \nabla_x \ln p_t(x).
$$

This time-dependent score plays a central role in defining the reverse diffusion process.

---

## 5. Deriving the Reverse Diffusion Process via Score Matching

Once we have trained $s_\theta(x,t)$ using (denoising) score matching, we can substitute it for the true score in the reverse SDE. Recall that the reverse SDE is given by:

$$
dx = \Big[f(x,t) - g^2(t)\, \nabla_x \ln p_t(x)\Big]\,dt + g(t)\,d\bar{w}.
$$

Replacing $\nabla_x \ln p_t(x)$ with our trained approximation $s_\theta(x,t)$ yields:

$$
dx = \Big[f(x,t) - g^2(t)\, s_\theta(x,t)\Big]\,dt + g(t)\,d\bar{w}.
$$

For example, with the choices

$$
f(x,t) = -\frac{1}{2}\beta(t)x \quad \text{and} \quad g(t) = \sqrt{\beta(t)},
$$

the reverse SDE becomes:

$$
dx = \Big[-\frac{1}{2}\beta(t)x - \beta(t)\, s_\theta(x,t)\Big]\,dt + \sqrt{\beta(t)}\,d\bar{w}.
$$

This equation provides a principled way to sample from the data distribution $p_0(x)$. Starting from $x_T$ drawn from a Gaussian distribution (which approximates $p_T(x)$ ), one can simulate the reverse-time SDE to gradually “denoise” the sample. The learned score function $s_\theta(x,t)$ guides this reverse process, ensuring that the sample evolves toward regions where the original data $p_0(x)$ is concentrated.

---

## 6. Discussion

The beauty of diffusion models lies in their two-phase process:
- **Forward Diffusion:** Data is progressively perturbed until it becomes nearly indistinguishable from pure noise.
- **Reverse Diffusion:** The model leverages a learned score function to carefully reverse the perturbation, transforming noise back into data.

Score matching is the linchpin in this framework. By learning the gradient of the log density without ever computing the full density (and its problematic normalizing constant), score matching enables a flexible and tractable way to implement the reverse diffusion process.

This derivation not only explains why diffusion models can generate high-quality samples but also unifies concepts from stochastic differential equations, statistical physics (through Langevin dynamics), and modern deep learning. The reverse SDE, with its corrective drift term provided by the score function, encapsulates the essential mechanism that turns a random noise sample into a realistic data point.

---

## 7. Conclusion

In this post, we have combined two key aspects of diffusion models into one comprehensive narrative. We started with the mathematical foundations of the forward diffusion process and introduced score matching—a method for learning the gradient of the log-density without direct likelihood estimation. Building on this foundation, we derived the reverse-time SDE, which uses the learned score function to guide the denoising process. Together, these ideas form the backbone of diffusion models, explaining their ability to generate high-fidelity samples by carefully reversing a noise-corrupting process.

By understanding both the forward dynamics and the crucial role of score matching in deriving the reverse process, one gains deeper insight into why diffusion models have become such a compelling tool in modern generative modeling.
