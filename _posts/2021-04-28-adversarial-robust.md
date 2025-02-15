# Adversarial Attacks and Robust Deep Learning

Adversarial examples are inputs that have been carefully perturbed to cause a machine learning model to make a mistake—even though to human eyes (or ears) the input appears unchanged. In many ways, they are the “optical illusions” of deep learning. In this post, we will explore the mathematical underpinnings of adversarial attacks, including popular methods like the Fast Gradient Sign Method (FGSM) and Projected Gradient Descent (PGD), and then discuss strategies such as adversarial training and defensive distillation to improve model robustness.

---

## 1. Introduction

Machine learning models, especially deep neural networks, achieve high accuracy on standard test sets—but only on a small subset of all possible inputs. In real-world settings, models must handle every possible input, including those adversaries design specifically to fool them. In this post, we dive into:

- The precise definition of adversarial examples and adversarial risk.
- Mathematical derivations of FGSM and PGD attacks.
- Strategies to train robust models (e.g., adversarial training and defensive distillation).
- Intuitions behind why adversarial examples exist and how robust representations differ from standard ones.

---

## 2. Understanding Adversarial Examples

Adversarial examples can be defined as follows:

> **Definition:** An adversarial example is an input $x_{\text{adv}}$ crafted as
>
> $$
> x_{\text{adv}} = x + \delta,
> $$
>
> where $\delta$ is a small perturbation (often measured in some norm) chosen so that the model’s prediction $h_\theta(x_{\text{adv}})$ is incorrect even though $x$ and $x_{\text{adv}}$ are nearly indistinguishable.

A key concept is the **adversarial risk**:
 
$$
R_{\mathrm{adv}}(h_\theta, \mathcal{D}) = E_{(x,y)\sim \mathcal{D}} \left[ \max_{\delta \in \Delta(x)} \ell\big(h_\theta(x+\delta), y\big) \right],
$$

which contrasts with the standard risk that averages the loss over the data distribution. Instead, adversarial risk considers the worst-case loss within a neighborhood $\Delta(x)$ (often defined by an $L_p$ norm ball around $x$).

*This formulation highlights the challenge: our models must perform well even on inputs chosen by an attacker.*

---

## 3. Mathematical Formulation of Adversarial Attacks

### 3.1. Fast Gradient Sign Method (FGSM)

One of the earliest and most intuitive attacks is the FGSM, introduced by Goodfellow et al. Its key idea is to perturb the input in the direction of the gradient of the loss with respect to the input.

The adversarial example is computed as:

$$
x_{\text{adv}} = x + \epsilon \cdot sgn\big(\nabla_x \ell(\theta, x, y)\big),
$$

where:
- $\epsilon$ controls the magnitude of the perturbation,
- $\ell(\theta, x, y)$ is the loss function (e.g., cross-entropy),
- $sgn(\cdot)$ is applied elementwise.

### 3.2. Projected Gradient Descent (PGD)

PGD can be seen as a multi-step, iterative variant of FGSM. At each iteration, the adversary takes a step similar to FGSM and then projects the perturbed input back onto the allowed set (often an \( L_\infty \) ball of radius \( \epsilon \) around the original input):

$$
x^{t+1} = \Pi_{x+\mathcal{S}}\Big(x^{t} + \alpha \cdot sgn\big(\nabla_x \ell(\theta, x^t, y)\big)\Big),
$$

where:
- $\alpha$ is the step size,
- $\Pi_{x+\mathcal{S}}(\cdot)$ denotes the projection onto the set $\{x' : \|x' - x\|_p \le \epsilon\}$.

*PGD is particularly important because robustness to PGD attacks has been shown to transfer to robustness against a wide range of first-order attacks.*

---

## 4. Pseudocode for FGSM and PGD

Below are simple pseudocode snippets for FGSM and PGD.

### FGSM Pseudocode

```python
def FGSM(x, y, model, epsilon):
    # Compute the gradient of loss w.r.t. input
    gradient = compute_gradient(model, x, y)
    # Create adversarial perturbation
    perturbation = epsilon * sign(gradient)
    # Generate adversarial example
    x_adv = x + perturbation
    return clip(x_adv, min_value, max_value)
```

### PGD Pseudocode

```python
def PGD(x, y, model, epsilon, alpha, num_steps):
    x_adv = x.copy()
    for _ in range(num_steps):
        # Compute gradient at current adversarial example
        grad = compute_gradient(model, x_adv, y)
        # Take a small step in the direction of the gradient sign
        x_adv = x_adv + alpha * sign(grad)
        # Project back onto the epsilon-ball around x
        x_adv = project(x_adv, x, epsilon)
        # Ensure pixel values remain valid
        x_adv = clip(x_adv, min_value, max_value)
    return x_adv
```

*Here, `clip` ensures the adversarial example remains within valid bounds (e.g., pixel values between 0 and 1), and `project` enforces the norm constraint.*

---

## 5. Building Robust Models

### 5.1. Adversarial Training

One of the most effective defenses against adversarial attacks is **adversarial training**. Instead of optimizing the standard risk, we train the model to minimize the worst-case loss over an allowed set of perturbations:

$$
\min_\theta \; \frac{1}{|D_{\text{train}}|} \sum_{(x,y) \in D_{\text{train}}} \max_{\delta \in \Delta(x)} \ell\big(h_\theta(x+\delta), y\big).
$$

By incorporating adversarial examples during training, the model learns to be invariant to small, worst-case perturbations. However, this often comes with a trade-off: increasing robustness can sometimes reduce standard accuracy.

### 5.2. Defensive Distillation

Another approach is **defensive distillation**. In this method, a model is first trained to output class probabilities (soft labels) rather than hard decisions. A second model is then trained on these soft labels, resulting in a smoother decision boundary. This smoothness can make it harder for an adversary to find small perturbations that cause misclassification.

---

## 6. Robust Representations

Robust models not only improve defense against adversarial attacks but also tend to learn representations that align better with human concepts. In robust representation learning, we desire that if two inputs $x$ and $x'$ are close in the input space (e.g., ${\lVert}x - x'{\rVert}_2 \le \epsilon$ ), then their internal representations should also be close:

$$
{\lVert} f(x) - f(x')\rVert \le C \cdot \epsilon,
$$

where $f(x)$ is a representation (or feature) of $x$ extracted by the network, and $C$ is a constant. This not only leads to more interpretable features but also forces the network to rely on high-level semantic cues rather than “non-robust” features that can be easily exploited by adversaries.

---

## 7. Mathematical Intuitions Behind Adversarial Vulnerability

Why do adversarial examples exist? One key observation is that many deep neural networks are highly linear in high-dimensional spaces. Even though networks contain non-linearities (like ReLU), the linear behavior dominates when small perturbations are applied. In high-dimensional settings, small changes along many dimensions can add up to a significant change in the model’s output—even if each change is imperceptible. This is why adversarial examples can be crafted with tiny $\epsilon$ values yet cause dramatic misclassifications.

Furthermore, because the training data only covers a tiny fraction of the input space, there are many “blind spots” where the model’s behavior is poorly constrained. Adversaries exploit these gaps by finding perturbations that push an input across the decision boundary.

---

## 8. Trade-offs and Practical Considerations

While defenses like adversarial training improve robustness, they come with costs:

- **Accuracy vs. Robustness:** Restricting a model to ignore “non-robust” features can lower overall accuracy on natural examples.
- **Model Capacity:** More robust training often requires larger models or longer training times.
- **Transferability:** Even robust models might still be vulnerable to certain attacks, especially if the adversary can adapt to the defense.

A holistic approach to robustness considers not just adversarial training but also regularization, architectural improvements, and even post-training methods to repair vulnerabilities.

---

## 9. Conclusion

In this post we have:

- Defined adversarial examples and adversarial risk.
- Explored the mathematics behind FGSM and PGD attacks, including their iterative formulations and pseudocode.
- Discussed defenses such as adversarial training and defensive distillation, and the importance of learning robust representations.
- Delved into intuitive explanations of why adversarial vulnerabilities occur in high-dimensional spaces.

As the field evolves, robust deep learning remains an active area of research with many open questions—especially regarding the trade-offs between accuracy, computational cost, and security. Future research may yield new training algorithms or architectures that can both achieve high accuracy and withstand adversarial manipulations without sacrificing performance.

By understanding these mathematical foundations and the corresponding defense strategies, researchers and practitioners can better design systems that are not only accurate on standard benchmarks but also resilient in the face of adversarial challenges.
