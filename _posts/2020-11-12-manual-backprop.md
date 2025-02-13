# **Backpropagation on a Paper Sheet**

## Table of Contents
1. [What is Backpropagation?](#backprop)
2. [Mathematical Foundation of Backpropagation](#backprop_math)
3. [Step-by-Step Manual Gradient Calculation](#gradient)
4. [Pseudocode Implementation of Backpropagation](#impl)
5. [Extending to Multi-Layer Networks](#mlp)
6. [Backpropagation in Convolutional Neural Networks](#conv)
7. [Backpropagation Through Skip Connections](#skip_con)
8. [Backpropagation Through Dropout](#dropout)
9. [Backpropagation Through Batch Normalization](#batch_norm)
10. [Under the Hood: Symbolic Differentiation in PyTorch](#symbolic)

Backpropagation is the backbone of modern deep learning. While frameworks like TensorFlow and PyTorch handle it automatically, understanding how gradients propagate manually is essential for building an intuition for deep learning. In this blog, we'll break down gradient backpropagation with math and implement it using pseudocode.  

---

## **1. What is Backpropagation?**<a name="backprop"></a>

Backpropagation is an optimization algorithm used to train neural networks. It efficiently computes gradients using the **chain rule** from calculus, adjusting weights to minimize loss.  

### **Why Manually Compute Gradients?**  
- Deepen understanding of neural network internals.  
- Debug custom implementations.  
- Optimize performance in specialized applications.  

---

## **2. Mathematical Foundation of Backpropagation**<a name="backprop_math"></a>

We consider a simple **single-layer** neural network with one hidden unit:  

$$ y = f(w \cdot x + b) $$

where:  
-  $x$ is the input,  
-  $w$ is the weight,  
-  $b$ is the bias,  
-  $f$ is an activation function,  
-  $y$ is the output.  

For a given loss function $L(y, \hat{y})$, backpropagation computes the gradients:  

$$\frac{\partial L}{\partial w}, \quad \frac{\partial L}{\partial b}$$

using the **chain rule**:  

$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial w}$$

$$\frac{\partial L}{\partial b} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial b}$$

---

## **3. Step-by-Step Manual Gradient Calculation**<a name="gradient"></a>

### **Step 1: Forward Pass**  

Consider a **single-layer** neural network with **sigmoid activation**:  

$$y = \sigma(z), \quad z = w \cdot x + b$$

where:  

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Given a **loss function** (e.g., Mean Squared Error for simplicity):  

$$L = \frac{1}{2} (y - \hat{y})^2$$

---

### **Step 2: Compute Gradients Using the Chain Rule**  

We need:  
1. **Derivative of loss** w.r.t. output**:  

   
   $$\frac{\partial L}{\partial y} = (y - \hat{y})$$

2. **Derivative of activation (sigmoid function)**:  

   $$\frac{\partial y}{\partial z} = \sigma(z) (1 - \sigma(z))$$

3. **Derivatives w.r.t. parameters**:  

   $$\frac{\partial z}{\partial w} = x, \quad \frac{\partial z}{\partial b} = 1$$

Using the **chain rule**:  


$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial z} \cdot \frac{\partial z}{\partial w}$$

$$\frac{\partial L}{\partial b} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial z} \cdot \frac{\partial z}{\partial b}$$

Expanding:  


$$\frac{\partial L}{\partial w} = (y - \hat{y}) \cdot \sigma(z) (1 - \sigma(z)) \cdot x$$


$$\frac{\partial L}{\partial b} = (y - \hat{y}) \cdot \sigma(z) (1 - \sigma(z))$$

---

## **4. Pseudocode Implementation of Backpropagation**<a name="impl"></a>

Here’s how to compute gradients manually in Python-like pseudocode:  

```python
# Sigmoid activation function
def sigmoid(z):
    return 1 / (1 + exp(-z))

# Forward pass
def forward(x, w, b):
    z = w * x + b
    y = sigmoid(z)
    return y, z

# Compute gradients
def backward(x, y, z, y_true):
    dL_dy = y - y_true  # Loss derivative
    dy_dz = y * (1 - y)  # Sigmoid derivative
    dz_dw = x            # Weight derivative
    dz_db = 1            # Bias derivative
    
    # Chain rule
    dL_dw = dL_dy * dy_dz * dz_dw
    dL_db = dL_dy * dy_dz * dz_db
    
    return dL_dw, dL_db

# Gradient Descent Step
def update_params(w, b, dL_dw, dL_db, lr=0.01):
    w -= lr * dL_dw
    b -= lr * dL_db
    return w, b

# Example usage
x = 0.5       # Input
y_true = 1.0  # Target output
w, b = 0.2, 0.1  # Initial parameters

# Training Step
y, z = forward(x, w, b)
dL_dw, dL_db = backward(x, y, z, y_true)
w, b = update_params(w, b, dL_dw, dL_db)

print(f"Updated w: {w}, Updated b: {b}")
```

---

## **5. Extending to Multi-Layer Networks**<a name="mlp"></a>

For deep neural networks, backpropagation extends recursively:  

1. Compute gradients layer by layer.  
2. Store intermediate activations.  
3. Propagate errors backward.  

For each layer $l$, gradients propagate as:  


$$\delta^l = (\delta^{l+1} W^{l+1}) \odot f'(z^l)$$


where $\delta^l$ is the error term for layer $l$, and $\odot$ represents element-wise multiplication.  

---

## **6. Backpropagation in Convolutional Neural Networks**<a name="conv"></a>

In Convolutional Neural Networks (CNNs), the forward pass involves applying filters (kernels) to input feature maps. The weights of these filters must be updated during backpropagation.  

### **6.1 Forward Pass in a Convolution Layer**  
A convolution operation applies a filter $W$ to an input feature map $X$, producing an output feature map $Y$:  

$$
Y_{ij} = \sum_m \sum_n W_{mn} X_{(i+m)(j+n)}
$$

where $m$, $n$ are the kernel dimensions.  

### **6.2 Gradient Computation in Backpropagation**  
During backpropagation, we compute gradients for:  
- The **filter weights** $W$  
- The **input feature map** $X$  

Using the **chain rule**, we compute:  

$$
\frac{\partial L}{\partial W_{mn}} = \sum_{i,j} \frac{\partial L}{\partial Y_{ij}} X_{(i+m)(j+n)}
$$

$$
\frac{\partial L}{\partial X_{ij}} = \sum_{m,n} \frac{\partial L}{\partial Y_{(i-m)(j-n)}} W_{mn}
$$

This involves **convolution operations in reverse**:  
- **Gradients w.r.t. filters** are computed using **convolutions with the input**.  
- **Gradients w.r.t. inputs** are computed using **convolutions with the flipped filter**.  

### **6.3 Pseudocode for Backpropagation in a Convolutional Layer**
```python
def backward_conv(dL_dY, X, W, stride=1, padding=0):
    dL_dW = convolve(X, dL_dY, mode="valid")  # Compute gradient w.r.t. filters
    dL_dX = convolve(dL_dY, flip(W), mode="full")  # Compute gradient w.r.t. input
    return dL_dW, dL_dX
```

---

## **7. Backpropagation Through Skip Connections**<a name="skip_con"></a>

### **7.1 What Are Skip Connections?**
Skip connections allow gradients to bypass certain layers, preventing vanishing gradients and improving deep network training. The output of a residual block is:  

$$
Y = f(X) + X
$$

where $f(X)$ is a function of $X$ (e.g., a convolutional layer).  

### **7.2 How Gradients Flow Through Skip Connections**  
Using the chain rule:  

$$
\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y} \cdot \left( \frac{\partial Y}{\partial X} \right)
$$

Since $Y = f(X) + X$, we differentiate:

$$
\frac{\partial Y}{\partial X} = \frac{\partial f(X)}{\partial X} + I
$$

where $I$ is the identity matrix. This allows gradients to propagate efficiently, avoiding the vanishing gradient problem.  

### **7.3 Pseudocode for Skip Connection Backpropagation**
```python
def backward_skip(dL_dY, dL_dX_residual):
    dL_dX = dL_dY + dL_dX_residual  # Gradient flows directly through skip connection
    return dL_dX
```

---

## **8. Backpropagation Through Dropout**<a name="dropout"></a>

### **8.1 What Is Dropout?**  
Dropout is a regularization technique that randomly sets some activations to zero during training.  

### **8.2 Effect on Backpropagation**  
During the forward pass, neurons are dropped with probability $p$. In backpropagation:  
- Dropped neurons do not receive gradients.  
- Active neurons scale their gradients by $\frac{1}{1 - p}$ to maintain expected value consistency.  

### **8.3 Pseudocode for Backpropagation Through Dropout**
```python
def backward_dropout(dL_dY, mask, p):
    dL_dX = (dL_dY * mask) / (1 - p)  # Scale gradient for active neurons
    return dL_dX
```

---

## **9. Backpropagation Through Batch Normalization**<a name="batch_norm"></a>

### **9.1 What Is Batch Normalization?**  
Batch normalization (BN) normalizes activations to stabilize training. It transforms activations $X$ as:  

$$
\hat{X} = \frac{X - \mu}{\sigma}
$$

$$
Y = \gamma \hat{X} + \beta
$$

where $\gamma$, $\beta$ are learnable parameters, and $\mu$, $\sigma$ are the batch mean and variance.  

### **9.2 Gradients for Batch Normalization Parameters**  
Using the chain rule:  

$$
\frac{\partial L}{\partial \gamma} = \sum \left( \frac{\partial L}{\partial Y} \cdot \hat{X} \right)
$$

$$
\frac{\partial L}{\partial \beta} = \sum \frac{\partial L}{\partial Y}
$$

### **9.3 Pseudocode for Batch Normalization Backpropagation**
```python
def backward_batchnorm(dL_dY, X, mu, sigma, gamma):
    dL_dX_hat = dL_dY * gamma
    dL_dX = (1 / sigma) * (dL_dX_hat - np.mean(dL_dX_hat) - X * np.mean(dL_dX_hat * X))
    dL_dgamma = np.sum(dL_dY * (X - mu) / sigma)
    dL_dbeta = np.sum(dL_dY)
    return dL_dX, dL_dgamma, dL_dbeta
```

---

## **10. Under the Hood: Symbolic Differentiation in PyTorch**<a name="symbolic"></a>

PyTorch's **autograd** system uses symbolic differentiation through a dynamic computation graph (also called a **computational tape**). Instead of explicitly computing derivatives step by step like we did earlier, PyTorch builds a **computational graph** dynamically during the forward pass and then traverses it backward to compute gradients.  

---

### **10.1 Dynamic Computation Graph**  

When performing operations on tensors with `requires_grad=True`, PyTorch records every mathematical operation in a directed acyclic graph (DAG), where:  

- **Nodes** represent tensors and intermediate values.  
- **Edges** represent operations applied to tensors.  

Each operation stores:  
- The **function used** (e.g., addition, multiplication, activation function).  
- The **input tensors** and their connections.  
- The **gradient function** to compute derivatives efficiently.  

This graph is **dynamically constructed** during the forward pass. Once a loss function is defined, calling `backward()` **traverses the graph in reverse order**, computing gradients using the **chain rule** efficiently.  

---

### **10.2 Reverse-Mode Automatic Differentiation**  

PyTorch uses **reverse-mode differentiation**, also known as **backpropagation**. Instead of computing derivatives from inputs to outputs (like forward-mode differentiation), it computes gradients from outputs back to inputs, making it efficient for neural networks with many parameters.  

For a scalar loss function $L$ with respect to multiple parameters:  

$$
\frac{dL}{dx}, \frac{dL}{dw}, \frac{dL}{db}
$$

PyTorch backpropagates gradients using:  

$$
\frac{dL}{dx} = \frac{dL}{dy} \cdot \frac{dy}{dx}
$$

$$
\frac{dL}{dw} = \frac{dL}{dy} \cdot \frac{dy}{dw}
$$

$$
\frac{dL}{db} = \frac{dL}{dy} \cdot \frac{dy}{db}
$$

This is **efficient** because we only need to traverse the graph **once**, reusing intermediate gradient computations.  

---

### **10.3 How PyTorch Implements Backpropagation**  

Under the hood, PyTorch’s **autograd engine** works in three steps:  

1. **Forward Pass: Build the Graph**  
   - Each tensor operation (addition, multiplication, etc.) is registered in the graph.  
   - Each node keeps track of the function used to create it.  

2. **Backward Pass: Traverse the Graph in Reverse**  
   - PyTorch starts from the **loss function** and computes gradients backward.  
   - Each operation's **gradient function** (stored from the forward pass) is applied recursively.  
   - Results are stored in the `.grad` attribute of each tensor.  

3. **Gradient Accumulation and Optimization**  
   - PyTorch accumulates gradients in `.grad` instead of overwriting them (use `zero_grad()` to reset).  
   - Optimizers like SGD or Adam use these gradients to update weights.  

---

### **10.4 Example: How PyTorch Stores and Computes Gradients**  

Let's say we define:  

$$
y = w \cdot x + b
$$

$$
L = (y - 5)^2
$$

1. **Forward Pass: PyTorch Constructs a Graph**  
   ```
   x ---> (*) ---> (+) ---> (square) ---> L
          |        |
          w        b
   ```
   - Each operation is stored in the graph with its gradient function.  

2. **Backward Pass: PyTorch Computes Gradients**  
   - `dL/dy = 2(y - 5)`
   - `dL/dw = dL/dy * x`
   - `dL/db = dL/dy * 1`  
   
PyTorch automatically tracks these dependencies, eliminating the need for explicit differentiation.  

---

### **10.5 Low-Level Autograd Example: Inspecting PyTorch’s Computational Graph**  

To see how PyTorch builds the computation graph dynamically, we can inspect the `.grad_fn` attribute:  

```python
import torch

x = torch.tensor(2.0, requires_grad=True)
w = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

y = w * x + b
loss = (y - 5) ** 2

print(loss.grad_fn)  # Shows that loss is computed from a square function
print(y.grad_fn)  # Shows y comes from an AddBackward function
```

Output:  
```
<MulBackward0 object at 0x...>
<AddBackward0 object at 0x...>
```

This confirms that PyTorch stores operations dynamically and computes gradients automatically.  

---

### **10.6 Why PyTorch Uses Reverse-Mode Differentiation?**  

**Efficiency:**  
- **Reverse-mode differentiation (backpropagation)** is efficient for functions with many inputs but a single output (e.g., neural network loss functions).  
- **Forward-mode differentiation** (computing derivatives in the forward pass) is better for functions with fewer inputs but many outputs, making it inefficient for deep learning.  

By computing **all gradients in one backward pass**, PyTorch makes training large models computationally feasible.  

In summary PyTorch **does not symbolically differentiate functions algebraically** but instead:  
- **Records** operations dynamically in a **computation graph**.  
- **Uses stored gradient functions** to efficiently compute derivatives in **reverse mode**.  
- **Traverses the graph once** using **autograd**, reusing intermediate computations.  

---

## **Conclusion**  

We've covered advanced backpropagation techniques for:  
- Dense layers
- Convolutional layers  
- Skip connections
- Dropout regularization  
- Batch normalization  

By understanding how gradients propagate in these advanced structures, you can debug and optimize deep learning models more effectively.

If you’re feeling ambitious, try implementing a ResNet with manual backpropagation!
