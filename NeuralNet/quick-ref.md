Backprop Cheat Note
## 1. Error propagation (recursive)

Equation:

$$
\text{upstreamErr}_l = (\text{upstreamErr}_{l+1} \cdot W_{l+1}^T) \odot F_l'(Z_l)
$$

Steps:

- Pull error from next layer: $\text{upstreamErr}_{l+1}$
- Multiply by weight transpose: $\cdot W_{l+1}^T$
- Element-wise multiply ($\odot$) with activation derivative: $F_l'(Z_l)$

## 2. Gradient w.r.t. weights

Equation:

$$
dW_l = A_{l-1}^T \cdot \text{upstreamErr}_l
$$

Notes:

- Use activations from the previous layer: $A_{l-1}$
- Combine with current layer error: $\text{upstreamErr}_l$
