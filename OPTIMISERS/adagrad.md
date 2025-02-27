Adagrad is an adaptive learning rate method that adjusts the learning rate for each model parameter individually, based on how large and how frequent the gradients of that parameter are. The name comes from Adaptive Gradient. The core idea is to give frequently updated parameters smaller learning rates and rarely updated parameters larger learning rates over time ￼. This is especially useful in situations like natural language processing or recommendation systems, where some features are very sparse (infrequent) and others are dense – Adagrad will automatically slow down learning for the dense, frequently occurring features and speed up learning for the sparse ones.

Adagrad achieves this by keeping track of the sum of squares of gradients for each parameter. For each parameter $\theta_j$ (the $j$-th component of $\theta$), we accumulate:

$$
G_{t,j} = G_{t-1,j} + \big( g_{t,j} \big)^2
$$

where $g_{t,j}$ is the gradient of the loss w.r.t. $\theta_j$ at time $t$ (and $G_{0,j}$ is initialized to 0). Then Adagrad updates the parameter as:

$$
\theta_{t,j} = \theta_{t-1,j} - \frac{\eta}{\sqrt{G_{t,j}} + \epsilon} g_{t,j}
$$

where $\eta$ is a global base learning rate and $\epsilon$ is a small constant (like $10^{-8}$) to prevent division by zero ￼ ￼. Notice that as $G_{t,j}$ (the accumulated squared gradient) grows, the effective learning rate $\frac{\eta}{\sqrt{G_{t,j}}}$ for that parameter shrinks.
