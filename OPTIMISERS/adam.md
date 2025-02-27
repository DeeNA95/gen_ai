Adam keeps track of both the first moment (mean) and second moment (uncentered variance) of the gradients for each parameter. In simpler terms, for each parameter $\theta_j$, Adam maintains:

	•	$m_{t,j}$, an exponentially decaying average of past gradients (similar to momentum),
	•	$v_{t,j}$, an exponentially decaying average of past squared gradients (similar to the RMSprop cache).

The update rules for Adam at iteration $t$ are:

$$
m_t = \beta_1, m_{t-1} + (1-\beta_1), g_t,,
$$

$$
v_t = \beta_2, v_{t-1} + (1-\beta_2), g_t^2,,
$$

$$
\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \qquad
\hat{v}_t = \frac{v_t}{1-\beta_2^t},,
$$

$$
\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t,,
$$

where $g_t$ is the gradient at time $t$, $\beta_1$ is the decay rate for the first moment (momentum term), $\beta_2$ is the decay for the second moment (RMS term), and $\epsilon$ is a small constant to avoid division by zero. The $\hat{m}_t$ and $\hat{v}_t$ are bias-corrected estimates of the first and second moments (this correction compensates for the fact that $m_0$ and $v_0$ start at 0, which would otherwise bias the averages in early iterations) ￼. In implementation, often the bias correction is done as above for completeness, but conceptually it’s the $m_t$ and $v_t$ that matter for understanding.
