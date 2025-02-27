Update Rule:
$$
\theta_{t+1} = \theta_t - \eta \nabla_{!\theta}L(\theta_t; x_i, y_i)
$$
Where:
$
\nabla_{!\theta}L(\theta_t; x_i, y_i)
$ is the gradient of the loss $L$ evaluated at the current parameter values $\theta_t$ on a training example $(x_i, y_i)$, and $\eta$ is the learning rate.

This formula means we adjust each parameter $\theta$ in the direction of negative gradient, i.e. subtract the derivative of the loss with respect to that parameter.
