Mathematically, momentum introduces a new variable (velocity) $v_t$ that carries information from past gradients. The update with momentum is typically written as:

$$
v_t = \gamma v_{t-1} - \eta \nabla_{!\theta}L(\theta_{t-1})
$$

$$
\theta_t = \theta_{t-1} + v_t
$$

where $0 < \gamma < 1$ is the momentum coefficient (e.g. $\gamma=0.9$), and $\eta$ is the learning rate as before. Here $v_t$ (the velocity at step $t$) is a running aggregate of past gradients: it is set to the previous velocity $\gamma v_{t-1}$ (thus retaining momentum from past steps) plus the current negative gradient step $-\eta \nabla L(\theta_{t-1})$. The parameter $\theta$ is then updated by adding this velocity (note that $v_t$ already has the negative sign for descent, so we add it). Effectively, the update direction combines the current gradient with the past direction.
