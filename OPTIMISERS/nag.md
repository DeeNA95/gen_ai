Nesterov Accelerated Gradient (NAG), also known as Nesterov momentum, is a slight modification of the momentum method that often improves its performance. The idea, proposed by Yurii Nesterov, is to look ahead at where the current momentum is about to take the parameters, and compute the gradient at that lookahead position, rather than at the current position ￼. This provides an opportunity to correct the course early if the momentum is going in a suboptimal direction.

In Nesterov momentum, we first make a partial step with the previous momentum before computing the gradient. The update equations can be written as:

$$
v_t = \gamma v_{t-1} - \eta \nabla_{!\theta}L\Big(\theta_{t-1} + \gamma v_{t-1}\Big)
$$

$$
\theta_t = \theta_{t-1} + v_t
$$

with $\gamma$ the momentum coefficient (e.g. 0.9) and $\eta$ the learning rate ￼. Notice that compared to standard momentum, the gradient $\nabla L(\theta_{t-1} + \gamma v_{t-1})$ is evaluated at $\theta_{t-1}$ plus the previous momentum step (i.e. at a point ahead of the current parameters).

```python
def nesterov_momentum(v_prev, theta_prev, gradient, gamma, eta):
    v_t = gamma * v_prev - eta * gradient(theta_prev + gamma * v_prev)
    theta_t = theta_prev + v_t
    return v_t, theta_t
```
