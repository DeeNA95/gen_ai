RMSprop (Root Mean Square Propagation) is an adaptive learning rate method that was designed to resolve Adagrad’s main weakness – the relentless accumulation of gradient history. It was proposed by Geoff Hinton in an online course lecture (and the method remains unpublished in formal literature) ￼. RMSprop modifies Adagrad by introducing a forgetting factor for the past squared gradients, so that the algorithm adapts to recent trends in the gradients rather than all historical gradients.

In RMSprop, instead of the sum of squares $G_t$, we maintain an exponentially decaying moving average of squared gradients (let’s call it $E[g^2]_t$). The update rule is:

$$
E[g^2]t = \rho , E[g^2]{t-1} + (1-\rho), g_t^2,
$$

$$
\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{E[g^2]_t} + \epsilon} g_t,,
$$

where $\rho$ is a decay rate (e.g. $\rho = 0.9$) that controls how quickly the history is forgotten ￼. This looks very much like Adagrad except that the “cache” $E[g^2]_t$ (often called the mean squared gradient) is a leaky accumulation – it decays the old squared gradients each step. In code, one might implement: cache = rho * cache + (1 - rho) * (grad**2) and then update parameters with grad / sqrt(cache) as in Adagrad
