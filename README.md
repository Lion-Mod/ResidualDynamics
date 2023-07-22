# ResidualDynamics
A fastai callback to visualise how residuals change after each epoch and at the end of training. This helps understand if certain examples may have poor prediction performance which may be resolved to improve overall model performance.

## How it works
1. Before fitting - store all ground truths
2. After each epoch - store the current predictions (these change after each epoch) and plot a residuals plot

## Visualisation of it in action
Blue = residual dynamics (ground truth against residual)
Green = prediction dynamics (ground truth against prediction)

![residualdynamicsexample](https://user-images.githubusercontent.com/70057706/109860509-2a154200-7c56-11eb-9f14-d273e4408ab7.gif)
