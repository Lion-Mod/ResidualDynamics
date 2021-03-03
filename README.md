# ResidualDynamics
Fastai callback to visualise how residuals change after each epoch.

## Requirements
Latest `fastai` version.

## How it works
Before fitting - store all ground truths

After each epoch - store the current predictions (these change after each epoch) and plot a residuals plot

## Visualisation
Blue = residual dynamics (ground truth against residual)
Green = prediction dynamics (ground truth against prediction)
![residualdynamicsexample](https://user-images.githubusercontent.com/70057706/109860509-2a154200-7c56-11eb-9f14-d273e4408ab7.gif)
