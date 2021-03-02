class ResidualDynamics(Callback):
    order, run_valid = 65, True

    def __init__(self, figsize = (6, 6), alpha = .3, size = 30, color = 'aqua', cmap = 'gist_rainbow'):
        store_attr("figsize, alpha, size, color, cmap")

    def before_fit(self):
        """Called before doing anything, ideal for initial setup."""
        self.run = not hasattr(self.learn, 'lr_finder') and not hasattr(self, "gather_preds")
        if not self.run:
            return

        # Prepare ground truth container, set here as y_true's always stay the same
        self.y_true = []

    def before_epoch(self):
        """Called at the beginning of each epoch, useful for any behavior you need to reset at each epoch."""

        # Prepare empty pred container in every epoch, set/reset here as new preds each epoch as NN learns
        self.y_pred = []

    def after_pred(self):
        """Called after computing the output of the model on the batch. It can be used to change that output before it's fed to the loss."""
        # If training then skip
        if self.training:
            return

        # Get y_true in epoch 0
        if self.epoch == 0:
            self.y_true.extend(self.y.cpu().flatten().numpy())

        # Gather last prediction for every batch
        y_pred = self.pred.detach().cpu()
        
        self.y_pred.extend(y_pred.flatten().numpy())

    def after_epoch(self):
        """Called at the end of an epoch, for any clean-up before the next one."""

        self.y_true = np.array(self.y_true)                                     # Ground truths
        self.y_pred = np.array(self.y_pred)                                     # Predictions for this epoch
        self.residuals = self.y_true - self.y_pred                              # Residuals for this epoch
        self.x_bounds = (np.min(self.y_true), np.max(self.y_true))              # x bounds for the graph (min and max of ground truths)
        self.y_bounds = (-np.max(self.residuals), np.max(self.residuals))       # y bounds for the graph (max residual either side of 0)
        
        # Update the graph with this epoch's "data"
        self.update_graph(residuals = self.residuals, 
                          y_true = self.y_true, 
                          x_bounds = self.x_bounds, 
                          y_bounds = self.y_bounds)

    def after_fit(self):
        """Called at the end of training, for final clean-up."""
        plt.close(self.graph_ax.figure)

    def update_graph(self, residuals, y_true, x_bounds = None, y_bounds = None):
        """Called at the end of an epochs ones ground truths, preds, residuals and x and y bounds are calculated (see above)"""

        # If no graph in place then great dataframe to output and graph related objects
        if not hasattr(self, 'graph_fig'):
            self.df_out = display("", display_id = True)
            self.graph_fig, self.graph_ax = plt.subplots(1, figsize = self.figsize)
            self.graph_out = display("", display_id = True)
        
        # Clear any graphs
        self.graph_ax.clear()

        # Plotting the residuals and formatting
        self.graph_ax.scatter(x = y_true, y = residuals, color = self.color, edgecolor = 'black', alpha = self.alpha, s = self.size)
        self.graph_ax.set_xlim(*x_bounds)
        self.graph_ax.set_ylim(*y_bounds)
        self.graph_ax.plot([*x_bounds], [0, 0], color = 'gainsboro')
        self.graph_ax.set_xlabel('y_true', fontsize = 12)
        self.graph_ax.set_ylabel('residuals', fontsize = 12)
        self.graph_ax.grid(color = 'gainsboro', linewidth = .2)
        self.graph_ax.set_title(f'Residuals \nepoch: {self.epoch +1}/{self.n_epoch}')

        # Output metrics
        self.df_out.update(pd.DataFrame(np.stack(self.learn.recorder.values)[-1].reshape(1,-1),
                                        columns=self.learn.recorder.metric_names[1:-1], index=[self.epoch]))
        self.graph_out.update(self.graph_ax.figure)
