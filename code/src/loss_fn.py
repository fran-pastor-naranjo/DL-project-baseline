import torch

class RMSELoss(torch.nn.Module):
    """
    Root Mean Squared Error (RMSE) loss function.
    """
    def __init__(self, eps: float=1e-8) -> None:
        """
        Initializes the RMSE loss function.
        
        Args:
            eps (float, optional): A small value to prevent division by zero. Defaults to 1e-8.
        """
        super().__init__()
        self.mse = torch.nn.MSELoss()
        self.eps = eps

    def forward(self, yhat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Computes the RMSE loss.

        Args:
            yhat (torch.Tensor): A 2D tensor of shape (N, n_outputs) that represents the predicted values, where:

                - **N**: Number of samples in the batch.
                - **n_outputs**: Number of output units.
            y (torch.Tensor): A 2D tensor of shape (N, n_outputs) that represents the actual values.

        Returns:
            loss (torch.Tensor): A 0D tensor that represents the computed RMSE loss.
        """
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss