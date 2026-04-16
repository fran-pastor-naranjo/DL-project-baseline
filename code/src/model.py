import torch
from typing import Dict, Any

class MLP(torch.nn.Module):
    """
    A Multi-Layer Perceptron (MLP) model.
    """
    def __init__(self, params: Dict[str, Any], input_shape: int) -> None:
        """
        Initializes the MLP model.

        Args:
            params (Dict[str, Any]): Dictionary containing model parameters:
                - **units (List[int])**: List of integers specifying the number of units in each layer.
                - **n_outputs (int)**: Number of output units.
                - **drop_coef (float)**: Dropout coefficient.
                - **activation (str)**: Activation function to use.
                - **classify (bool)**: Indicate if is a classification task. If True, it adds a softmax layer
            input_shape (int): An integer representing the shape of the input data.
        """

        super(MLP, self).__init__()
        units = params['units'].copy()
        n_outputs = params['n_outputs']
        drop_coef = params['drop_coef']

        if params.get('activation') == 'relu':
            activation = torch.nn.ReLU()
        elif params.get('activation') == 'tanh':
            activation = torch.nn.Tanh()
        elif params.get('activation') == 'sigmoid':
            activation = torch.nn.Sigmoid()
        elif params.get('activation') == 'leaky_relu':
            activation = torch.nn.LeakyReLU()
        elif params.get('activation') == 'elu':
            activation = torch.nn.ELU()
        elif params.get('activation') == 'selu':
            activation = torch.nn.SELU()
        elif params.get('activation') == 'gelu':
            activation = torch.nn.GELU()
        else:
            raise ValueError("The specified activation function is not implemented")

        classifier = []
        units.insert(0, input_shape)

        for i in range(len(units) - 1):
            classifier.append(torch.nn.Linear(units[i], units[i + 1]))
            classifier.append(activation)
            if drop_coef > 0:
                classifier.append(torch.nn.Dropout(drop_coef))

        classifier.append(torch.nn.Linear(units[-1], n_outputs))
        if params['classify']:
            classifier.append(torch.nn.Softmax(dim=-1))

        self.classifier = torch.nn.Sequential(*classifier)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the MLP model.

        Args:
            x (torch.Tensor): A 2D tensor of shape (batch_size, L) that represents the input tensor, where:

                - **batch_size**: Number of samples in the batch.
                - **L**: Length of each 1D sample.

        Returns:
            x (torch.Tensor): A 2D tensor of shape (batch_size, n_outputs) that represents the output tensor, where:

                - **n_outputs**: Number of output units.
        """
        x = self.classifier(x)
        return x