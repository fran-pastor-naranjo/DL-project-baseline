import matplotlib.pyplot as plt
from typing import Tuple, List
import numpy as np

def plot_graphs(data: List[Tuple[List[float], List[float]]], labels: List[str]=None, filename: str='./Graphic.png') -> None:
    """
    Plots multiple graphs on the same figure.

    Args:
        data: Arbitrary number of (x, y) tuples. If x is None, it will default to an index range of y.
        labels (List[str], optional): Labels for each plotted line. Must match the number of (x, y) pairs.
        filename (str, optional): Path to save the plot image. Defaults to './Graphic.png'.
    """
    print('\n[PLOT INFO]: Representando gráfica ...')

    plt.style.use("ggplot")
    plt.figure()

    for idx, (x, y) in enumerate(data):
        if y is not None:
            if x is None:
                x = np.arange(len(y))  # Default x to index values if not provided
            label = labels[idx] if labels and idx < len(labels) else f'Plot {idx + 1}'
            plt.plot(x, y, label=label)

    plt.title("Graphic")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    
    print('[PLOT INFO]: Guardando gráfica')
    plt.savefig(filename)
    plt.close()