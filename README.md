# Automated Inversion Time selection for Late Gadolinium Enhanced Cardiac MRI

This repository provides tools and a pre-trained model to determine the optimal time inversion (TI) in LGE cardiac MRI imaging. The methodology uses CNN-LSTM architectures to infer the optimal TI based on the input data (Scout Imaging Series).

## Repository Contents:

- `models.py`: Contains the model architecture for the CNN-LSTM-based TI prediction.
- `utils.py`: Utility functions for the repository.
- `config.json`: Configuration settings for the models, including architecture parameters settings.
- `demo.ipynb`: A Jupyter notebook demo showcasing how to load and infer using the pre-trained model.

## Quick Start:

### Prerequisites:
Ensure you have the required libraries installed. For this repository, the key libraries are:

- `PyTorch`
- `Monai`
- `NumPy`
- `Timm`

### Downloading Model Weights:

1. Navigate to the **Releases** section of this GitHub repository.
2. Select the desired release.
3. Download the associated `.pth` file which contains the model weights.
4. Save the weights in the appropriate directory. For the demo, ensure the weights are saved in the `./model` directory.

### Using the Demo:

1. Open `demo.ipynb` in Jupyter or any compatible notebook environment.
2. Execute the cells to:
    - Import necessary libraries.
    - Define the `load_model` function to load the pre-trained model.
    - Load the model and its configuration using `load_model('./model')`.
    - To infer using the model, use `model.infer(x, standardise=True)`, where `x` is your input data.

## Contributing:

Feel free to contribute to this repository by creating pull requests or opening issues to suggest improvements or report bugs.

## License:

MIT License

## Contact:

For any questions or suggestions, please contact [Stefan.KACHEL@austin.org.au].
