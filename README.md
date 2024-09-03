# Neural Style Transfer

This repository implements Neural Style Transfer (NST) using PyTorch and a pre-trained VGG19 model. The project focuses on fine-tuning the model for style transfer by adjusting convolutional layers, ReLU activations, and employing advanced optimization techniques with LBFGS and Adam.

## Overview

Neural Style Transfer is a technique that combines the content of one image with the style of another to create a new image that retains the content of the original image while adopting the artistic style of the second image. This repository demonstrates how to achieve this using PyTorch and a pre-trained VGG19 model.

## Features

- **Pre-trained VGG19 Model**: Utilizes a VGG19 model with pre-trained weights to extract features from images.
- **Content and Style Loss**: Implements both content and style loss functions to guide the optimization process.
- **Optimizers**: Fine-tunes the target image using LBFGS and Adam optimizers.
- **Image Processing**: Includes preprocessing and postprocessing functions to handle image input and output.

## Requirements

To run the code, you need:

- Python 3.x
- PyTorch
- torchvision
- PIL (Python Imaging Library)
- matplotlib
- numpy

You can install the required Python packages using:

```bash
pip install torch torchvision pillow matplotlib numpy
```

## Usage

1. **Prepare Images**:
   - Place your content and style images in the appropriate directory.

2. **Modify Paths**:
   - Update the image paths in the script to point to your content and style images.

3. **Adjust Parameters**:
   - Customize hyperparameters such as `alpha`, `beta`, `epochs`, and `show_every` as needed.

4. **Run the Script**:
   - Execute the provided script to perform the style transfer. The script will handle image preprocessing, optimization, and result generation.

5. **View Results**:
   - The final output will include intermediate results and the stylized image, which can be visualized using matplotlib.

## Example

The script provided in this repository will automatically handle the following:

- Load and preprocess content and style images.
- Extract features from the VGG19 model.
- Compute content and style loss.
- Optimize the target image to achieve the desired style transfer.
- Display the results for visual inspection.

## Notes

- **GPU Support**: The code is designed to utilize GPU acceleration if available. If CUDA is not available, it will fall back to using the CPU.
- **Performance**: The quality of the output can vary depending on the images and the hyperparameters used. Experiment with different settings to achieve the best results.

## References

- [VGG19 Model Pre-trained Weights](#) (Insert link here)

