# Neural Style Transfer

This repository implements Neural Style Transfer (NST) using PyTorch and a pre-trained VGG19 model. The project focuses on fine-tuning the model for style transfer by adjusting **convolutional layers and ReLU activations** and employing advanced optimization techniques with **LBFGS** and **Adam**.

## Overview

Neural Style Transfer is a technique that combines the content of one image with the style of another to create a new image that retains the content of the original image while adopting the artistic style of the second image. This repository demonstrates how to achieve this using PyTorch and a pre-trained VGG19 model.

## Features

- **Pre-trained VGG19 Model**: Utilizes a VGG19 model with pre-trained weights to extract features from images.
- **Content and Style Loss**: Implements both content and style loss functions to guide the optimization process.
- **Optimizers**: Fine-tunes the target image using **LBFGS** and **Adam** optimizers.
- **Image Processing**: Includes preprocessing and postprocessing functions to handle image input and output.

## Requirements

To run the code, you need:

- Python 3.x
- PyTorch
- torchvision
- PIL (Python Imaging Library)
- matplotlib
- numpy

## Cloning the Repository

To clone this repository, use the following command:

```bash
git clone https://github.com/Aadhishreevijay/neural-style-transfer.git
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

## Notebooks

- **Convolutional Layers Style Transfer**: The notebook [nst_conv_lbfgs_adam.ipynb](https://github.com/Aadhishreevijay/neural-style-transfer/blob/main/nst_conv_lbfgs_adam.ipynb) demonstrates style transfer using convolutional layers.
- **ReLU Layers Style Transfer**: The notebook [nst_relu_lbfgs_adam.ipynb](https://github.com/Aadhishreevijay/neural-style-transfer/blob/main/nst_relu_lbfgs_adam.ipynb) demonstrates style transfer using ReLU layers.

## Output Examples

Here are a few examples of outputs generated using the style transfer method implemented in this repository:

1. ![Output 1](https://github.com/user-attachments/assets/3825d3f6-0904-47d5-8299-34c8f01bf937)
2. ![Output 2](https://github.com/user-attachments/assets/cbf3d48d-1e01-4836-b1eb-1e2015771dbb)
3. ![Output 3](https://github.com/user-attachments/assets/d71a6215-3d48-4da1-a3c3-cc8ceb6837c9)
4. ![Output 4](https://github.com/user-attachments/assets/13979308-9945-4b9b-ab93-da93f2c2e12c)
5. ![Output 5](https://github.com/user-attachments/assets/0fd90a5d-ae7a-4980-a73c-0803467718f4)

## Notes

- **GPU Support**: The code is designed to utilize GPU acceleration if available. If CUDA is not available, it will fall back to using the CPU.
- **Performance**: The quality of the output can vary depending on the images and the hyperparameters used. Users are encouraged to try different weights and optimizers to achieve the best results. Experiment with different settings to fine-tune the performance of the style transfer.
- **Optimizers**: The repository uses **LBFGS** and **Adam** optimizers. Feel free to experiment with these or other optimizers to see their effect on the style transfer results.

## References

- [VGG19 Model Pre-trained Weights](https://drive.google.com/file/d/1HMN7_oKlXHniX745z40ZT09Ffz3fZaxi/view?usp=drive_link)
