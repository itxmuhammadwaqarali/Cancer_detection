# Cancer Detection Using Histopathological Images

## Objective
This project uses deep learning to detect cancerous cells from histopathological images. The goal is to build an accurate model that highlights cancerous regions in images and assists in medical diagnostics.

## Dataset
- **Source:** Breast Cancer Histopathological Dataset from Kaggle
- **Categories:** 
  - `cancerous`
  - `non_cancerous`

## Prerequisites
- Python 3.7+
- TensorFlow 2.x
- OpenCV
- Matplotlib
- scikit-learn

Install dependencies using:
```bash
pip install tensorflow opencv-python matplotlib scikit-learn
```

## Project Structure
1. **Data Loading**:
   - Loads images from the dataset directory.
   - Resizes images to 224x224 pixels.
   - Converts labels into one-hot encoded format.

2. **Data Augmentation**:
   - Augments training images using transformations like rotation, zoom, and horizontal flip.
   - Preprocessing is applied using the ResNet50 `preprocess_input` function.

3. **CNN Model**:
   - A custom convolutional neural network (CNN) built using TensorFlow.
   - Layers include Conv2D, MaxPooling2D, Flatten, Dense, and Dropout.

4. **Transfer Learning with ResNet50**:
   - Uses ResNet50 as the base model with pre-trained ImageNet weights.
   - Adds custom layers for the classification task.

5. **Grad-CAM Visualization**:
   - Highlights cancerous regions in images.
   - Visualizes the contribution of different regions to the model's predictions.

## Acknowledgments
- Kaggle for providing the dataset.
- TensorFlow and ResNet50 for pre-trained model support.

## Future Enhancements
- Implement additional preprocessing techniques.
- Experiment with other transfer learning architectures like VGG16 or EfficientNet.
- Use advanced visualization methods to interpret predictions.
