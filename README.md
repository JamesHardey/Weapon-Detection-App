# Weapon-Detection-App

![Weapon Detection Demo](demo.gif)

## Overview

This repository contains a Weapon Detection System that utilizes a YOLOv8 model for detecting various weapons in images and videos. The system is integrated into a Streamlit application, making it easy to use and deploy. The trained model can identify weapons such as axes, guns, knives, and various types of firearms.

![Weapon Detection Architecture](architecture.png)

## Features

- Detect weapons in images and videos
- Streamlit application for easy user interaction
- Supports GPU acceleration for real-time video detection
- Pretrained YOLOv8 model for weapon detection

## Getting Started

Follow these steps to set up and run the Weapon Detection System:

### Prerequisites

1. **Python Environment**: Make sure you have Python 3.7 or later installed.

2. **GPU (Optional)**: For video detection with GPU acceleration, you need a compatible NVIDIA GPU and the necessary GPU drivers and libraries (CUDA, cuDNN).

### Installation

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/JamesHardey/Weapon-Detection-App.git
   cd weapon-detection-system
   ```

2. Install the required Python packages using `pip`:

   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

1. Start the Streamlit application:

   ```bash
   streamlit run app.py
   ```

2. Access the application by opening your web browser and navigating to `http://localhost:8501`.

3. Use the application to upload images or video files for weapon detection.

### Using GPU (Optional)

If you have a compatible GPU and wish to enable GPU acceleration for video detection, make sure to install the appropriate GPU libraries and CUDA-compatible PyTorch. Refer to the PyTorch website for installation instructions.

## Model Training

If you want to retrain the weapon detection model or fine-tune it for specific use cases, you can follow these steps:

1. Prepare your custom dataset with weapon images. The dataset should include annotations in YOLO format.

2. Modify the model configuration and training parameters in the `train.py` script to suit your needs.

3. Start the training process:

   ```bash
   python train.py --data data/custom.yaml --cfg models/yolov8-custom.cfg --weights weights/yolov8-custom.pt
   ```

4. Once the training is complete, replace the pretrained weights in the `weights` directory with your custom weights.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [YOLOv8](https://github.com/WongKinYiu/yolov8) - The YOLOv8 model implementation used in this project.
- [Streamlit](https://www.streamlit.io/) - Streamlit is used for creating the user-friendly web application.

## Contributors

- Your Name <jamesade646@gmail.com>
- Additional contributors can be added here.

## Support

If you have any questions or need assistance, please create an issue or contact us at your-email@example.com.

**Enjoy detecting weapons with our Weapon Detection System!**
