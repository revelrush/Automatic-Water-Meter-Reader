# Automatic-Water-Meter-Reader

This is the official repository for the Automated Water Meter Reading Through Image Recognition project by Concio et al. for TENCON 2022

## About the Project
In many parts of the world, water utility companies still heavily rely on manual meter reading, requiring readings to be transcribed by hand, which is inefficient and leaves room for human errors. Smart metering infrastructures have been proposed as a solution, but while effective and convenient, are costly and impractical to implement on a broader scale. This work presents a solution compatible with the existing infrastructure using image recognition techniques to create an end-to-end system for automated water meter reading. This repository covers the codebase for the automated image recognition pipeline created by the project proponents to extract the meter reading value from an image of a water meter.

## About the Pipeline
The pipeline follows the architechture laid out by past works on image based automatic meter reading, which is to utilize deep learning methods, and to split the task into two stages to improve the performance of the pipeline. These are the counter detection stage, which detects the location of the meter counter, and the counter recognition stage, that identifies the digits within the counter. The counter detection stage was implemented using a binary segmentation model with the U-Net architecture using a Resnet34 backbone, which was created using the [Segmentation Models library](https://github.com/qubvel/segmentation_models). On the other hand, the counter recognition stage was implemented using an object detection model using the Faster RCNN architecture with a ResNet101 backbone created with the [TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection). Aside from these, there are extra processing stages before each of the abovementioned stages and a final post-processing stage. The final completed pipeline ca be seen in the image below.

<img src="pipeline v2.png" width=800px />
<h5>Complete Pipeline Architechture</h5>

## Image Datasets Used
The study used two open-access image datasets to train and test the deep learning models within the pipeline. The primary dataset utilized was created by Roman K. called the [Water Meters Dataset](https://www.kaggle.com/datasets/tapakah68/yandextoloka-water-meters-dataset) consisting of 1244 images of water meters with the meter counter within each image completely annotated, and was also partially annotated by [Olivier K.](https://www.kaggle.com/datasets/merrickolivier/water-meter-ocr-images) which added individual annotations for each digit within the meter counter to some of the images. The secondary dataset utilized was the [UFPR-AMR Dataset](https://web.inf.ufpr.br/vri/databases/ufpr-amr/) created by Laroca et al., which was used to train an image-based electric meter reader, which consists of 2,000 images of electrical meters, all with annotations for the meter counter and the meter digits.  

## How-to
1. Install all the required libraries listed in requirements.txt. The command pip install -r requirements.txt can be used to install everything.
2. Download the model weights for the deep learnig models utilized [here](https://drive.google.com/file/d/1cR7rT8JEVS6iYMnm5x-ewDNTKNfmeuox/view?usp=sharing), unzip the files and put into the working directory. Afterwards, edit Image-Based AMR.py with the respective location of these files.
3. Run Image-Based AMR.py, afterwards indicate the image to be processed, the number of digits and the number of whole digits. In addition to outputting the meter reading value, an image file for the processed image will be also created for the user to view, like so:

<img src="sample out.jpg" width=400px />

## Acknowledgements
In addition to the codebases and datasets mentioned thus far, the [deskew library](https://github.com/sbrunner/deskew) by sbrunner was also used during post processing. The research folder was taken from the TensorFlow Object Detection API, the project does not claim ownership of its contents in any way, its inclusion is merely for convenience.

## Miscellaneous
The conference paper can be found in IEEE (link to be added), and a full, more in-depth version of the paper is available in the repository for those interested. 
