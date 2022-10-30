# Automatic-Water-Meter-Reader

##About
The official repository for the Automated Water Meter Reading Through Image Recognition project by Concio et al. for TENCON 2022

##About the Project
In many parts of the world, water utility companies still heavily rely on manual meter reading, requiring readings to be transcribed by hand, which is inefficient and leaves room for human errors. Smart metering infrastructures have been proposed as a solution, but while effective and convenient, are costly and impractical to implement on a broader scale. This work presents a solution compatible with the existing infrastructure using image recognition techniques to create an end-to-end system for automated water meter reading. This repository covers the codebase for the automated image recognition pipeline created by the project proponents to extract the meter reading value from an image of a water meter.

##About the Pipeline
The pipeline follows the architechture laid out by past works on image based automatic meter reading, which is to utilize deep learning methods, and to split the task into two stages to improve the performance of the pipeline. These are the counter detection stage, which detects the location of the meter counter, and the counter recognition stage, that identifies the digits within the counter. The model used for the counter detection stage was a binary segmentation model with the U-Net architecture using a Resnet34 backbone created using the [segmentation models library](https://github.com/qubvel/segmentation_models).  

This combination was chosen for its relatively light model architecture and because it has been demonstrated to perform well on the counter detection task \cite{Merric}. The model was trained using the \emph{Water Meters Dataset} for 50 epochs, and only the model weights of the epoch with the highest test accuracy were saved, which was found to be 98\% at the end of training.

\paragraph{Counter Recognition}

According to the literature, the counter recognition stage is usually implemented using an object detection model \cite{laroca2}, \cite{8923966}. An object detection model created used the Faster RCNN \cite{fasterrcnn} architecture with a ResNet101 \cite{ResNet} backbone, created using the TensorFlow Object Detection API \cite{API}.




Aside from these, there are extra processing stages before each of the abovementioned stages and a final post-processing stage. 
