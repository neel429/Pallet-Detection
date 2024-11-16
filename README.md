# YOLOv11 Object Detection Pipeline

This project demonstrates the development of an Pallets detection pipeline, from data annotation to model training and optimization for deployment.

---

## Table of Contents
1. [Data Annotation](#data-annotation)
2. [Model Training](#model-training)
3. [Model Conversion and Optimization](#model-conversion-and-optimization)
4. [Performance Summary](#performance-summary)

---

## Data Annotation

### Process
- **Tool Used**: [Roboflow](https://roboflow.com)  
- **Subset Annotated**: Annotated 160 images manually becuase the automated data annotation of DINO did not give desired annotations.  
- **Augmentation Techniques**:
  - Increased brightness by 22%.
  - Introduced random rotations to the images.
- **Final dataset**:
  - Total size: 354 images
  - Train: 309 images
  - Test: 15 images
  - Validation: 30 images

These augmentations enriched the dataset and helped improve model robustness during training. One mistake that I made was a relatively small size of validation and test dataset.
---

## Model Training

### Model Used
- **YOLOv11**: The latest YOLO (You Only Look Once) architecture for object detection.  

### Results
- **mAP50 Score**: 0.509.  
Despite the limited dataset size, the model performs exceptionally well under the conditions.

---

## Model Conversion and Optimization

### Steps
1. **Convert to ONNX**:
   - The trained YOLOv11 model was converted into ONNX format.  
2. **Quantization**:
   - Quantized the weights of the ONNX model for reduced size and faster inference.  
3. **Convert to TensorRT**:
   - Converted the ONNX model to TensorRT format, introducing FP16 precision for optimized performance on supported hardware.  

---

## Performance Summary

- **Annotation Strategy**: Focused on a subset of data with augmentation to compensate for limited annotations.  
- **Model Accuracy**: Achieved a mAP50 score of 0.509 given the limited data (could be improved).   
- **Optimization**: Successfully prepared the model for deployment using TensorRT with FP16 precision.

---
