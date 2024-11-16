# YOLOv11 Object Detection Pipeline

This project demonstrates the development of an object detection pipeline, from data annotation to model training and optimization for deployment.

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
- **Subset Annotated**: Annotated 160 images manually due to issues with the DINO data annotation tool.  
- **Augmentation Techniques**:
  - Increased brightness by 22%.
  - Introduced random rotations to the images.

These augmentations enriched the dataset and helped improve model robustness during training.

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
   - The trained YOLOv11 model was converted into ONNX format for compatibility and deployment flexibility.  
2. **Quantization**:
   - Quantized the weights of the ONNX model for reduced size and faster inference.  
3. **Convert to TensorRT**:
   - Converted the ONNX model to TensorRT format, leveraging FP16 precision for optimized performance on supported hardware.  

---

## Performance Summary

- **Annotation Strategy**: Focused on a subset of data with augmentation to compensate for limited annotations.  
- **Model Accuracy**: Achieved a respectable mAP50 score of 0.509 given the limited data.  
- **Optimization**: Successfully prepared the model for deployment using TensorRT with FP16 precision for high-performance inference.

---

## Future Work

- Expand the annotated dataset to improve model accuracy further.  
- Experiment with additional augmentation techniques.  
- Deploy the TensorRT-optimized model in a real-time application.  

---

## Acknowledgments

- Thanks to [Roboflow](https://roboflow.com) for the annotation tool.  
- Inspired by the advancements in YOLO object detection architectures.

---
