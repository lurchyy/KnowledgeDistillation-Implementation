# Knowledge Distillation Implementation

This repository contains a Jupyter notebook demonstrating the implementation and principles of Knowledge Distillation, a technique for transferring knowledge from a large, complex model (teacher) to a smaller, more efficient model (student).

## Overview

Knowledge Distillation is a model compression technique introduced by Geoffrey Hinton and his team. This implementation showcases how to distill knowledge from a larger neural network (BERT) to a smaller one (DistilBERT) while maintaining comparable performance on an intent classification task.

## Contents

- `kdistill.ipynb`: A comprehensive Jupyter notebook that walks through:
  - Implementation of custom KDTrainer and KDTrainingArgs classes for knowledge distillation
  - Using BERT as a teacher model and DistilBERT as a student model
  - Training methodology using soft targets with temperature scaling
  - Performance evaluation on the CLINC intent classification dataset
  - Visualization of training progress and accuracy metrics

## Key Concepts Demonstrated

- **Temperature Scaling**: Implementation of temperature parameter (0.2) to soften probability distributions
- **Custom Loss Function**: KL divergence loss combined with cross-entropy loss using an alpha parameter
- **Performance Analysis**: Achieving 94.4% accuracy after 10 epochs of distillation training
- **Model Architecture**: Using pre-trained BERT (teacher) and DistilBERT (student) models

## Technical Details

The implementation demonstrates knowledge distillation using:
- BERT (bert-base-uncased-finetuned-clinc) as the teacher model
- DistilBERT (distilbert-base-uncased) as the student model
- The CLINC intent classification dataset with multiple intent categories
- A combined loss function with adjustable alpha parameter to balance between mimicking the teacher and predicting ground truth
- Training over 10 epochs with learning rate of 2e-5 and batch size of 48

## Performance

The distilled model shows impressive performance improvement during training:
- Starting with 73.8% accuracy after the first epoch
- Reaching 94.4% accuracy by the end of training (epoch 10)
- Training loss reduction from 3.74 to 0.06

## Applications

This knowledge distillation approach can be applied to:
- Deploying more efficient models for intent classification in conversational AI
- Reducing model size while maintaining high accuracy for production environments
- Creating faster models for real-time natural language understanding tasks

## References

- Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the Knowledge in a Neural Network. arXiv preprint arXiv:1503.02531.
- Hugging Face Transformers library
- CLINC intent classification dataset
