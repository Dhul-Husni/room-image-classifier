# Room condition classification

The SigLIP2-SO400M encoder supplies image features. We trained a logistic regression classifier with L2 regularization. The encoder weights stay fixed.

**Evaluation accuracy: 73.33% (44 of 60 images). Macro F1: 0.7384.**

We used 240 images for training and 60 for evaluation. Each evaluation class contains 20 images. Five-fold cross-validation used only training images. We also used evaluation results to compare models.

Property labels can be wrong for individual images. We kept all labels. We used horizontal flips and small changes to image brightness and contrast.
