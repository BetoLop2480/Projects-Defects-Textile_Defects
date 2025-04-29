# Projects-Defects-Textile_Defects
In this project, we implement a simple but still effective approach for determining defects in textile samples.


1) First we select samples of the dataset to create the subset of instances to train and validate the proposal approach.
   In our proposal to used the K-fold strategy to statistically validate the model.

3) We aim to create a simple system using a feature extraction stage and machine learning techniques.
   We propose to use the widely used features of Haralick, such as the co-ocurrence matrix (GLCM). 
   From the GLCM, we compute 5 features: "Energy", "Contrast", "Homogeneity", "Entropy", and "Disimilarity".  
   For each image, a feature vector of 5 elements is calculated.

4) We calculate a classification model using a machine learning technique. Specifically, we use the Support Vector Machine
   configured with a Radial Basis Function Kernel. We aim to create a model to distinguish between defective textiles and those without defects.
   Parameters:
   kernel=RBF: Used for create non-linear boundaries decision. K(x, y) = exp(-γ ||x - y||²)
   C=1.0: Regularization factor. Higher values tends to overfitting the model. Lower values, allows a better generalization.
   gamma='scale': 1/(n_features * X.var()) determines the influence that each training sample has on their neighbors. Higher values, most on the neighbors.
   Lower values widely influence.

 5) Finally, we evaluate the model using two metrics: Accuracy and F1-score. In accuracy our proposal attains 0.854, which can be considered as a acceptable performance.
    In fact, considering that we used handcrafted features the model is still competitive. In terms of the F1-score, we compute 0.856.

Our model attained competitive Accuracy and F1-score. The GLCM has demonstrated to be a suitbale feature for binary classification in samples of textiles. 
The feature vector computed is small but still effective to construct a SVM model to binary classification. 
   
