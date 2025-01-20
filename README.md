# Human Activity Recognition Analysis

## Overview

This project implements a comprehensive machine learning pipeline to classify human activities using smartphone sensor data from the UCI HAR Dataset. Multiple models are compared to determine the most effective approach for activity recognition.

## Dataset Details

-   **Size**: 10,299 samples (7,352 training + 2,947 test)
-   **Dimensions**: 561 features
-   **Classes**: 6 activities (Walking, Walking Upstairs, Walking Downstairs, Sitting, Standing, Laying)

## Analysis Pipeline

### 1. Data Preprocessing

-   Feature standardization with StandardScaler
-   PCA reduction (95% variance retention)
-   Dataset visualization and distribution analysis

### 2. Model Implementation

#### Unsupervised Learning

1. K-Means Clustering

    - Initial k=6 implementation
    - Optimal k analysis via elbow method
    - Binary clustering (moving vs. stationary)

2. Agglomerative Clustering
    - Ward linkage method
    - 6-cluster configuration
    - Hierarchical structure analysis

#### Supervised Learning

1. Support Vector Classification

    - Without PCA: Full feature set
    - With PCA: Reduced dimensionality
    - GridSearchCV optimization (C: 0.001 to 10)

2. Random Forest

    - Base model (100 estimators)
    - Tuned model with GridSearchCV
    - Parameter optimization:
        - n_estimators: [100, 200]
        - max_depth: [10, 20, None]
        - min_samples_split: [2, 5]
        - min_samples_leaf: [1, 2]

3. XGBoost

    - Hyperparameter tuning:
        - max_depth: [3, 5]
        - learning_rate: [0.01, 0.1]
        - n_estimators: [100, 200]
        - min_child_weight: [1, 3]

4. Neural Network (MLP)
    - Architecture optimization:
        - Hidden layers: [(100,), (100, 50)]
        - Activation: [relu, tanh]
        - Learning rate: [0.001, 0.01]

## Results

### Model Performance Comparison

| Model                 | Accuracy |
| --------------------- | -------- |
| SVC (without PCA)     | 96.20%   |
| MLP Classifier        | 95.11%   |
| XGBoost               | 94.44%   |
| SVC (with PCA)        | 92.94%   |
| Random Forest         | 92.67%   |
| Random Forest (tuned) | 92.33%   |
| Agglomerative         | 50.04%   |
| KMeans                | 41.99%   |

### Key Findings

1. Supervised models significantly outperform unsupervised approaches
2. SVC without PCA achieves the highest accuracy (96.20%)
3. PCA maintains good performance while reducing complexity
4. Neural network and XGBoost show competitive results
5. Clustering models effectively separate mobile/stationary activities

## Usage

1. Install dependencies:

```bash
pip install numpy pandas scikit-learn xgboost matplotlib tqdm
```

2. Execute notebook:

```bash
jupyter notebook main.ipynb
```

## Dependencies

-   Python 3.x
-   NumPy
-   Pandas
-   Scikit-learn
-   XGBoost
-   Matplotlib
-   tqdm

## License

MIT License
