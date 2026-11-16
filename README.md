# Task 38: Classification Accuracy

## Objective
Calculate the accuracy of a classification model and compare predicted values with actual target values to understand one of the most fundamental classification evaluation metrics.

## Tools Used
- Python
- Pandas
- Scikit-learn
- Jupyter Notebook

## Dataset Overview
The project uses the Iris Dataset (`iris_dataset.csv`), containing measurements of iris flowers across three species: `setosa`, `versicolor`, and `virginica`.

**Features:**
- `sepal length (cm)`
- `sepal width (cm)`
- `petal length (cm)`
- `petal width (cm)`

**Target Variable:**
- `species`

## Methodology
1. **Data Loading:** Loaded the dataset using Pandas.
2. **Feature-Target Separation:** Separated input features (`X`) from the target label (`y`).
3. **Train-Test Split:** Split the dataset into 80% training data and 20% testing data (`random_state=42`).
4. **Model Training:** Trained a K-Nearest Neighbors (`KNeighborsClassifier` with `n_neighbors=3`) model on the training set.
5. **Prediction and Evaluation:** Predicted species on the test set and evaluated performance using `accuracy_score()`.

## Results

### Actual vs Predicted Values (Sample)
| Index | Actual | Predicted |
| :--- | :--- | :--- |
| 0 | versicolor | versicolor |
| 1 | setosa | setosa |
| 2 | virginica | virginica |
| 3 | versicolor | versicolor |
| 4 | versicolor | versicolor |
| 5 | setosa | setosa |
| 6 | versicolor | versicolor |
| 7 | virginica | virginica |
| 8 | versicolor | versicolor |
| 9 | versicolor | versicolor |

### Model Performance Metrics
- **Accuracy Score:** `1.0000`
- **Accuracy Percentage:** `100.00%`

## Performance Interpretation
The K-Nearest Neighbors classifier achieved a perfect accuracy of 1.0000 (100.00%) on the test set, correctly classifying every flower species. Because the Iris dataset is balanced, clean, and has well-separated feature boundaries—particularly between `setosa` and the other two species—basic distance-based classifiers perform exceptionally well on it.

## Interview Questions and Answers

### 1. What is classification accuracy?
**Answer:** Classification accuracy is the ratio of correctly predicted observations to the total number of observations evaluated. It measures how frequently a classification model makes the right prediction.

### 2. How is accuracy calculated?
**Answer:** Accuracy is calculated by dividing the number of correct predictions by the total number of predictions made:

`Accuracy = (Number of Correct Predictions) / (Total Number of Predictions)`

In terms of a confusion matrix:

`Accuracy = (TP + TN) / (TP + TN + FP + FN)`

Where:
- **TP:** True Positives
- **TN:** True Negatives
- **FP:** False Positives
- **FN:** False Negatives

### 3. Can accuracy always be trusted as the only evaluation metric?
**Answer:** No. Accuracy can be misleading when working with imbalanced datasets (for example, fraud detection or rare disease diagnosis where 99% of cases belong to one class). A naive model that predicts only the majority class would still achieve 99% accuracy while completely failing to identify the minority class. In such scenarios, metrics like Precision, Recall, F1-Score, and ROC-AUC should be used alongside accuracy.
