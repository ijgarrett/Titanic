# Titanic Survival Prediction

This project uses the [Kaggle Titanic dataset](https://www.kaggle.com/c/titanic) to build a machine learning model that predicts whether a passenger survived the sinking of the RMS Titanic. I completed this project from July 29 to July 30, implementing a full pipeline for data preprocessing, model training, and prediction generation.

## Project Overview

The objective is to predict passenger survival using features such as age, class, fare, sex, and port of embarkation. The project uses a Random Forest Classifier and achieves a final accuracy score of 0.77990 on the Kaggle test set.

## Dataset

- `train.csv`: Contains labeled data used for training and validation.
- `test.csv`: Contains unlabeled data for which predictions are submitted to Kaggle.

## Tools and Libraries

- Python
- NumPy and Pandas
- Matplotlib and Seaborn (for data visualization)
- scikit-learn (for machine learning, pipelines, preprocessing, and model tuning)

## Process and Methodology

### 1. Exploratory Data Analysis
- Checked for missing values
- Explored distributions of `Survived`, `Pclass`, and other key features
- Visualized feature correlations with a heatmap

### 2. Data Cleaning and Preprocessing
- Created a pipeline to ensure consistent preprocessing:
  - Imputed missing values in the `Age` column using the mean
  - One-hot encoded `Sex` and `Embarked`
  - Dropped unnecessary or high-missing columns like `Cabin`, `Name`, and `Ticket`

### 3. Train/Test Split
- Used `StratifiedShuffleSplit` to maintain proportional representation of `Survived`, `Pclass`, and `Sex` in both training and test sets

### 4. Feature Scaling
- Standardized numerical features using `StandardScaler`

### 5. Model Selection and Tuning
- Used `RandomForestClassifier`
- Tuned hyperparameters (`n_estimators`, `max_depth`, `min_samples_split`) using `GridSearchCV` with 3-fold cross-validation
- Selected the best-performing model

### 6. Evaluation
- Final model achieved approximately 81.6% accuracy on the internal test split
- Achieved 0.77990 accuracy on the Kaggle test set (final submission)

### 7. Submission
- Generated predictions using the cleaned and transformed test dataset
- Saved results as `predictions.csv` in the required format

## Final Model Performance

- Internal Validation Accuracy: ~81.6%
- Kaggle Submission Accuracy: 0.77990

## Timeline

Project completed from July 29 to July 30, 2025.

## Future Improvements

- Engineer additional features (such as extracting titles from names)
- Experiment with other models (such as XGBoost or Logistic Regression)
- Analyze feature importance and misclassifications
- Automate the pipeline for use on new datasets

---

Thank you for checking out this project. Feedback and suggestions are always welcome.

