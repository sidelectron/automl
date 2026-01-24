# Machine Learning PDF Integration

This document outlines the methods and best practices from the Machine Learning PDF that have been integrated into our AutoML platform.

## Integrated Methods

### 1. sklearn Pipeline and ColumnTransformer

**From PDF**: Pages 6083-6427 - Machine Learning Pipelines in Scikit-Learn

**What we integrated**:
- Replaced custom pipeline implementation with sklearn's `Pipeline` and `ColumnTransformer`
- Proper use of `ColumnTransformer` for parallel transformations on different column types
- Pipeline prevents data leakage by ensuring preprocessing parameters are calculated only on training data
- Support for `remainder='passthrough'` to keep non-transformed columns

**Code Location**: `src/ml/preprocessing/pipeline.py`

**Key Features**:
- Uses `ColumnTransformer` to apply different transformations to numeric vs categorical columns
- Proper feature name handling with `get_feature_names_out()`
- Pipeline can be saved/loaded with pickle for production deployment

### 2. Cross-Validation with cross_val_score

**From PDF**: Pages 6364-6383 - Cross-Validation Integration

**What we integrated**:
- Proper use of `cross_val_score` for model evaluation
- StratifiedKFold for classification tasks
- KFold for regression tasks
- Cross-validation runs on entire pipeline (preprocessing + model)

**Code Location**: `src/ml/training/trainer.py`

**Key Features**:
- `train_with_cv()` method returns mean, std, and all fold scores
- `train_pipeline_with_cv()` for evaluating complete pipelines
- Automatic CV strategy selection based on task type

### 3. Hyperparameter Tuning (GridSearchCV & RandomizedSearchCV)

**From PDF**: Pages 7486-7540, 30000-30224 - Automated Selection with GridSearchCV

**What we integrated**:
- `HyperparameterTuner` class supporting both GridSearchCV and RandomizedSearchCV
- Default parameter grids for common models (Random Forest, XGBoost, LightGBM, Logistic Regression)
- Tuning can be applied to entire pipelines (preprocessing + model)
- Support for different scoring metrics based on task type

**Code Location**: `src/ml/training/hyperparameter_tuner.py`

**Key Features**:
- GridSearchCV for exhaustive search
- RandomizedSearchCV for faster approximate search
- Parameter grids use sklearn's double underscore notation (e.g., `model__n_estimators`)
- Returns best parameters, best score, and best estimator

### 4. Improved Imputation with Missing Indicators

**From PDF**: Pages 7544-7577 - Missing Indicator

**What we integrated**:
- All imputation strategies now use `add_indicator=True` by default
- This creates binary features indicating which values were missing
- Helps models learn patterns from missingness itself

**Code Location**: `src/ml/preprocessing/pipeline.py` (TRANSFORMER_MAP)

**Example**:
```python
SimpleImputer(strategy="mean", add_indicator=True)
```

### 5. Feature Selection with SelectKBest

**From PDF**: Pages 6256, 6296 - Feature Selection in Pipelines

**What we integrated**:
- Support for `SelectKBest` in preprocessing pipelines
- Automatic selection of score function (chi2 for classification, f_regression for regression)
- Integrated into pipeline for proper cross-validation

**Code Location**: `src/ml/preprocessing/pipeline.py` (build_from_strategy method)

### 6. Mathematical Transformations

**From PDF**: Pages 6430-7074 - Mathematical Transformations

**What we integrated**:
- Log transformation using `FunctionTransformer`
- Square root transformation
- Power transformation (Box-Cox, Yeo-Johnson)
- All transformations integrated into pipeline

**Code Location**: `src/ml/preprocessing/pipeline.py` (TRANSFORMER_MAP - transformation section)

### 7. Proper Train/Test Split

**From PDF**: Pages 7556-7575 - Workflow: Always split before fitting

**What we integrated**:
- Train/test split happens BEFORE any preprocessing fitting
- Prevents data leakage by ensuring test set statistics don't influence training
- Stratified split for classification tasks

**Code Location**: `src/engine/executor.py`, `src/ml/training/trainer.py`

## Best Practices Adopted

### 1. Pipeline Construction Pattern

**From PDF**: Pages 6301-6308

**Pattern**:
```python
pipe = Pipeline([
    ('step1', transformer1),
    ('step2', transformer2),
    ('step3', model)
])
```

**Our Implementation**: Uses same pattern in `PreprocessingPipeline.build_from_strategy()`

### 2. ColumnTransformer for Mixed Data Types

**From PDF**: Pages 5983-6042

**Pattern**:
```python
trf = ColumnTransformer([
    ('impute_age', SimpleImputer(), [2]),
    ('ohe_sex', OneHotEncoder(), [1])
], remainder='passthrough')
```

**Our Implementation**: Automatically groups transformations by type and applies via ColumnTransformer

### 3. Cross-Validation on Pipeline

**From PDF**: Pages 6382-6383

**Pattern**:
```python
cross_val_score(pipe, X_train, y_train, cv=5, scoring='accuracy').mean()
```

**Our Implementation**: `train_pipeline_with_cv()` method

### 4. Hyperparameter Tuning on Pipeline

**From PDF**: Pages 7521-7540

**Pattern**:
```python
grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(X_train, y_train)
```

**Our Implementation**: `HyperparameterTuner.tune()` method

## Benefits

1. **Data Leakage Prevention**: Pipeline ensures preprocessing parameters are calculated only on training data
2. **Production Ready**: Pipelines can be saved and loaded, making deployment straightforward
3. **Proper Evaluation**: Cross-validation on entire pipeline gives realistic performance estimates
4. **Automated Tuning**: Hyperparameter tuning can optimize both preprocessing and model parameters
5. **Industry Standard**: Using sklearn's standard tools makes code maintainable and familiar

## Usage Example

```python
from src.ml.preprocessing import PreprocessingPipeline
from src.ml.training import HyperparameterTuner, ModelTrainer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# Build preprocessing pipeline
preprocessing = PreprocessingPipeline()
preprocessing.build_from_strategy(strategy, X=X_train)
X_processed = preprocessing.fit_transform(X_train, y_train)

# Create full pipeline with model
full_pipeline = Pipeline([
    ('preprocessing', preprocessing.pipeline),
    ('model', RandomForestClassifier())
])

# Tune hyperparameters
tuner = HyperparameterTuner(method="random", n_iter=10)
param_grid = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [10, 20, None]
}
results = tuner.tune(full_pipeline, param_grid, X_train, y_train, task_type="binary_classification")

# Cross-validate
trainer = ModelTrainer()
cv_results = trainer.train_pipeline_with_cv(full_pipeline, X_train, y_train, task_type="binary_classification")
```

## Future Enhancements

Based on PDF content, we could also add:
1. Mixed variable handling (extracting numeric/categorical parts from strings)
2. Date/time feature extraction
3. Polynomial features
4. More advanced feature selection methods
5. Outlier detection and handling
