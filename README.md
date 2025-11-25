# Optuna - Hyperparameter Tuning Framework

A comprehensive guide to understanding and using Optuna for hyperparameter optimization in Machine Learning and Deep Learning projects.

---

## Introduction

**Optuna** is a hyperparameter tuning framework that allows you to efficiently optimize hyperparameters for your machine learning and deep learning models. It uses intelligent search strategies (primarily Bayesian optimization) to find the best hyperparameter values faster and more effectively than traditional methods like Grid Search and Random Search.

Optuna has become one of the preferred ways of doing hyperparameter tuning in the industry today, making it an essential tool for any ML practitioner.

---

## Why Optuna?

### Grid Search Limitations

Grid Search is a straightforward but computationally expensive approach:
- It tries **every possible combination** of hyperparameters
- For large search spaces, this becomes infeasible
- Example: With 5 values each for 2 parameters = 25 combinations
- With 20 values for one parameter and 5 for another = 100 combinations
- With deep learning models on large datasets, training can take hours per trial

**The Problem**: While Grid Search guarantees finding the best combination (if it exists in your grid), it's computationally prohibitive for real-world scenarios.

### Random Search Limitations

Random Search is a simplified approach:
- Randomly samples combinations from the search space
- More efficient than Grid Search
- However, it's **non-intelligent** - it doesn't learn from previous trials
- May miss the best hyperparameter values if they weren't randomly sampled

**The Problem**: Random Search might miss optimal regions of the hyperparameter space, leading to suboptimal results.

### Bayesian Optimization

Optuna uses **Bayesian Optimization** (specifically Tree-structured Parzen Estimator - TPE), which offers the best of both worlds:

1. **Intelligent Search**: Learns from previous trials to suggest better hyperparameter values
2. **Computationally Efficient**: Focuses on promising regions rather than exhaustive search
3. **Adaptive**: Each trial informs the next one, continuously improving

**How it works**:
- Optuna establishes that there's a mathematical relationship between hyperparameters and the objective (e.g., accuracy)
- As trials are conducted, it builds a probabilistic model of this relationship
- Uses this model to predict which hyperparameter combinations are most likely to yield better results
- Intelligently focuses computational resources on promising regions

**Key Advantage**: Previous search results help future searches, making the optimization process intelligent and efficient.

---

## Core Concepts

Understanding these terms is crucial for working with Optuna:

### Study

A **Study** in Optuna is an optimization session that encompasses multiple trials. It's essentially the overall experiment where you're trying to find the best hyperparameter values for your model.

```python
study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
```

- Contains multiple trials
- Stores the history of all optimization attempts
- Can be thought of as the complete hyperparameter tuning process for one problem

### Trial

A **Trial** is a single run/experiment within a study. Each trial:
- Tests one specific combination of hyperparameter values
- Trains the model with those hyperparameters
- Evaluates and returns the objective value (e.g., accuracy)

```python
def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 200)
    max_depth = trial.suggest_int('max_depth', 3, 20)
    # ... train model and return accuracy
```

### Trial Parameters

**Trial Parameters** are the specific hyperparameter values chosen for a particular trial. For example:
- Trial 1: `max_depth=5`, `n_estimators=100`
- Trial 2: `max_depth=10`, `n_estimators=150`

These are the values suggested by the sampler for each trial.

### Objective Function

The **Objective Function** is the relationship you're trying to optimize. It:
- Takes hyperparameter values as input
- Trains the model with those values
- Returns the metric you want to optimize (e.g., accuracy, loss)

```python
def objective(trial):
    # Get hyperparameter values
    n_estimators = trial.suggest_int('n_estimators', 50, 200)
    max_depth = trial.suggest_int('max_depth', 3, 20)
    
    # Create and train model
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    score = cross_val_score(model, X_train, y_train, cv=3).mean()
    
    # Return the value to optimize
    return score
```

### Sampler

The **Sampler** is the core intelligence behind Optuna. It:
- Decides which hyperparameter values to try next
- Uses information from previous trials
- Implements the Bayesian optimization strategy (by default, TPE - Tree-structured Parzen Estimator)

**Common Samplers**:
- `TPESampler` (default): Bayesian optimization using TPE
- `RandomSampler`: Random search
- `GridSampler`: Grid search

---

## Installation

Install Optuna using pip:

```bash
pip install optuna
```

For visualization support:

```bash
pip install plotly
```

For integration with specific libraries:

```bash
pip install optuna-integration[xgboost]
```

---

## Quick Start

Here's a minimal example to get started:

```python
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Define objective function
def objective(trial):
    # Suggest hyperparameter values
    n_estimators = trial.suggest_int('n_estimators', 50, 200)
    max_depth = trial.suggest_int('max_depth', 3, 20)
    
    # Create model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    
    # Evaluate using cross-validation
    score = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy').mean()
    
    return score

# Create study
study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())

# Optimize
study.optimize(objective, n_trials=50)

# Get best results
print(f'Best accuracy: {study.best_trial.value}')
print(f'Best parameters: {study.best_trial.params}')
```

---

## Features

### Multiple Samplers

Optuna provides flexibility to use different search strategies:

#### 1. TPESampler (Default - Bayesian Optimization)

```python
study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler()
)
```

Best for: Most scenarios where you want intelligent optimization.

#### 2. RandomSampler

```python
study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.RandomSampler()
)
```

Best for: Quick baseline or when you want random exploration.

#### 3. GridSampler

```python
search_space = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [5, 10, 15, 20]
}

study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.GridSampler(search_space)
)
```

Best for: Small, discrete search spaces where you want exhaustive search.

### Visualizations

Optuna provides powerful visualization tools to understand the optimization process:

#### 1. Optimization History Plot

Shows how the objective value improves over trials:

```python
from optuna.visualization import plot_optimization_history

plot_optimization_history(study).show()
```

**Insight**: Helps determine if you need more trials or if optimization has plateaued.

#### 2. Parallel Coordinate Plot

Visualizes relationships between hyperparameters and objective value:

```python
from optuna.visualization import plot_parallel_coordinate

plot_parallel_coordinate(study).show()
```

**Insight**: Identifies promising regions and hyperparameter interactions.

#### 3. Slice Plot

Shows distribution of trials across each hyperparameter:

```python
from optuna.visualization import plot_slice

plot_slice(study).show()
```

**Insight**: Reveals which ranges of each hyperparameter were most explored.

#### 4. Contour Plot

3D surface showing the relationship between two hyperparameters and objective:

```python
from optuna.visualization import plot_contour

plot_contour(study).show()
```

**Insight**: Identifies hotspots in the hyperparameter space where best results occur.

#### 5. Parameter Importance Plot

Shows which hyperparameters are most important:

```python
from optuna.visualization import plot_param_importances

plot_param_importances(study).show()
```

**Insight**: Helps prioritize which hyperparameters matter most for your problem.

### Define-by-Run

One of Optuna's most powerful features is **define-by-run**, which allows you to create dynamic search spaces. This is particularly useful when:

- You want to optimize multiple algorithms simultaneously
- Your search space depends on other hyperparameters
- You need conditional hyperparameters

**Example: Optimizing Multiple Algorithms**

```python
def objective(trial):
    # Choose which algorithm to use
    classifier_name = trial.suggest_categorical(
        'classifier', 
        ['SVM', 'RandomForest', 'GradientBoosting']
    )
    
    if classifier_name == 'SVM':
        # SVM-specific hyperparameters
        c = trial.suggest_float('C', 0.1, 100, log=True)
        kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly'])
        model = SVC(C=c, kernel=kernel)
        
    elif classifier_name == 'RandomForest':
        # RandomForest-specific hyperparameters
        n_estimators = trial.suggest_int('n_estimators', 50, 300)
        max_depth = trial.suggest_int('max_depth', 3, 20)
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth
        )
        
    elif classifier_name == 'GradientBoosting':
        # GradientBoosting-specific hyperparameters
        n_estimators = trial.suggest_int('n_estimators', 50, 300)
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
        model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate
        )
    
    # Train and evaluate
    score = cross_val_score(model, X_train, y_train, cv=3).mean()
    return score

# Run optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

# Check which algorithm was selected most often
df = study.trials_dataframe()
print(df['params_classifier'].value_counts())
```

**Benefits**:
- Optimize algorithm selection AND hyperparameters in one go
- Bayesian optimization intelligently focuses on better-performing algorithms
- Saves time by not manually tuning each algorithm separately

---

## Examples

This repository includes a comprehensive Jupyter notebook (`optuna_basics.ipynb`) with examples covering:

1. **Basic Optuna Usage**: Simple hyperparameter tuning for Random Forest
2. **Different Samplers**: TPESampler, RandomSampler, and GridSampler
3. **Visualizations**: All major visualization types
4. **Multi-Algorithm Optimization**: Using define-by-run to optimize multiple models
5. **XGBoost Integration**: Advanced example with pruning callbacks

### Running the Examples

```bash
jupyter notebook optuna_basics.ipynb
```

Or using JupyterLab:

```bash
jupyter lab optuna_basics.ipynb
```

---

## Key Advantages

1. **Intelligent Optimization**: Bayesian optimization learns from previous trials
2. **Flexibility**: Multiple samplers and define-by-run capabilities
3. **Rich Visualizations**: Comprehensive plotting tools for analysis
4. **Industry Standard**: Widely used in production ML pipelines
5. **Framework Agnostic**: Works with scikit-learn, PyTorch, TensorFlow, XGBoost, etc.
6. **Distributed Computing**: Can distribute trials across multiple devices
7. **Pruning**: Early stopping for unpromising trials to save computation

---

## Best Practices

1. **Start with TPE Sampler**: Default Bayesian optimization usually works best
2. **Set Appropriate Trials**: 50-100 trials is a good starting point
3. **Use Cross-Validation**: Get robust estimates of model performance
4. **Visualize Results**: Always plot optimization history and parameter importance
5. **Save Studies**: Use database storage for long-running optimizations
6. **Consider Pruning**: Enable pruning to stop unpromising trials early

---

## Resources

### Official Documentation
- [Optuna Documentation](https://optuna.org/)
- [Optuna GitHub Repository](https://github.com/optuna/optuna)

---

## Summary

Optuna is a powerful hyperparameter tuning framework that:

- ✅ Uses intelligent Bayesian optimization (TPE)
- ✅ Provides flexible search strategies
- ✅ Offers rich visualization tools
- ✅ Supports define-by-run for dynamic search spaces
- ✅ Integrates with popular ML/DL frameworks
- ✅ Scales from simple to complex optimization problems

Whether you're working on machine learning or deep learning projects, Optuna provides the tools you need to efficiently find optimal hyperparameters and improve your model performance.

---

**Happy Hyperparameter Tuning! 🚀**

