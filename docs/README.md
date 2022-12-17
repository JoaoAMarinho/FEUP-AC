# Aprendizagem Computacional

## Table of contents
- **[Structure](#structure)**
- **[Setup](#setup)**
- **[Usage](#usage)**
- **[Authors](#authors)**

## Structure

The project is structured in 4 main folders: `data`, `docs`, `results` and `src`.

The Data folder keeps all *.csv* files, the train and competition, as well as the cleaned ones.

The Docs folder contains all the project documentation.

The Results folder stores the achieved *.csv* files and respective model configuration after predicting the probabilities over the competition *.csv* files.

The Src folder includes all the notebooks and remaining *.py* files.

## Setup

#### 1. Install dependencies

```sh
$ pip install -r /src/requirements.txt.
```

## Usage
#### 1. Exploratory analysis:
```
run exploratory_analysis notebook
```

#### 2. Descriptive analysis:
```
run descriptive_analysis notebook
```

#### 3. Data preparation:
```
run data_preparation notebook
```

#### 4. Model validation:

Warning: The `machine_learning` notebook compares all model results (AUC score) with all prepared dataframes, it may take a while to run the whole script so it is not recommended running.

```
run machine_learning notebook
```

#### 5. Model submission:

The `model_submission` notebook is used for submitting the wanted model with the best parameters.

```
run model_submission notebook
```

## Authors

| Name             | Number    | Work percentage             |
| ---------------- | :---------: | :------------------: |
| Beatriz Aguiar   | 201906230 | 33.33% |
| João Marinho     | 201905952 | 33.33% |
| Margarida Vieira | 201907907 | 33.33% |
