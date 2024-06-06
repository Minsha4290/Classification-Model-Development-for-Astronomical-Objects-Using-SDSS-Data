### **Feature Importance and Classification Model Development for Astronomical Objects Using SDSS Data**

#### Brief Project Description
The project aims to conduct an in-depth feature importance analysis to determine which attributes are most influential in classifying astronomical objects—specifically stars, galaxies, and quasars—using the Sloan Digital Sky Survey (SDSS) DR17 dataset. Following this analysis, we will develop and evaluate machine learning models to classify these objects based on the most critical features identified. This project integrates data preprocessing, exploratory data analysis (EDA), feature importance evaluation, model development, and performance evaluation, culminating in a robust, interpretable classification model and comprehensive visualizations.

### Detailed Introduction to Stellar Classification

Stellar classification is a fundamental aspect of astronomy, focusing on categorizing celestial objects based on their spectral characteristics. This process helps astronomers understand the nature, evolution, and properties of stars, galaxies, and quasars. The Sloan Digital Sky Survey (SDSS) has significantly contributed to this field by providing extensive spectral data on a vast number of celestial objects.

#### Importance of Classifying Astronomical Objects
Classifying astronomical objects is critical for several reasons:
1. **Understanding Cosmic Evolution**: By classifying stars, galaxies, and quasars, astronomers can infer the life cycles of these objects and understand the processes of formation, evolution, and demise within the universe.
2. **Galaxy Formation and Evolution**: Classifying stars, galaxies, and quasars helps in understanding the formation and evolution of galaxies, which are the building blocks of the universe.
3. **Chemical Composition**: Spectral classification reveals the chemical composition of celestial objects, shedding light on the processes of nucleosynthesis and the enrichment of the interstellar medium.
4. **Distance Measurement**: Classifying different types of celestial objects, including those with known intrinsic luminosities, aids in measuring astronomical distances, which is crucial for mapping the scale of the universe.

#### The Role of Spectroscopy
Spectroscopy is the primary tool for classifying astronomical objects. It involves analyzing the light emitted or absorbed by objects to determine their properties. Each object, whether a star, galaxy, or quasar, emits light at specific wavelengths, producing a unique spectrum. By examining these spectra, astronomers can classify objects based on their temperature, chemical composition, and other physical properties.

In the SDSS, spectroscopy has been used extensively to gather data on millions of celestial objects. The SDSS spectrograph captures light across multiple wavelengths, from ultraviolet to infrared, providing a detailed spectrum for each object observed. This spectral data is crucial for accurate classification.

#### The SDSS DR17 Dataset
The SDSS DR17 dataset is one of the most comprehensive astronomical datasets available, containing observations of approximately 100,000 objects, including stars, galaxies, and quasars. Each observation is described by 17 feature columns and a class column identifying the type of object.
- **obj_ID**: Unique identifier for each astronomical object in the image catalog.
- **alpha**: Right Ascension angle, specifying the object's position in the sky (longitude-like coordinate).
- **delta**: Declination angle, specifying the object's position in the sky (latitude-like coordinate).
- **u**: Magnitude measurement in the ultraviolet filter of the photometric system.
- **g**: Magnitude measurement in the green filter of the photometric system.
- **r**: Magnitude measurement in the red filter of the photometric system.
- **i**: Magnitude measurement in the near-infrared filter of the photometric system.
- **z**: Magnitude measurement in the infrared filter of the photometric system.
- **run_ID**: Identifier for the specific scan or observation run.
- **rerun_ID**: Identifier for the specific processing run of the image data.
- **cam_col**: Column number of the camera used to capture the image.
- **field_ID**: Identifier for the specific field of view in the sky.
- **spec_obj_ID**: Unique identifier for optical spectroscopic objects, indicating shared classification across observations.
- **class**: Classification of the object as a star, galaxy, or quasar.
- **redshift**: Measure of the object's redshift, indicating the degree of wavelength increase due to the universe's expansion.
- **plate**: Identifier for the plate used in the SDSS spectroscopic observation.
- **MJD**: Modified Julian Date, indicating when the observation was taken.
- **fiber_ID**: Identifier for the fiber optic cable used to collect light for spectroscopic analysis.

#### Methodology
The project will begin with data preprocessing to clean and prepare the dataset for analysis. Exploratory Data Analysis (EDA) will then be conducted to visualize and understand the distribution and relationships of the features. Feature importance analysis will follow, using techniques such as Random Forest Feature Importances, SHAP (SHapley Additive exPlanations), or LIME (Local Interpretable Model-agnostic Explanations). These methods will help identify which features most significantly impact the classification of astronomical objects.

Subsequently, the project will focus on developing and evaluating various machine learning models, including Logistic Regression, Random Forest, Support Vector Machines (SVM), and Neural Networks. The models will be trained and tested using the most important features identified. Performance will be evaluated using metrics like accuracy, precision, recall, and F1-score. Hyperparameter tuning will optimize the best-performing model, and advanced interpretability techniques will ensure the model's decisions are understandable.

# Data Preprocessing

```python
# Import libraries and mount drive
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from google.colab import drive
drive.mount('/content/drive')
```
> Mounted at /content/drive

```python
# Load the dataset
data_path = '/content/drive/My Drive/star_classification.csv'
dataset = pd.read_csv(data_path)
```

```python
dataset.head()
```
>
|     obj_ID      |    alpha     |    delta     |     u      |     g      |     r      |     i      |     z      | run_ID | rerun_ID | cam_col | field_ID |     spec_obj_ID     | class  | redshift | plate |    MJD    | fiber_ID |
|-----------------|--------------|--------------|------------|------------|------------|------------|------------|--------|----------|---------|----------|---------------------|--------|----------|-------|-----------|----------|
| 1.237661e+18   | 135.689107   | 32.494632    | 23.87882   | 22.27530   | 20.39501   | 19.16573   | 18.79371   | 3606   | 301      | 2       | 79       | 6.543777e+18       | GALAXY | 0.634794 | 5812  | 56354     | 171      |
| 1.237665e+18   | 144.826101   | 31.274185    | 24.77759   | 22.83188   | 22.58444   | 21.16812   | 21.61427   | 4518   | 301      | 5       | 119      | 1.176014e+19       | GALAXY | 0.779136 | 10445 | 58158     | 427      |
| 1.237661e+18   | 142.188790   | 35.582444    | 25.26307   | 22.66389   | 20.60976   | 19.34857   | 18.94827   | 3606   | 301      | 2       | 120      | 5.152200e+18       | GALAXY | 0.644195 | 4576  | 55592     | 299      |
| 1.237663e+18   | 338.741038   | -0.402828    | 22.13682   | 23.77656   | 21.61162   | 20.50454   | 19.25010   | 4192   | 301      | 3       | 214      | 1.030107e+19       | GALAXY | 0.932346 | 9149  | 58039     | 775      |
| 1.237680e+18   | 345.282593   | 21.183866    | 19.43718   | 17.58028   | 16.49747   | 15.97711   | 15.54461   | 8102   | 301      | 3       | 137      | 6.891865e+18       | GALAXY | 0.116123 | 6121  | 56187     | 842      |

```python
dataset.info()
```
| Type                              |   Count   | Dtype   |
|-----------------------------------|:---------:|:-------:|
| obj_ID                            | 100000    | float64 |
| alpha                             | 100000    | float64 |
| delta                             | 100000    | float64 |
| u                                 | 100000    | float64 |
| g                                 | 100000    | float64 |
| r                                 | 100000    | float64 |
| i                                 | 100000    | float64 |
| z                                 | 100000    | float64 |
| run_ID                            | 100000    | int64   |
| rerun_ID                          | 100000    | int64   |
| cam_col                           | 100000    | int64   |
| field_ID                          | 100000    | int64   |
| spec_obj_ID                       | 100000    | float64 |
| class                             | 100000    | object  |
| redshift                          | 100000    | float64 |
| plate                             | 100000    | int64   |
| MJD                               | 100000    | int64   |
| fiber_ID                          | 100000    | int64   |

```python
missing_values = dataset.isnull().sum()
print("Missing values in each column:\n", missing_values)
```

| Column        | Missing Values |
|---------------|:--------------:|
| obj_ID        |       0        |
| alpha         |       0        |
| delta         |       0        |
| u             |       0        |
| g             |       0        |
| r             |       0        |
| i             |       0        |
| z             |       0        |
| run_ID        |       0        |
| rerun_ID      |       0        |
| cam_col       |       0        |
| field_ID      |       0        |
| spec_obj_ID   |       0        |
| class         |       0        |
| redshift      |       0        |
| plate         |       0        |
| MJD           |       0        |
| fiber_ID      |       0        |

```python
numeric_columns = dataset.select_dtypes(include=[np.number]).columns
plt.figure(figsize=(15, 10))
for i, column in enumerate(numeric_columns, 1):
    plt.subplot(5, 4, i)
    sns.boxplot(data=dataset, x=column)
    plt.title(f'Boxplot of {column}')
plt.tight_layout()
plt.show()
```
![Box Plot](/Graphs/boxplot_1.png)
Format: ![Alt Text](url)
