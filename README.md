🏡 California Housing Price Prediction
This project provides an end-to-end demonstration of building a machine learning model using the California Housing dataset. It covers every step from data acquisition and exploration to model deployment, showcasing a practical approach to a regression problem.

🌟 Project Overview
The goal of this project is to predict the median housing price in California districts based on various features like population, median income, housing age, etc. The project heavily utilizes the Scikit-Learn library to build a robust pipeline for data preprocessing, feature engineering, model training, and hyperparameter tuning.

Detail

Description

Problem Type

Regression (Predicting a continuous value)

Core Model

Random Forest Regressor

Performance Metric

Root Mean Squared Error (RMSE)

Final Test RMSE

$47,730

🚀 Key Features & Workflow
This project walks through several critical stages of a machine learning workflow:

Data Exploration & Visualization (EDA):

Analyzed data distributions, outliers, and feature relationships using histograms, scatter plots, and correlation matrices.

Visualized geographical data to reveal that prices are concentrated near coastal areas and major cities (Los Angeles, San Francisco).

Discovered that median_income is the strongest predictor of median_house_value.

Data Preprocessing & Feature Engineering:

Performed a stratified split of the data based on income categories using StratifiedShuffleSplit to avoid sampling bias.

Missing Value Imputation: Used SimpleImputer to fill missing numerical features with the median value.

Categorical Data Handling: Encoded the ocean_proximity feature into a numerical format using OneHotEncoder.

Feature Creation: Engineered meaningful new features like rooms_per_household, bedrooms_per_room, and population_per_household.

Custom Transformers: Built a custom Scikit-Learn transformer (ClusterSimilarity) to add features based on geographical cluster similarity.

Feature Scaling: Standardized all numerical features using StandardScaler.

Transformation Pipelines:

Combined all preprocessing steps into a single, robust pipeline using Pipeline and ColumnTransformer. This improves code reusability and makes it easy to apply the same transformations to new data.

Model Selection and Training:

Experimented with several models:

Linear Regression: Served as a baseline model but was found to underfit.

Decision Tree Regressor: Severely overfit the data.

Random Forest Regressor: Showed the most promising performance.

Cross-Validation: Used K-fold cross-validation to get a more reliable estimate of model performance.

Model Fine-Tuning:

Grid Search (GridSearchCV): Searched through a predefined grid of hyperparameters to find the optimal combination.

Randomized Search (RandomizedSearchCV): Efficiently searched a larger hyperparameter space to fine-tune the model.

Final Evaluation:

Evaluated the best model on the held-out test set to estimate its generalization error. The final model achieved an RMSE of approximately $47,730.

🛠️ Technologies Used
Python 3.x

Scikit-Learn: For data preprocessing, modeling, and evaluation.

Pandas: For data manipulation and analysis.

NumPy: For numerical operations.

Matplotlib: For data visualization.

Jupyter Notebook: As the interactive development environment.

⚙️ How to Install and Run
Clone the repository:

git clone [https://github.com/your-username/california-housing-project.git](https://github.com/your-username/california-housing-project.git)
cd california-housing-project

Create a virtual environment (recommended):

python -m venv venv
source venv/bin/activate  # On macOS/Linux
.\venv\Scripts\activate    # On Windows

Install the required packages:

pip install -r requirements.txt

(Note: Create a requirements.txt file containing pandas, numpy, scikit-learn, matplotlib, and jupyter)

Run the Jupyter Notebook:

jupyter notebook

From your browser, open the notebook file (end_to_end_project.ipynb) and run the cells to reproduce the workflow.

📈 Results & Conclusion
This project successfully built a Random Forest model to predict California housing prices. The most important features were identified as median_income, geographical location (via cluster similarity), and population_per_household.

The project serves as an excellent case study for the entire lifecycle of a data science project, highlighting best practices for writing clean, reproducible machine learning code with Scikit-Learn's Pipeline and custom transformers. The final model achieves an RMSE of $47,730 on the test set, meaning its price predictions are, on average, off by this amount.
