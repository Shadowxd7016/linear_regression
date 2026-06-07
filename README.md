# Zameen.com Property Price Predictor

This repository contains a Jupyter Notebook that scrapes property data from **Zameen.com**, cleans and preprocesses the dataset, and builds a **Multiple Linear Regression model from scratch** using NumPy to predict property prices in Islamabad.

---

## ## Project Overview

The goal of this project is to predict real estate prices in Islamabad based on features like **Area (in Kanals)** and **Location**. The data was obtained by scraping Zameen.com. Rather than using high-level machine learning libraries like `scikit-learn` for the model, the linear regression algorithm, gradient descent optimizer, and evaluation metrics were implemented **manually** using fundamental mathematical formulas.

---

## ### Repository Structure

* `zameen.csv`: The scraped dataset containing raw real estate information.
* `notebook.ipynb`: The primary Jupyter Notebook containing data preprocessing, model building, and evaluation.
* `README.md`: Project documentation.

---

## ## Dataset & Preprocessing

The raw data consists of features like Area, Baths, Bedrooms, Price, and Location. The notebook performs the following data preparation steps:

1. **Data Cleaning:** Rows with missing or invalid values for bathrooms and bedrooms are removed, and columns are cast to appropriate numerical types.


2. **Price Normalization:** The `Price` column contains values in text format (e.g., "PKR15 Crore", "PKR11.9 Crore"). These are parsed and converted into a standard numerical scale representing **Crores (10 Million PKR)**.


3. **Area Unit Standardization:** Properties listed in "Marla" are converted to "Kanal" (1 Kanal = 20 Marlas) so that all area values are uniform.


4. **Categorical Encoding:** Popular sectors and locations across Islamabad (like DHA, Bahria Town, G-9, F-11, etc.) are mapped to discrete numerical codes.


5. **Feature Scaling:** Feature normalization (Standardization) is applied to both the Area and Location features using the formula:

$$x' = \frac{x - \mu}{\sigma}$$



where $\mu$ is the mean and $\sigma$ is the standard deviation.



---

## ## Model Architecture

The regression model is built entirely on **NumPy** using explicit mathematical formulations.

### Gradient Descent Implementation

The model trains using a batch gradient descent algorithm defined as:

```python
def alg(x, y, coeff, epoches, learning_rate):
    past_cost = []
    past_coeff = [coeff]
    for i in range(epoches):
        pred = np.dot(x, coeff)
        error = pred - y
        cost = 1 / (2 * N) * np.dot(error.T, error)
        past_cost.append(cost)
        der = 1 / N * learning_rate * np.dot(x.T, error)
        coeff = coeff - der
        past_coeff.append(coeff)
    return past_coeff, past_cost

```

### Hyperparameters

* **Learning Rate ($\alpha$):** 0.02


* **Epochs:** 2000


* **Train/Test Split:** 80% Training, 20% Testing



---

## ## Results and Performance

After training for 2000 epochs, the model converges with the following final coefficients:

* **Final Coeff values ($w_1, w_2, b$):** `[16.32947635, 0.71297414, 12.60146324]`


### Evaluation Metrics

The model is evaluated using Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and the R-squared ($R^2$) score:

| Metric | Value |
| --- | --- |
| **Training MSE** | 79.746

 |
| **Test RMSE** | 8.117

 |
| **R-2 Score** | 0.533

 |

An $R^2$ score of approximately **0.53** implies that about 53% of the variation in Islamabad property prices can be explained by the property's scaled area and location code alone.

---

## ## Requirements & Installation

To run this notebook locally, make sure you have the following Python libraries installed:

```bash
pip install pandas numpy matplotlib

```

1. Clone this repository.
2. Ensure `zameen.csv` is in the same directory as the notebook.


3. Open `notebook.ipynb` in Jupyter Notebook or VS Code and run all cells.
