# Real Estate Investment Advisor

A data science project that classifies whether a property is a **good investment** and estimates its **price after 5 years**, delivered through an interactive Streamlit app.

## 1. Objective

- **Classification:** predict `Good_Investment` (1 = good, 0 = not good).
- **Regression:** predict `Future_Price_Lakhs`, the estimated price after 5 years.
- **App:** let a user enter property details, get both predictions with a confidence score, and explore the market.

## 2. Dataset

- 250,000 properties, 23 original columns (state, city, locality, property type, BHK, size, price, price per sq ft, year built, floors, nearby schools/hospitals, transport, parking, security, amenities, facing, owner type, availability).
- No missing values and no duplicate rows.
- The dataset has **no crime-rate or RERA column**, so infrastructure score and availability status are used as proxies.

## 3. Project structure

```
Real_estate_advisor
├── data
│   ├── raw/india_housing_prices.csv
│   └── processed/housing_clean.csv
├── notebooks
│   ├── 01_preprocessing.ipynb
│   ├── 02_eda.ipynb
│   └── 03_modeling.ipynb
├── models
│   ├── investment_classifier.joblib
│   ├── price_regressor.joblib
│   └── feature_list.joblib
├── app.py
├── requirements.txt
└── README.md
```

## 4. Data preprocessing

| Step | What was done |
|---|---|
| Quality checks | No missing values or duplicates found; `Age_of_Property` and `Price_per_SqFt` verified as consistent with other columns |
| Price per sq ft | Original column was rounded and in lakhs per sq ft; recomputed in rupees (`Price_per_SqFt_Rs`) |
| Invalid floors | 116,304 rows (about 47%) had `Floor_No` greater than `Total_Floors`; flagged in `Floor_Invalid` and capped at `Total_Floors` instead of dropping half the data |
| Outliers | IQR method: none in size or price; about 8% in price per sq ft. Inspected and **kept**, because they are valid ratios of independent price and size values, not errors |
| Encoding | Yes/No columns mapped to 1/0; categorical columns one-hot encoded inside a scikit-learn `Pipeline`; numeric columns scaled |
| Engineered features | `Amenity_Count`, `Infrastructure_Score`, `PPSF_vs_City`, `Floor_Invalid`, `Growth_Rate`, `Future_Price_Lakhs`, `Investment_Score`, `Good_Investment` |

**Infrastructure score (0 to 10):** the average of nearby schools (scaled 0 to 1), nearby hospitals (scaled 0 to 1), and transport accessibility (Low/Medium/High mapped to 1/2/3, scaled 0 to 1), multiplied by 10.

## 5. Assumptions (important)

The dataset contains no future prices and no investment outcomes, so both targets are defined by documented rules.

**Growth rate (annual):**

`Growth_Rate = 6% base + 0.4% per infrastructure-score point above 5.9 + 0.5% if Ready_to_Move + valuation adjustment (up to ±0.5%, cheaper-than-city-median properties grow faster) + random noise (std 0.5%, fixed seed)`, clipped to 2% to 12%.

`Future_Price_Lakhs = Price_in_Lakhs × (1 + Growth_Rate)^5`

**Good Investment label:** a property is good if at least 3 of these 4 conditions hold:
1. Growth rate is at or above the median growth rate
2. Price per sq ft is at or below the city median
3. BHK is 3 or more
4. Availability is Ready_to_Move

Class split: about 62% not good, 38% good.

## 6. Exploratory data analysis (key findings)

- **The dataset appears synthetic.** Price and size are uniformly distributed, and almost all numeric features are uncorrelated (|r| close to 0). Size vs. price correlation is -0.003.
- Average price per sq ft differs by under 3% across states, cities, property types, furnishing, facing, parking, and amenity counts, so none of these shows a meaningful price effect.
- BHK shares are about 20% in every city, and owner types and availability are evenly split.
- **Locality labels (`Locality_1` to `Locality_500`) repeat across all 42 cities**, so they do not identify real places. The column was excluded from the models, and locality-level trends are statistical noise.
- Price per sq ft is right-skewed because price and size are independent (a small, expensive property gives a very high ratio).
- The only visible correlations are mechanical: price per sq ft with price and size, and infrastructure score with schools and hospitals.
- Transport accessibility is linked to the good-investment rate only because infrastructure feeds the growth rate in the label, not as an independent market finding.

## 7. Modeling

- 80/20 train/test split (stratified on the classification label, random seed 42).
- 20 input features. Excluded: `Locality`, `Amenities` (replaced by `Amenity_Count`), `Year_Built` (duplicates age), and `Growth_Rate`, `Future_Price_Lakhs`, `Investment_Score` (derived from the targets).
- All preprocessing sits inside a scikit-learn `Pipeline`, so the app applies exactly the same transformations.

### Classification results (test set, 50,000 rows)

| Model | Accuracy | F1 |
|---|---|---|
| Logistic Regression | 86.4% | 0.82 |
| Random Forest | 90.6% | 0.87 |
| **XGBoost (selected)** | **92.4%** | **0.90** |

XGBoost confusion matrix: 29,230 true negatives, 1,926 false positives, 1,866 false negatives, 16,978 true positives.

### Regression results (5-year future price)

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Linear Regression | 11.52 | 8.77 | 0.9962 |
| Random Forest | 11.04 | 7.93 | 0.9965 |
| **XGBoost (selected)** | **9.36** | **6.57** | **0.9975** |

RMSE and MAE are in lakhs. **How to read these numbers:** the app takes the current price as an input, and future price is the current price scaled by a modest growth factor, so even a plain Linear Regression reaches R² = 0.996. The very high R² reflects this structure and should not be read as strong market-forecasting skill. XGBoost's gain over Linear Regression (RMSE 9.36 vs 11.52) comes from capturing how the growth adjustments depend on the property's features.

Growth-rate prediction check (XGBoost): R² = 0.72, MAE = 0.004 (0.4 percentage points). This is close to the ceiling set by the random noise in the growth assumption.

### Feature importance

The classifier relies mainly on price per sq ft, availability status, BHK, price, and size, which are the ingredients of the label rule. Age, floor numbers, and amenity count contribute almost nothing.

## 8. Streamlit app

- **Predict tab:** enter property details to get a good/not-good verdict, a confidence score, and the estimated 5-year price.
- **Explore market tab:** filter by city, price range, and BHK; view price distribution, top cities by price per sq ft, and good-investment share by property type.
- **Model insights tab:** top feature importances of the classifier.

## 9. How to run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Run the command from the project root, where `app.py`, `models/`, and `data/` are located. The notebooks run in order (01, 02, 03); notebook 03 saves the models that the app loads.

## 10. Limitations

- The data looks synthetic, with independent features, so no model can learn real market relationships from it.
- `Good_Investment` is a rule-based label, so high classifier accuracy shows that the model reproduces the defined rule. It is not evidence of predicting real-world investment outcomes. The remaining error comes from the random noise in the growth rate.
- Future prices are produced by an assumed growth model, not observed market data, and should not be used as financial advice.
- Crime rate and RERA information were not available in the dataset.

## 11. Tech stack

Python, pandas, NumPy, scikit-learn, XGBoost, Plotly, Matplotlib, Seaborn, Streamlit, joblib.
