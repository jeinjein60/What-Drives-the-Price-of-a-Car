# What Drives the Price of a Used Car?

A regression analysis of 427,000 used car listings to identify the
strongest drivers of price, built for a used car dealership deciding how to
prioritize and price inventory.

**Full analysis notebook:** [UsedCarPrice.ipynb](./UsedCarPrice.ipynb)

## Summary of Findings

Vehicle **age** and **mileage** are by far the strongest drivers of used
car price. Together they account for the large majority of what the
best-performing model could explain. Engine size, drivetrain, 
and body style matter too, but at a noticeably smaller scale.

| Direction | Factor |
|---|---|
| Lowers price (strongest) | Vehicle age |
| Lowers price | Higher mileage |
| Lowers price | Gas-powered vs. some alternative fuel types |
| Lowers price | Front-wheel drive vs. other drivetrains |
| Raises price | Higher cylinder count |
| Raises price | Pickup trucks and truck body styles |

**Model performance:** A Random Forest regressor explained about **79%**
of the variation in log-price (R^2 = 0.793) on held-out test data,
outperforming both plain Linear Regression and a cross-validated Ridge
regression (R^2 ≈ 0.694–0.699). Ridge's coefficients were used as the
primary interpretability tool, since they translate directly into
"this feature moves price by $X" statements a dealer can act on.

**Reliability varies by price range.** Predictions are most accurate for
vehicles in the $25,000–$50,000 range and least accurate for the
cheapest vehicles (under $10,000) and the small number priced above
$50,000. This is worth keeping in mind when applying these findings to a
specific segment of inventory.

## Repository Contents

| File | Description |
|---|---|
| `UsedCarPrice.ipynb` | Full CRISP-DM analysis: business understanding, data cleaning, modeling, evaluation, and the client-facing findings report |
| `images/` | Supporting images referenced in the notebook |
| `vehicles.csv` | Raw Dataset |

*Note: the raw `vehicles.csv` dataset (~427K rows) is not included in this
repository due to file size — see the Data Understanding section of the
notebook for the source and a full data dictionary.*

## Methodology (CRISP-DM)

1. **Business Understanding**: reframe "what drives used car prices" as
   a supervised regression problem with `price` as the target.
2. **Data Understanding**: explore missingness, implausible outliers
   (ex: $0 listings, 1900 model years, million-mile odometers), and
   duplicate/near-empty listings.
3. **Data Preparation**: filter to a plausible price/year/odometer
   range, engineer `car_age` and a numeric `cylinders` feature, impute
   or drop remaining missing values, log-transform price, and one-hot
   encode categoricals.
4. **Modeling**: compare Linear Regression, cross-validated/grid-searched
   Ridge regression, and a Random Forest regressor.
5. **Evaluation**:  assess both predictive accuracy (R², RMSE, dollar-scale
   error) and interpretability, and check how error varies across price
   brackets.
6. **Deployment**: translate findings into a plain-language pricing
   guide for a nontechnical dealership audience, with monitoring
   recommendations for future refreshes.

## Tools

Python, pandas, NumPy, scikit-learn, matplotlib, seaborn
Google Colab
