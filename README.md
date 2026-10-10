# Linear Regression from Scratch
A mini project where I built linear regression from scratch: the cost function, gradients and gradient descent are written by hand with NumPy, without any ML library. I used pandas and seaborn to load the data, matplotlib for the plots, and scikit-learn only at the end to check my result. I made this while learning the basics of machine learning.

## Dataset

The `tips` dataset from seaborn (244 rows).

- Input (X): `total_bill`
- Output (y): `tip`

The goal is to predict the tip from the bill amount using a straight line: `tip = w * total_bill + b`

## What I built

- `compute_cost`: calculates the mean squared error cost for a given `w` and `b`
- `compute_gradient`: calculates the gradients `dj_dw` and `dj_db` (loop version and vectorized version)
- `gradient_descent`: repeatedly updates `w` and `b` to reduce the cost, and stores the cost at every iteration
- Plots: scatter plot of the data, cost vs iteration curve, and the fitted line on top of the data

## Result

| | w | b |
|---|---|---|
| My gradient descent | 0.10502451 | 0.92026956 |
| scikit-learn | 0.10502452 | 0.92026961 |

My values match scikit-learn up to about 7 decimal places.

## What I learned

- How cost, gradient and gradient descent work together
- The difference between a loop version and a vectorized version, and how broadcasting works in NumPy
- A learning rate that is too big (0.01 here) makes the cost blow up to `nan`
- A cost curve that goes down and becomes flat shows that training has converged

## How to run

```
pip install numpy pandas matplotlib seaborn scikit-learn
```

Then open `linear_regression.ipynb` and run all cells.