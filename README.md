# rvflnet

Implements nonlinear glmnet; Random Vector Functional Link (RVFL) networks
using glmnet for elastic-net regularized output layer training.
Supports Gaussian, uniform, and Sobol random projections with 
various activation functions.

## Install 

```R
remotes::install_github("thierrymoudiki/rvflnet")
```

## Example

Also read the [vignettes](https://github.com/thierrymoudiki/rvflnet/tree/main/vignettes). 

```R
library(rvflnet)
library(glmnet)
library(MASS)

set.seed(123)

# -------------------------
# Data
# -------------------------

X <- as.matrix(MASS::Boston[, -14])  # predictors
y <- MASS::Boston$medv              # response

# Train/test split
n <- nrow(X)
idx <- sample(1:n, size = round(0.8 * n))

X_train <- X[idx, ]
y_train <- y[idx]

X_test <- X[-idx, ]
y_test <- y[-idx]

# -------------------------
# Fit model (No CV)
# -------------------------

fit <- rvflnet(X_train, y_train, 
               n_hidden = 200,
               activation = "sigmoid",
               W_type = "gaussian")
plot(fit)
print(fit)
head(coef(fit, s = c(0.1, 0.05, 0.01)))
tail(coef(fit, s = c(0.1, 0.05, 0.01)))
preds <- predict(fit, newx = X_test, s = c(0.05, 0.03, 0.01))
print(head(preds))
sqrt(colMeans((preds - y_test)**2))
```
