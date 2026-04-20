# =========================
# RVFL FEATURE ENGINEERING
# =========================

#' Generate RVFL random features
#'
#' Internal helper to build the RVFL design matrix.
#'
#' @param x numeric matrix of predictors
#' @param W random projection matrix
#' @param center centering vector
#' @param scale_vec scaling vector
#' @param activation activation function name or custom function
#'
#' @return augmented design matrix cbind(x, hidden features)
.rvfl_features <- function(x, W, center, scale_vec, activation) {

  x <- as.matrix(x)
  x_scaled <- scale(x, center = center, scale = scale_vec)

  # Handle activation function
  if (is.function(activation)) {
    g <- activation
  } else {
    g <- switch(activation,
                tanh = function(z) tanh(z),
                relu = function(z) z*(z>0),
                sigmoid = function(z) 1 / (1 + exp(-z)),
                identity = function(z) z,
                stop("Unknown activation function. Use: 'tanh', 'relu', 'sigmoid', 'identity', or a custom function")
    )
  }

  H <- g(x_scaled %*% W)
  #misc::debug_print(dim(x))
  #misc::debug_print(dim(x_scaled))
  #misc::debug_print(dim(W))
  #misc::debug_print(dim(H))
  Z <- cbind(x, H)
  colnames_x <- colnames(x)
  if (!is.null(colnames_x))
  {
    colnames(Z) <- c(colnames_x,
                     paste0("H", seq_len(ncol(W))))
  } else {
    colnames(Z) <- c(paste0("X", seq_len(ncol(x))),
                     paste0("H", seq_len(ncol(W))))
  }
  Z
}

#' Generate random weights for RVFL
#'
#' Internal helper for generating random projection matrix
#'
#' @param p number of input features
#' @param n_hidden number of hidden units
#' @param W_type type of random weights ("gaussian", "sobol", "uniform")
#' @param seed random seed for reproducibility
#'
#' @return random weight matrix of dimension p x n_hidden
.generate_random_weights <- function(p, n_hidden, W_type, seed) {

  if (n_hidden <= 0) {
    stop("n_hidden must be positive")
  }

  set.seed(seed)

  W <- switch(W_type,
              gaussian = matrix(rnorm(p * n_hidden), p, n_hidden),
              uniform = matrix(runif(p * n_hidden, -1, 1), p, n_hidden),
              sobol = {
                if (!requireNamespace("randtoolbox", quietly = TRUE)) {
                  stop("Package 'randtoolbox' required for Sobol features. Install with: install.packages('randtoolbox')")
                }
                # Generate n_hidden Sobol points in p dimensions
                # Returns: n_hidden x p matrix (each row is a point in p-dimensional hypercube)
                # Transpose to p x n_hidden
                # This preserves the low-discrepancy property across dimensions
                W <- t(randtoolbox::sobol(n = n_hidden, dim = p)) #t(qnorm(sobol_points))

                # Verify dimensions: p x n_hidden
                if (nrow(W) != p || ncol(W) != n_hidden) {
                  stop(sprintf("Sobol weight dimension error: expected %dx%d, got %dx%d",
                               p, n_hidden, nrow(W), ncol(W)))
                }
                W
              },
              stop("W_type must be 'gaussian', 'uniform', or 'sobol'")
  )

  return(W)
}

#' Compute scaling parameters from data
#'
#' @param x input matrix
#' @param scale_input logical, whether to scale
#'
#' @return list with center and scale vectors
.compute_scaling <- function(x, scale_input = TRUE) {
  x <- as.matrix(x)
  p <- ncol(x)

  if (scale_input) {
    center <- colMeans(x)
    scale_vec <- apply(x, 2, sd)
    scale_vec[scale_vec == 0] <- 1
  } else {
    center <- rep(0, p)
    scale_vec <- rep(1, p)
  }

  list(center = center, scale_vec = scale_vec)
}

#' Apply scaling to new data
#'
#' @param x input matrix
#' @param center centering vector
#' @param scale_vec scaling vector
#'
#' @return scaled matrix
.apply_scaling <- function(x, center, scale_vec) {
  scale(x, center = center, scale = scale_vec)
}

#' Prepare RVFL design matrix
#'
#' Internal helper for consistent preprocessing
#'
#' @param x input matrix
#' @param W random weight matrix
#' @param center centering vector (if NULL, compute from data)
#' @param scale_vec scaling vector (if NULL, compute from data)
#' @param scale_input logical, whether to scale inputs
#' @param activation activation function
#' @param include_original logical, whether to include original features
#'
#' @return list with Z (design matrix) and preprocessing parameters
.prepare_rvfl_data <- function(x, W, center = NULL, scale_vec = NULL,
                               scale_input = TRUE, activation,
                               include_original = TRUE) {

  x <- as.matrix(x)
  p <- ncol(x)

  # Compute or use provided scaling parameters
  if (is.null(center) || is.null(scale_vec)) {
    scaling <- .compute_scaling(x, scale_input)
    center <- scaling$center
    scale_vec <- scaling$scale_vec
  }

  Z <- .rvfl_features(x, W, center, scale_vec, activation)

  # Optionally remove original features
  if (!include_original) {
    Z <- Z[, (p+1):ncol(Z), drop = FALSE]
  }

  list(
    Z = Z,
    center = center,
    scale_vec = scale_vec,
    scaled_input = scale_input
  )
}


# =========================
# TRAINING FUNCTION
# =========================

#' Random Vector Functional Link Network (glmnet backend)
#'
#' Fits an RVFL model using glmnet on augmented random feature space.
#'
#' @param x design matrix
#' @param y response vector
#' @param n_hidden number of random hidden units (must be > 0)
#' @param activation activation function ("tanh", "relu", "sigmoid", "identity", or custom function)
#' @param W_type type of random weights ("gaussian", "uniform", "sobol")
#' @param seed random seed for reproducibility
#' @param scale logical, whether to standardize inputs
#' @param include_original logical, whether to include original features (default TRUE)
#' @param store_y logical, whether to store y in model (for residuals, default FALSE)
#' @param family response type ("gaussian", "binomial", "poisson", "multinomial", "cox", "mgaussian").
#'        For family = "cox", y must be a Surv object from the survival package.
#' @param ... additional arguments passed to glmnet::glmnet
#'
#' @return object of class "rvflnet"
#'
#' @import glmnet
#' @export
#'
#' @examples
#' \dontrun{
#' x <- matrix(rnorm(100*5), 100, 5)
#' y <- x[,1] + sin(x[,2]) + rnorm(100, 0, 0.1)
#' model <- rvflnet(x, y, n_hidden = 50, activation = "tanh")
#' predict(model, newx = x[1:10,])
#' coef(model)  # Note: coefficients apply to [X | random features] space
#' }
rvflnet <- function(x, y,
                    n_hidden = 200L,
                    activation = c("sigmoid", "tanh", "relu", "identity"),
                    W_type = c("gaussian", "uniform", "sobol"),
                    seed = 1,
                    scale = TRUE,
                    include_original = TRUE,
                    store_y = FALSE,
                    family = c("gaussian", "binomial", "poisson", "multinomial", "cox", "mgaussian"),
                    ...) {

  activation <- match.arg(activation)
  W_type <- match.arg(W_type)
  family <- match.arg(family)

  x <- as.matrix(x)
  #y <- as.numeric(y)
  p <- ncol(x)

  # if (nrow(x) != length(y)) {
  #   stop("Number of rows in x must match length of y")
  # }

  if (family == "cox") {
    if (!inherits(y, "Surv")) {
      stop("For family='cox', y must be a Surv object created by survival::Surv()")
    }
  }

  # Generate random weights
  W <- .generate_random_weights(p, n_hidden, W_type, seed)

  # Prepare RVFL features
  prep <- .prepare_rvfl_data(x, W,
                             center = NULL,
                             scale_vec = NULL,
                             scale_input = scale,
                             activation = activation,
                             include_original = include_original)

  Z <- prep$Z

  # Fit glmnet
  fit <- glmnet::glmnet(Z, y, family = family, ...)

  structure(list(
    fit = fit,
    W = W,
    center = prep$center,
    scale_vec = prep$scale_vec,
    scaled_input = prep$scaled_input,
    activation = activation,
    W_type = W_type,
    seed = seed,
    n_hidden = n_hidden,
    include_original = include_original,
    p = p,
    family = family,
    y = if (store_y) y else NULL,
    call = match.call()
  ), class = "rvflnet")
}


# =========================
# CROSS-VALIDATION
# =========================

#' Cross-validated RVFLNet model
#'
#' Fits RVFL features then applies cv.glmnet.
#'
#' @param x design matrix
#' @param y response vector
#' @param n_hidden number of hidden units
#' @param activation activation function
#' @param W_type random feature type ("gaussian", "uniform", "sobol")
#' @param seed random seed
#' @param scale logical scaling
#' @param include_original logical, whether to include original features
#' @param store_y logical, whether to store y in model
#' @param family response type
#' @param ... additional arguments passed to glmnet::cv.glmnet
#'
#' @return object of class "cv.rvflnet"
#'
#' @import glmnet
#' @export
#'
#' @examples
#' \dontrun{
#' x <- matrix(rnorm(100*5), 100, 5)
#' y <- x[,1] + sin(x[,2]) + rnorm(100, 0, 0.1)
#' cv_model <- cv.rvflnet(x, y, n_hidden = 50, nfolds = 5)
#' plot(cv_model)
#' predict(cv_model, newx = x[1:10,])
#' }
cv.rvflnet <- function(x, y,
                       n_hidden = 200L,
                       activation = c("sigmoid", "tanh", "relu", "identity"),
                       W_type = c("gaussian", "uniform", "sobol"),
                       seed = 1,
                       scale = TRUE,
                       include_original = TRUE,
                       store_y = FALSE,
                       family = c("gaussian", "binomial", "poisson", "multinomial", "cox", "mgaussian"),
                       ...) {

  activation <- match.arg(activation)
  W_type <- match.arg(W_type)
  family <- match.arg(family)

  x <- as.matrix(x)
  #y <- as.numeric(y)
  p <- ncol(x)

  # if (nrow(x) != length(y)) {
  #   stop("Number of rows in x must match length of y")
  # }

  if (family == "cox") {
    if (!inherits(y, "Surv")) {
      stop("For family='cox', y must be a Surv object created by survival::Surv()")
    }
  }

  # Generate random weights
  W <- .generate_random_weights(p, n_hidden, W_type, seed)

  # Prepare RVFL features
  #misc::debug_print(dim(x))
  #misc::debug_print(dim(W))
  prep <- .prepare_rvfl_data(x, W,
                             center = NULL,
                             scale_vec = NULL,
                             scale_input = scale,
                             activation = activation,
                             include_original = include_original)

  Z <- prep$Z

  # Cross-validate glmnet
  cvfit <- glmnet::cv.glmnet(Z, y, family = family, ...)

  structure(list(
    cvfit = cvfit,
    W = W,
    center = prep$center,
    scale_vec = prep$scale_vec,
    scaled_input = prep$scaled_input,
    activation = activation,
    W_type = W_type,
    seed = seed,
    n_hidden = n_hidden,
    include_original = include_original,
    p = p,
    family = family,
    y = if (store_y) y else NULL,
    call = match.call()
  ), class = "cv.rvflnet")
}


# =========================
# PREDICT METHODS
# =========================

#' Predict method for RVFLNet
#'
#' @param object rvflnet model
#' @param newx new data matrix (if NULL, returns fitted values if available)
#' @param s lambda value (passed to glmnet)
#' @param type type of prediction ("response", "coefficients", etc.)
#' @param ... additional arguments passed to predict.glmnet
#'
#' @return predictions
#' @export
#'
#' @examples
#' \dontrun{
#' predictions <- predict(model, newx = x_test, s = "lambda.min")
#' }
predict.rvflnet <- function(object, newx = NULL, s = NULL, type = "response", ...) {

  # Handle NULL newx (return fitted values if available)
  if (is.null(newx)) {
    if (!is.null(object$y)) {
      # Compute fitted values from model using training data
      # This requires the training data - ideally stored
      warning("newx = NULL requires training data to be stored. Returning NULL.")
      return(NULL)
    } else {
      stop("newx = NULL but no training data stored. Provide newx or refit with store_y = TRUE")
    }
  }

  newx <- as.matrix(newx)

  # Check dimensions
  if (ncol(newx) != object$p) {
    stop(sprintf("newx must have %d columns (got %d)", object$p, ncol(newx)))
  }

  # Prepare RVFL features
  #misc::debug_print(dim(newx))
  #misc::debug_print(dim(object$W))
  prep <- .prepare_rvfl_data(newx, object$W,
                             center = object$center,
                             scale_vec = object$scale_vec,
                             scale_input = object$scaled_input,
                             activation = object$activation,
                             include_original = object$include_original)

  Z <- prep$Z

  predict(object$fit, newx = Z, s = s, type = type, ...)
}

#' Predict method for cross-validated RVFLNet
#'
#' @param object cv.rvflnet model
#' @param newx new data matrix (if NULL, returns fitted values if available)
#' @param s lambda value ("lambda.min", "lambda.1se", or numeric)
#' @param type type of prediction ("response", "coefficients", etc.)
#' @param ... additional arguments passed to predict.glmnet
#'
#' @return predictions
#' @export
#'
#' @examples
#' \dontrun{
#' predictions <- predict(cv_model, newx = x_test, s = "lambda.min")
#' }
predict.cv.rvflnet <- function(object, newx = NULL, s = "lambda.min", type = "response", ...) {

  # Handle NULL newx (return fitted values if available)
  if (is.null(newx)) {
    if (!is.null(object$y)) {
      warning("newx = NULL but fitted values not directly available. Use predict on training data.")
      return(NULL)
    } else {
      stop("newx = NULL but no training data stored. Provide newx or refit with store_y = TRUE")
    }
  }

  newx <- as.matrix(newx)

  # Check dimensions
  if (ncol(newx) != object$p) {
    stop(sprintf("newx must have %d columns (got %d)", object$p, ncol(newx)))
  }

  # Prepare RVFL features
  prep <- .prepare_rvfl_data(newx, object$W,
                             center = object$center,
                             scale_vec = object$scale_vec,
                             scale_input = object$scaled_input,
                             activation = object$activation,
                             include_original = object$include_original)

  Z <- prep$Z

  predict(object$cvfit, newx = Z, s = s, type = type, ...)
}


# =========================
# COEF METHODS
# =========================

#' Extract coefficients from RVFLNet model
#'
#' @param object rvflnet object
#' @param s lambda value (passed to glmnet)
#' @param ... additional arguments passed to coef.glmnet
#'
#' @return coefficient matrix. NOTE: Coefficients apply to the augmented feature space
#'         [original features | random hidden features]. The first p coefficients
#'         correspond to original inputs, remaining n_hidden to random features.
#'         These are NOT directly interpretable as "feature importance" in the
#'         original input space due to the nonlinear random projection.
#' @export
#'
#' @examples
#' \dontrun{
#' coef(model, s = "lambda.min")
#' }
coef.rvflnet <- function(object, s = NULL, ...) {
  coef(object$fit, s = s, ...)
}

#' Extract coefficients from cross-validated RVFLNet model
#'
#' @param object cv.rvflnet object
#' @param s lambda value ("lambda.min", "lambda.1se", or numeric)
#' @param ... additional arguments passed to coef.glmnet
#'
#' @return coefficient matrix. See coef.rvflnet for interpretation notes.
#' @export
#'
#' @examples
#' \dontrun{
#' coef(cv_model, s = "lambda.min")
#' }
coef.cv.rvflnet <- function(object, s = "lambda.min", ...) {
  coef(object$cvfit, s = s, ...)
}


# =========================
# FITTED METHODS
# =========================

#' Extract fitted values from RVFLNet model
#'
#' @param object rvflnet object
#' @param s lambda value
#' @param ... additional arguments
#'
#' @return fitted values
#' @export
fitted.rvflnet <- function(object, s = NULL, ...) {
  if (!is.null(object$y)) {
    # Compute fitted values from model
    predict(object, newx = NULL, s = s, ...)
  } else {
    stop("Fitted values require training data. Refit with store_y = TRUE")
  }
}

#' Extract fitted values from cross-validated RVFLNet model
#'
#' @param object cv.rvflnet object
#' @param s lambda value
#' @param ... additional arguments
#'
#' @return fitted values
#' @export
fitted.cv.rvflnet <- function(object, s = "lambda.min", ...) {
  if (!is.null(object$y)) {
    predict(object, newx = NULL, s = s, ...)
  } else {
    stop("Fitted values require training data. Refit with store_y = TRUE")
  }
}


# =========================
# RESIDUALS METHODS
# =========================

#' Residuals method for RVFLNet
#'
#' @param object rvflnet object
#' @param s lambda value
#' @param type type of residuals ("response", "deviance", etc.)
#' @param ... additional arguments
#'
#' @return residuals vector
#' @export
#'
#' @examples
#' \dontrun{
#' model <- rvflnet(x, y, store_y = TRUE)
#' residuals(model, s = "lambda.min")
#' }
residuals.rvflnet <- function(object, s = NULL, type = "response", ...) {
  if (is.null(object$y)) {
    stop("Residuals require stored y values. Refit with store_y = TRUE")
  }

  fitted_vals <- fitted(object, s = s)
  if (is.null(fitted_vals)) {
    stop("Could not compute fitted values")
  }

  residuals <- object$y - fitted_vals

  # For GLM families, transform residuals based on type
  if (object$family != "gaussian") {
    switch(type,
           response = residuals,
           deviance = {
             warning("Deviance residuals not yet implemented for non-Gaussian families")
             residuals
           },
           residuals
    )
  } else {
    residuals
  }
}

#' Residuals method for cross-validated RVFLNet
#'
#' @param object cv.rvflnet object
#' @param s lambda value ("lambda.min", "lambda.1se", or numeric)
#' @param type type of residuals
#' @param ... additional arguments
#'
#' @return residuals vector
#' @export
residuals.cv.rvflnet <- function(object, s = "lambda.min", type = "response", ...) {

  if (object$family == "cox") {
    stop("Residuals not defined for Cox regression. Use martingale or deviance residuals from the underlying coxph model if needed.")
  }

  if (is.null(object$y)) {
    stop("Residuals require stored y values. Refit with store_y = TRUE")
  }

  fitted_vals <- fitted(object, s = s)
  if (is.null(fitted_vals)) {
    stop("Could not compute fitted values")
  }

  residuals <- object$y - fitted_vals

  if (object$family != "gaussian") {
    warning("Deviance residuals not yet fully implemented for non-Gaussian families")
  }

  residuals
}


# =========================
# PRINT METHODS
# =========================

#' Print RVFLNet model
#'
#' @param x rvflnet object
#' @param ... ignored
#' @export
print.rvflnet <- function(x, ...) {
  cat("\n========================================\n")
  cat("RVFLNet Model (glmnet backend)\n")
  cat("========================================\n")
  cat(sprintf("Call: %s\n", paste(deparse(x$call), collapse = "")))
  cat(sprintf("Input features: %d\n", x$p))
  cat(sprintf("Hidden units: %d\n", x$n_hidden))
  cat(sprintf("Total features: %d\n", ifelse(x$include_original, x$p + x$n_hidden, x$n_hidden)))
  cat(sprintf("Activation: %s\n", x$activation))
  cat(sprintf("Weight distribution: %s\n", x$W_type))
  cat(sprintf("Seed: %d\n", x$seed))
  cat(sprintf("Input scaling: %s\n", ifelse(x$scaled_input, "Yes", "No")))
  cat(sprintf("Include original features: %s\n", x$include_original))
  cat(sprintf("Family: %s\n", x$family))

  # Show number of non-zero coefficients at default lambda
  if (!is.null(x$fit$lambda)) {
    coef_mat <- coef(x, s = min(x$fit$lambda))
    n_nonzero <- sum(coef_mat[-1, 1] != 0)
    cat(sprintf("Non-zero coefficients (min lambda): %d\n", n_nonzero))
  }
  cat("========================================\n")
  invisible(x)
}

#' Print cross-validated RVFLNet model
#'
#' @param x cv.rvflnet object
#' @param ... ignored
#' @export
print.cv.rvflnet <- function(x, ...) {
  cat("\n========================================\n")
  cat("Cross-Validated RVFLNet Model\n")
  cat("========================================\n")
  cat(sprintf("Call: %s\n", paste(deparse(x$call), collapse = "")))
  cat(sprintf("Input features: %d\n", x$p))
  cat(sprintf("Hidden units: %d\n", x$n_hidden))
  cat(sprintf("Total features: %d\n", ifelse(x$include_original, x$p + x$n_hidden, x$n_hidden)))
  cat(sprintf("Activation: %s\n", x$activation))
  cat(sprintf("Weight distribution: %s\n", x$W_type))
  cat(sprintf("Seed: %d\n", x$seed))
  cat(sprintf("Input scaling: %s\n", ifelse(x$scaled_input, "Yes", "No")))
  cat(sprintf("Include original features: %s\n", x$include_original))
  cat(sprintf("Family: %s\n", x$family))

  if (x$family == "cox") {
    cat("\nCox Regression Notes:\n")
    cat("--------------------\n")
    cat("Predictions return linear predictor (lp)\n")
    cat("Use type = 'response' to get hazard ratios (exp(lp))\n")
    cat("Evaluate using concordance index (C-index), not RMSE\n")
  }

  cat("\nCross-validation summary:\n")
  cat("------------------------\n")
  cat(sprintf("lambda.min: %.4f (min CV error)\n", x$cvfit$lambda.min))
  cat(sprintf("lambda.1se: %.4f (1se rule)\n", x$cvfit$lambda.1se))

  # Get optimal model size
  coef_min <- coef(x, s = "lambda.min")
  coef_1se <- coef(x, s = "lambda.1se")
  cat(sprintf("Non-zero coefficients at lambda.min: %d\n", sum(coef_min[-1, 1] != 0)))
  cat(sprintf("Non-zero coefficients at lambda.1se: %d\n", sum(coef_1se[-1, 1] != 0)))
  cat("========================================\n")
  invisible(x)
}


# =========================
# SUMMARY METHODS
# =========================

#' Summary method for RVFLNet
#'
#' @param object rvflnet object
#' @param s lambda value
#' @param ... additional arguments
#'
#' @return invisible summary object
#' @export
summary.rvflnet <- function(object, s = NULL, ...) {
  cat("\n========================================\n")
  cat("RVFLNet Model Summary\n")
  cat("========================================\n")
  print(object)

  # Extract coefficients
  coef_obj <- coef(object, s = s)
  cat("\nCoefficient summary:\n")
  cat("-------------------\n")
  cat(sprintf("Intercept: %.4f\n", coef_obj[1, 1]))

  # Separate original and hidden features
  if (object$include_original) {
    orig_coef <- coef_obj[2:(object$p+1), 1]
    hidden_coef <- coef_obj[(object$p+2):length(coef_obj), 1]
    cat(sprintf("Original features non-zero: %d / %d\n", sum(orig_coef != 0), length(orig_coef)))
    cat(sprintf("Hidden features non-zero: %d / %d\n", sum(hidden_coef != 0), length(hidden_coef)))
  } else {
    hidden_coef <- coef_obj[-1, 1]
    cat(sprintf("Hidden features non-zero: %d / %d\n", sum(hidden_coef != 0), length(hidden_coef)))
  }

  invisible(object)
}

#' Summary method for cross-validated RVFLNet
#'
#' @param object cv.rvflnet object
#' @param s lambda value ("lambda.min" or "lambda.1se")
#' @param ... additional arguments
#'
#' @return invisible summary object
#' @export
summary.cv.rvflnet <- function(object, s = "lambda.min", ...) {
  cat("\n========================================\n")
  cat("Cross-Validated RVFLNet Model Summary\n")
  cat("========================================\n")
  print(object)

  # Extract cross-validation results
  cat("\nCross-validation performance:\n")
  cat("----------------------------\n")
  cv_stats <- data.frame(
    Lambda = c(object$cvfit$lambda.min, object$cvfit$lambda.1se),
    CV_Error = c(min(object$cvfit$cvm),
                 object$cvfit$cvm[object$cvfit$lambda == object$cvfit$lambda.1se]),
    CV_SD = c(object$cvfit$cvsd[which.min(object$cvfit$cvm)],
              object$cvfit$cvsd[object$cvfit$lambda == object$cvfit$lambda.1se])
  )
  rownames(cv_stats) <- c("lambda.min", "lambda.1se")
  print(cv_stats)

  # Show coefficients at chosen lambda
  coef_obj <- coef(object, s = s)
  cat(sprintf("\nCoefficient summary at %s:\n", s))
  cat("-------------------------------\n")
  cat(sprintf("Intercept: %.4f\n", coef_obj[1, 1]))

  # Separate original and hidden features
  if (object$include_original) {
    orig_coef <- coef_obj[2:(object$p+1), 1]
    hidden_coef <- coef_obj[(object$p+2):length(coef_obj), 1]
    cat(sprintf("Original features non-zero: %d / %d\n", sum(orig_coef != 0), length(orig_coef)))
    cat(sprintf("Hidden features non-zero: %d / %d\n", sum(hidden_coef != 0), length(hidden_coef)))
  } else {
    hidden_coef <- coef_obj[-1, 1]
    cat(sprintf("Hidden features non-zero: %d / %d\n", sum(hidden_coef != 0), length(hidden_coef)))
  }

  invisible(object)
}


# =========================
# PLOT METHODS
# =========================

#' Plot CV curve for RVFLNet
#'
#' @param x cv.rvflnet object
#' @param ... passed to plot.glmnet
#'
#' @export
#'
#' @examples
#' \dontrun{
#' plot(cv_model)
#' }
plot.cv.rvflnet <- function(x, ...) {
  plot(x$cvfit, main = "RVFLNet Cross-Validation Curve", ...)
}

#' Plot RVFLNet model
#'
#' Generic plot method for rvflnet objects
#'
#' @param x rvflnet object
#' @param type type of plot ("coef" for coefficient path)
#' @param ... additional arguments passed to plot functions
#'
#' @export
plot.rvflnet <- function(x, type = c("coef"), ...) {
  type <- match.arg(type)

  if (type == "coef") {
    plot(x$fit, xvar = "lambda", label = TRUE,
         main = "RVFLNet Coefficient Path", ...)
  }
}


# =========================
# HELPER FUNCTIONS
# =========================

#' Check if object is an RVFLNet model
#'
#' @param x object to check
#'
#' @return logical
#' @export
is.rvflnet <- function(x) {
  inherits(x, "rvflnet")
}

#' Check if object is a CV-RVFLNet model
#'
#' @param x object to check
#'
#' @return logical
#' @export
is.cv.rvflnet <- function(x) {
  inherits(x, "cv.rvflnet")
}

#' Extract number of hidden units
#'
#' @param model rvflnet or cv.rvflnet object
#'
#' @return integer
#' @export
n_hidden <- function(model) {
  if (!is.rvflnet(model) && !is.cv.rvflnet(model)) {
    stop("Model must be an rvflnet or cv.rvflnet object")
  }
  model$n_hidden
}

#' Extract activation function
#'
#' @param model rvflnet or cv.rvflnet object
#'
#' @return character
#' @export
get_activation <- function(model) {
  if (!is.rvflnet(model) && !is.cv.rvflnet(model)) {
    stop("Model must be an rvflnet or cv.rvflnet object")
  }
  model$activation
}

#' Extract weight type
#'
#' @param model rvflnet or cv.rvflnet object
#'
#' @return character
#' @export
get_weight_type <- function(model) {
  if (!is.rvflnet(model) && !is.cv.rvflnet(model)) {
    stop("Model must be an rvflnet or cv.rvflnet object")
  }
  model$W_type
}

#' Extract best lambda from CV object
#'
#' @param model cv.rvflnet object
#' @param which which lambda to return ("min" or "1se")
#'
#' @return numeric lambda value
#' @export
best_lambda <- function(model, which = c("min", "1se")) {
  if (!is.cv.rvflnet(model)) {
    stop("Model must be a cv.rvflnet object")
  }
  which <- match.arg(which)
  if (which == "min") {
    model$cvfit$lambda.min
  } else {
    model$cvfit$lambda.1se
  }
}
