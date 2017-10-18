set.seed(42)
N <- 400
x <- runif(N, -1, 1)

Omega <- rbind( # correlation matrix
  c(1, 0.9),
  c(0.9, 1)
)
sigma <- c(0.6, 0.4) # residual SDs
Sigma <- diag(sigma) %*% Omega %*% diag(sigma) # covariance matrix
Sigma
errors <- mvtnorm::rmvnorm(N, c(0, 0), Sigma)
plot(errors)
cor(errors) # realized correlation

y1 <- -0.5 + x * 1.1 + errors[,1]
y2 <- 0.8 + x * 0.4 + errors[,2]
plot(x, y1)
plot(x, y2)
plot(y1, y2)

library(rstan)
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())
m <- stan("multi.stan", 
  data = list(J = 2, K = 2, N = length(y1), 
    x = matrix(c(rep(1, N), x), ncol = 2), 
    y = matrix(c(y1, y2), ncol = 2)),
  iter = 300)
print(m)
plot(m)

# beta[1,1] is the first intercept
# beta[1,2] is the first slope
# beta[2,1] is the second intercept
# beta[2,2] is the second slope
# L_sigma are the residual SDs
# Omega are the elements of the correlation matrix
# Sigma are the elements of the covariance matrix
