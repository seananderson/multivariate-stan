// based on 9.15 Multivariate Outcomes Stan manual section
data { 
  int<lower=1> K;
  int<lower=1> J;
  int<lower=0> N;
  vector[J] x[N];
  vector[K] y[N]; 
}
parameters { 
  matrix[K, J] beta; 
  cholesky_factor_corr[K] L_Omega;
  vector<lower=0>[K] L_sigma;
} 
model {
  vector[K] mu[N];
  for (n in 1:N) 
    mu[n] = beta * x[n];
  
  to_vector(beta) ~ normal(0, 2);
  L_Omega ~ lkj_corr_cholesky(1); 
  L_sigma ~ student_t(3, 0, 2);
  
  y ~ multi_normal_cholesky(mu, diag_pre_multiply(L_sigma, L_Omega));
}
generated quantities {
  matrix[K, K] Omega;
  matrix[K, K] Sigma;
  Omega = multiply_lower_tri_self_transpose(L_Omega);
  Sigma = quad_form_diag(Omega, L_sigma); 
}
