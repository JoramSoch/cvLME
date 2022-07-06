function LME = MGLM_LME(P, L0, O0, v0, Ln, On, vn)
% _
% Log Model Evidence of Multivariate GLM with Normal-Gamma Priors
% FORMAT LME = MGLM_LME(P, L0, O0, v0, Ln, On, vn)
% 
%     P   - an n x n precision matrix specifying correlations
%     L0  - a  p x p matrix (prior precision of regression coefficients)
%     O0  - a  v x v matrix (prior inverse scale matrix for covariance)
%     v0  - a  1 x 1 scalar (prior degrees of freedom for covariance)
%     Ln  - a  p x p matrix (posterior precision of regression coefficients)
%     On  - a  v x v matrix (posterior inverse scale matrix for covariance)
%     vn  - a  1 x 1 scalar (posterior degrees of freedom for covariance)
% 
%     LME - a  1 x v vector, thelog model evidences
% 
% FORMAT LME = MGLM_LME(P, L0, O0, v0, Ln, On, vn) returns the log model
% evidence for a multivariate general linear model with precision matrix P
% and normal-Wishart distributed priors/posteriors for regression
% coefficients (L0/Ln) and signal covariance (O0, v0 / On, vn).
% 
% References:
% [1] Wikipedia (2021): "Bayesian multivariate linear regression";
%     URL: https://en.wikipedia.org/wiki/Bayesian_multivariate_linear_regression#Posterior_distribution.
% [2] Soch J (2022): "Log model evidence for multivariate Bayesian linear regression";
%     URL: https://statproofbook.github.io/P/mblr-lme.
% 
% Author: Joram Soch, BCCN Berlin
% E-Mail: joram.soch@bccn-berlin.de
% Edited: 06/07/2022, 11:51


% Get model dimensions
%-------------------------------------------------------------------------%
n = size(P, 1);                 % number of observations
v = size(O0,1);                 % number of signals
p = size(Ln,1);                 % number of regressors

% Calculate log model evidence
%-------------------------------------------------------------------------%
LME = v/2*log(det(P))       - (n*v)/2*log(2*pi) + ...
      v/2*log(det(L0))      - v/2*log(det(Ln)) + ...
      v0/2*log(det(1/2*O0)) - vn/2*log(det(1/2*On)) + ...
      mgammaln(vn/2,v)      - mgammaln(v0/2,v);


% Function: logarithmized multivariate gamma function
%-------------------------------------------------------------------------%
function Y = mgammaln(X,p)
% Logarithm of Multivariate Gamma Function
% FORMAT Y = mgammaln(X,p)
% 
%     X - a scalar, vector or matrix of real values
%     p - an integer, the dimensionality of the function
% 
%     Y - the logarithm of the multivariate gamma function
% 
% FORMAT Y = mgammaln(X,p) computes the logarithm of the multivariate gamma
% function for argument X and order p [1]. This means that this function is
% the multivariate analogue to Y = gammaln(X).
% 
% Calling Y = mgammaln(X,p) is equivalent to calling log(mgamma(X,p)).
% However, this function avoids the overflow or underflow problems that
% are likely to occur when using the gamma function in non-log space.
% 
% References:
% [1] Wikipedia (2022): "Multivariate gamma function";
%     URL: https://en.wikipedia.org/wiki/Multivariate_gamma_function.

% compute offset
Y = p*(p-1)/4 * log(pi);

% compute sum
for j = 1:p
    Y = Y + gammaln(X + (1-j)/2);
end;