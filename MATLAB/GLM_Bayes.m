function [mn, Ln, an, bn] = GLM_Bayes(Y, X, P, m0, L0, a0, b0)
% _
% Bayesian Estimation of General Linear Model with Normal-Gamma Priors
% FORMAT [mn, Ln, an, bn] = GLM_Bayes(Y, X, P, m0, L0, a0, b0)
% 
%     Y  - an n x v data matrix of measured signals
%     X  - an n x p design matrix of predictor variables
%     P  - an n x n precision matrix specifying correlations
%     m0 - a  p x v matrix (prior means of regression coefficients)
%     L0 - a  p x p matrix (prior precision of regression coefficients)
%     a0 - a  1 x 1 scalar (prior shape of residual precision)
%     b0 - a  1 x v vector (prior rates of residual precision)
% 
%     mn - a  p x v matrix (posterior means of regression coefficients)
%     Ln - a  p x p matrix (posterior precision of regression coefficients)
%     an - a  1 x 1 scalar (posterior shape of residual precision)
%     bn - a  1 x v vector (posterior rates of residual precision)
% 
% FORMAT [mn, Ln, an, bn] = GLM_Bayes(Y, X, P, m0, L0, a0, b0) returns
% the posterior parameter estimates for a general linear model with data Y,
% design matrix X, precision matrix P and normal-gamma distributed priors
% for regression coefficients (m0, L0) and residual precision (a0, b0).
% 
% References:
% [1] Bishop CM (2006): "Pattern Recognition and Machine Learning".
%     Springer, ch. 3.3, pp. 175-177.
% [2] Koch KR (2000): "Einführung in die Bayes-Statistik".
%     Springer, ch. 4.3.2, pp. 117-119.
% 
% Author: Joram Soch, BCCN Berlin
% E-Mail: joram.soch@bccn-berlin.de
% Edited: 02/12/2016, 08:05


% Get model dimensions
%-------------------------------------------------------------------------%
v = size(Y,2);                  % number of instances
n = size(Y,1);                  % number of observations
p = size(X,2);                  % number of regressors

% Set precision if required
%-------------------------------------------------------------------------%
if nargin < 3 || isempty(P)
    P = eye(n);                 % precision = identity matrix
end;

% Enlarge priors if required
%-------------------------------------------------------------------------%
if size(m0,2) == 1
    m0 = repmat(m0,[1 v]);      % make m0 a p x v matrix
end;
if size(b0,2) == 1
    b0 = b0*ones(1,v);          % make b0 a 1 x v vector
end;

% Estimate posterior parameters
%-------------------------------------------------------------------------%
Ln = X'*P*X + L0;               % precision of regression coefficients
mn = inv(Ln) * (X'*P*Y + L0*m0);% means of regression coefficients
an = a0 + n/2;                  % shape of residual precision
bn = zeros(1,v);                % rates of residual precision
for j = 1:v
    bn(j) = b0(j) + 1/2*(Y(:,j)'*P*Y(:,j) + m0(:,j)'*L0*m0(:,j) - mn(:,j)'*Ln*mn(:,j));
end;