function [B_est, s2_est] = GLM_MLE(Y, X, V)
% _
% Maximum Likelihood Estimation for General Linear Model
% FORMAT [B_est, s2_est] = GLM_MLE(Y, X, V)
% 
%     Y      - an n x v data matrix of measured signals
%     X      - an n x p design matrix of predictor variables
%     V      - an n x n covariance matrix specifying correlations
% 
%     B_est  - a  p x v matrix of estimated regression coefficients
%     s2_est - a  1 x v vector of estimated residual variances
% 
% FORMAT [B_est, s2_est] = GLM_MLE(Y, X, V) returns maximum likelihood
% estimates for a general linear model with data Y, known design matrix X
% and covariance structure V, but unknown regression coefficients B and
% residual variance s2.
% 
% Author: Joram Soch, BCCN Berlin
% E-Mail: joram.soch@bccn-berlin.de
% Edited: 02/12/2016, 07:55


% Get model dimensions
%-------------------------------------------------------------------------%
v = size(Y,2);                  % number of instances
n = size(Y,1);                  % number of observations
p = size(X,2);                  % number of regressors

% Set covariance if required
%-------------------------------------------------------------------------%
if nargin < 3 || isempty(V)
    V = eye(n);                 % covariance = identity matrix
end;

% Prepare parameter estimation
%-------------------------------------------------------------------------%
P = inv(V);                     % precision = inverse covariance

% Perform parameter estimation
%-------------------------------------------------------------------------%
B_est = (X'*P*X)^-1 * X'*P*Y;   % estimated regression coefficients
E_est = Y - X*B_est;            % estimated residuals/errors/noise
s2_est= zeros(1,v);             % estimated residual variance
for j = 1:v
    s2_est(j) = 1/n * (E_est(:,j))' * P * (E_est(:,j));
end;