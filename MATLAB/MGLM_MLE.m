function [B_est, S_est] = MGLM_MLE(Y, X, V)
% _
% Maximum Likelihood Estimation for Multivariate General Linear Model
% FORMAT [B_est, S_est] = MGLM_MLE(Y, X, V)
% 
%     Y     - an n x v data matrix of measured signals
%     X     - an n x p design matrix of predictor variables
%     V     - an n x n covariance matrix specifying correlations
% 
%     B_est - a  p x v matrix of estimated regression coefficients
%     S_est - a  v x v matrix, the estimated residual variances
% 
% FORMAT [B_est, S_est] = MGLM_MLE(Y, X, V) returns maximum likelihood
% estimates for a multivariate general linear model with data matrix Y,
% known design matrix X and covariance structure V, but unknown regression
% coefficients B and signal covariance S.
% 
% Author: Joram Soch, BCCN Berlin
% E-Mail: joram.soch@bccn-berlin.de
% Edited: 06/07/2022, 11:31


% Get model dimensions
%-------------------------------------------------------------------------%
n = size(Y,1);                  % number of observations
v = size(Y,2);                  % number of signals
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
S_est = 1/n * E_est'*P*E_est;    % estimated signal covariance