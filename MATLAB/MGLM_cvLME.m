function [cvLME, oosLME] = MGLM_cvLME(Y, X, P, S)
% _
% Cross-Validated Log Model Evidence for Multivariate General Linear Model
% FORMAT [cvLME, oosLME] = MGLM_cvLME(Y, X, P, S)
% 
%     Y      - an n x v data matrix of measured signals
%     X      - an n x p design matrix of predictor variables
%     P      - an n x n precision matrix specifying correlations
%     S      - the number of subsets into which data are partitioned
% 
%     cvLME  - a  1 x v vector of cross-validated log model evidences
%     oosLME - an S x v matrix of out-of-sample log model evidences
% 
% FORMAT [cvLME, oosLME] = MGLM_cvLME(Y, X, P, S) calculates the cross-
% validated log model evidence for a multivariate general linear model with
% data matrix Y, design matrix X, precision matrix P and normal-Wishart
% distributed priors for regression coefficients and signal covariance
% using S data subsets.
% 
% Author: Joram Soch, BCCN Berlin
% E-Mail: joram.soch@bccn-berlin.de
% Edited: 06/07/2022, 11:59


% Get model dimensions
%-------------------------------------------------------------------------%
n  = size(Y,1);                 % number of observations
v  = size(Y,2);                 % number of signals
p  = size(X,2);                 % number of regressors

% Set precision if required
%-------------------------------------------------------------------------%
if nargin < 3 || isempty(P)
    P = eye(n);                 % precision = identity matrix
end;

% Set CV folds if required
%-------------------------------------------------------------------------%
if nargin < 4 || isempty(S)
    S = 2;                      % split-half cross-validation
end;

% Set non-informative priors
%-------------------------------------------------------------------------%
M0_ni = zeros(p,v);             % flat Gaussian
L0_ni = zeros(p,p);
O0_ni = zeros(v,v);             % non-informative Wishart
v0_ni = 0;

% Prepare cross-validation
%-------------------------------------------------------------------------%
npS = floor(n/S);               % number of data points per subset, truncated
is  = [1:(S*npS)];              % indices for all data, without remainders

% Calculate out-of-sample log model evidences
%-------------------------------------------------------------------------%
oosLME = zeros(S,1);
for i = 1:S
    % set indices
    i2 = [((i-1)*npS+1):(i*npS)];           % test indices
    i1 = setdiff(is,i2);                    % training indices
    % partition data
    Y1 = Y(i1,:);                           % training data
    X1 = X(i1,:);
    P1 = P(i1,i1);
    Y2 = Y(i2,:);                           % test data
    X2 = X(i2,:);
    P2 = P(i2,i2);
    % calculate oosLME
    M01 = M0_ni; L01 = L0_ni; O01 = O0_ni; v01 = v0_ni;
    [Mn1, Ln1, On1, vn1] = MGLM_Bayes(Y1, X1, P1, M01, L01, O01, v01);
    M02 = Mn1; L02 = Ln1; O02 = On1; v02 = vn1;
    [Mn2, Ln2, On2, vn2] = MGLM_Bayes(Y2, X2, P2, M02, L02, O02, v02);
    oosLME(i,:) = MGLM_LME(P2, L02, O02, v02, Ln2, On2, vn2);
end;

% Calculate cross-validated log model evidences
%-------------------------------------------------------------------------%
cvLME = sum(oosLME);