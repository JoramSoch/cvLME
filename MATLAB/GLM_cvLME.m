function [cvLME, oosLME] = GLM_cvLME(Y, X, P, S)
% _
% Cross-Validated Log Model Evidence for General Linear Model
% FORMAT [cvLME, oosLME] = GLM_cvLME(Y, X, P, S)
% 
%     Y      - an n x v data matrix of measured signals
%     X      - an n x p design matrix of predictor variables
%     P      - an n x n precision matrix specifying correlations
%     S      - the number of subsets into which data are partitioned
% 
%     cvLME  - a  1 x v vector of cross-validated log model evidences
%     oosLME - an S x v matrix of out-of-sample log model evidences
% 
% FORMAT [cvLME, oosLME] = GLM_cvLME(Y, X, P, S) calculates the cross-
% validated log model evidence for a general linear model with data Y,
% design matrix X, precision matrix P and normal-gamma distributed priors
% for regression coefficients and residual variance using S data subsets.
% 
% Author: Joram Soch, BCCN Berlin
% E-Mail: joram.soch@bccn-berlin.de
% Edited: 02/12/2016, 09:00


% Get model dimensions
%-------------------------------------------------------------------------%
v  = size(Y,2);                 % number of instances
n  = size(Y,1);                 % number of observations
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
m0_ni = zeros(p,1);             % flat Gaussian
L0_ni = zeros(p,p);
a0_ni = 0;                      % Jeffrey's prior
b0_ni = 0;

% Prepare cross-validation
%-------------------------------------------------------------------------%
npS = floor(n/S);               % number of data points per subset, truncated
is  = [1:(S*npS)];              % indices for all data, without remainders

% Calculate out-of-sample log model evidences
%-------------------------------------------------------------------------%
oosLME = zeros(S,v);
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
    m01 = m0_ni; L01 = L0_ni; a01 = a0_ni; b01 = b0_ni;
    [mn1, Ln1, an1, bn1] = GLM_Bayes(Y1, X1, P1, m01, L01, a01, b01);
    m02 = mn1; L02 = Ln1; a02 = an1; b02 = bn1;
    [mn2, Ln2, an2, bn2] = GLM_Bayes(Y2, X2, P2, m02, L02, a02, b02);
    oosLME(i,:) = GLM_LME(P2, L02, a02, b02, Ln2, an2, bn2);
end;

% Calculate cross-validated log model evidences
%-------------------------------------------------------------------------%
cvLME = sum(oosLME,1);