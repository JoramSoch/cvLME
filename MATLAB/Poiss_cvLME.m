function [cvLME, oosLME] = Poiss_cvLME(Y, x, S)
% _
% Cross-Validated Log Model Evidence for Poisson Distribution
% FORMAT [cvLME, oosLME] = Poiss_cvLME(Y, x, S)
% 
%     Y      - an n x v data matrix of measured counts
%     x      - an n x 1 design vector of exposure values
%     S      - the number of subsets into which data are partitioned
% 
%     cvLME  - a  1 x v vector of cross-validated log model evidences
%     oosLME - an S x v matrix of out-of-sample log model evidences
% 
% FORMAT [cvLME, oosLME] = Poiss_cvLME(Y, x, S) calculates the cross-
% validated log model evidence for a Poisson distribution with data Y,
% exposures x and gamma distributed priors for the Poisson rates using
% S data subsets.
% 
% Author: Joram Soch, BCCN Berlin
% E-Mail: joram.soch@bccn-berlin.de
% Edited: 21/02/2019, 14:20


% Get model dimensions
%-------------------------------------------------------------------------%
v  = size(Y,2);                 % number of instances
n  = size(Y,1);                 % number of observations

% Set exposures if required
%-------------------------------------------------------------------------%
if nargin < 2 || isempty(x)
    x = ones(n,1);              % standard Poisson distribution
end;

% Set CV folds if required
%-------------------------------------------------------------------------%
if nargin < 3 || isempty(S)
    S = 2;                      % split-half cross-validation
end;

% Set non-informative priors
%-------------------------------------------------------------------------%
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
    x1 = x(i1);
    Y2 = Y(i2,:);                           % test data
    x2 = x(i2);
    % calculate oosLME
    a01 = a0_ni; b01 = b0_ni;
    [an1, bn1] = Poiss_Bayes(Y1, x1, a01, b01);
    a02 = an1; b02 = bn1;
    [an2, bn2] = Poiss_Bayes(Y2, x2, a02, b02);
    oosLME(i,:) = Poiss_LME(P2, L02, a02, b02, Ln2, an2, bn2);
end;

% Calculate cross-validated log model evidences
%-------------------------------------------------------------------------%
cvLME = sum(oosLME,1);