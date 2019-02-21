function LME = GLM_LME(P, L0, a0, b0, Ln, an, bn)
% _
% Log Model Evidence of General Linear Model with Normal-Gamma Priors
% FORMAT LME = GLM_LME(P, L0, a0, b0, Ln, an, bn)
% 
%     P   - an n x n precision matrix embodying correlations
%     L0  - a  p x p matrix (prior precision of regression coefficients)
%     a0  - a  1 x 1 scalar (prior shape of inverse residual variance)
%     b0  - a  1 x v vector (prior rates of inverse residual variance)
%     Ln  - a  p x p matrix (posterior precision of regression coefficients)
%     an  - a  1 x 1 scalar (posterior shape of inverse residual variance)
%     bn  - a  1 x v vector (posterior rates of inverse residual variance)
% 
%     LME - a  1 x v vector of log model evidences
% 
% FORMAT LME = GLM_LME(P, L0, a0, b0, Ln, an, bn) returns the log model
% evidence for a general linear model with precision matrix P and normal-
% gamma distributed priors/posteriors for regression coefficients (L0/Ln)
% and residual variance (a0, b0 / an, bn).
% 
% Author: Joram Soch, BCCN Berlin
% E-Mail: joram.soch@bccn-berlin.de
% Edited: 02/12/2016, 08:25


% Get model dimensions
%-------------------------------------------------------------------------%
v = size(bn,2);                 % number of instances
n = size(P, 1);                 % number of observations
p = size(Ln,1);                 % number of regressors

% Calculate log model evidence
%-------------------------------------------------------------------------%
LME = 1/2*log(det(P))  - n/2*log(2*pi) + ...
      1/2*log(det(L0)) - 1/2*log(det(Ln)) + ...
      gammaln(an)      - gammaln(a0) + ...
      a0*log(b0)       - an*log(bn);