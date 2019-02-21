function LME = Poiss_LME(Y, x, a0, b0, an, bn)
% _
% Log Model Evidence of Poisson Distribution with Gamma Priors
% FORMAT LME = Poiss_LME(Y, x, a0, b0, an, bn)
% 
%     Y   - an n x v data matrix of measured counts
%     x   - an n x 1 design vector of exposure values
%     a0  - a  1 x v vector (prior shapes of the Poisson rates)
%     b0  - a  1 x 1 scalar (prior rate of the Poisson rates)
%     an  - a  1 x v vector (posterior shapes of the Poisson rates)
%     bn  - a  1 x 1 scalar (posterior rate of the Poisson rates)
% 
%     LME - a  1 x v vector of log model evidences
% 
% FORMAT LME = Poiss_LME(Y, x, a0, b0, an, bn) returns the log model
% evidence for a Poisson distribution with data Y, exposures x and gamma
% distributed priors/posteriors for the Poisson rates (a0, b0 / an, bn).
% 
% Author: Joram Soch, BCCN Berlin
% E-Mail: joram.soch@bccn-berlin.de
% Edited: 21/02/2019, 14:05


% Get model dimensions
%-------------------------------------------------------------------------%
v = size(Y,2);                  % number of instances
n = size(Y,1);                  % number of observations

% Set exposures if required
%-------------------------------------------------------------------------%
if nargin < 2 || isempty(x)
    x = ones(n,1);              % standard Poisson distribution
end;

% Calculate log model evidence
%-------------------------------------------------------------------------%
LME = sum(Y * repmat(log(x),[1 v]), 1) - sum(gammaln(Y + 1), 1) + ...
      gammaln(an) - gammaln(a0) + a0*log(b0) - an*log(bn);