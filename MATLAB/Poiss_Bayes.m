function [an, bn] = Poiss_Bayes(Y, x, a0, b0)
% _
% Bayesian Estimation of Poisson Distribution with Gamma Prior
% FORMAT [an, bn] = Poiss_Bayes(Y, x, a0, b0)
% 
%     Y  - an n x v data matrix of measured counts
%     x  - an n x 1 design vector of exposure values
%     a0 - a  1 x v vector (prior shapes of the Poisson rates)
%     b0 - a  1 x 1 scalar (prior rate of the Poisson rates)
% 
%     an - a  1 x v vector (posterior shapes of the Poisson rates)
%     bn - a  1 x 1 scalar (posterior rate of the Poisson rates)
% 
% FORMAT [an, bn] = Poiss_Bayes(Y, x, a0, b0) returns the posterior
% parameter estimates for a Poisson distribution with data Y, exposures x
% and gamma distributed priors for the Poisson rates (a0, b0).
% 
% References:
% [1] Gelman A et al. (2014): "Bayesian Data Analysis".
%     Third Edition, Chapman & Hall, ch. 2.6, pp. 43-46.
% 
% Author: Joram Soch, BCCN Berlin
% E-Mail: joram.soch@bccn-berlin.de
% Edited: 21/02/2019, 13:55


% Get model dimensions
%-------------------------------------------------------------------------%
v = size(Y,2);                  % number of instances
n = size(Y,1);                  % number of observations

% Set exposures if required
%-------------------------------------------------------------------------%
if nargin < 2 || isempty(x)
    x = ones(n,1);              % standard Poisson distribution
end;

% Enlarge priors if required
%-------------------------------------------------------------------------%
if size(a0,2) == 1
    a0 = a0*ones(1,v);          % make a0 a 1 x v vector
end;

% Estimate posterior parameters
%-------------------------------------------------------------------------%
an = a0 + sum(Y,1);             % shapes of the Poisson rates
bn = b0 + sum(x);               % rate of the Poisson rates