function l_est = Poiss_MLE(Y, x)
% _
% Maximum Likelihood Estimation for Poisson Distribution
% FORMAT l_est = Poiss_MLE(Y, x)
% 
%     Y     - an n x v data matrix of measured counts
%     x     - an n x 1 design vector of exposure values
% 
%     l_est - a  1 x v vector of estimated Poisson rates
% 
% FORMAT l_est = Poiss_MLE(Y, x) returns maximum likelihood estimates for
% a Poisson distribution with data Y and exposures x.
% 
% Author: Joram Soch, BCCN Berlin
% E-Mail: joram.soch@bccn-berlin.de
% Edited: 21/02/2019, 14:50


% Get model dimensions
%-------------------------------------------------------------------------%
v = size(Y,2);                  % number of instances
n = size(Y,1);                  % number of observations

% Set exposures if required
%-------------------------------------------------------------------------%
if nargin < 2 || isempty(x)
    x = ones(n,1);              % standard Poisson distribution
end;

% Perform parameter estimation
%-------------------------------------------------------------------------%
l_est = sum(Y,1)./sum(x);       % estimated Poisson rates