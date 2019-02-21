function post = MS_PP(LME, prior)
% _
% Posterior Probabilities of Several Models
% FORMAT post = MS_PP(LME, prior)
% 
%     LME   - an M x N matrix of log model evidences
%     prior - an M x N matrix of prior model probabilities
% 
%     post  - an M x N matrix of posterior model probabilities
% 
% FORMAT post = MS_PP(LME, prior) calculates posterior model probabilities
% post from log model evidences LMEs and prior model probabilities prior.
% 
% Author: Joram Soch, BCCN Berlin
% E-Mail: joram.soch@bccn-berlin.de
% Edited: 25/11/2016, 15:45


% Number of data sets
%-------------------------------------------------------------------------%
M = size(LME,1);
N = size(LME,2);

% Set uniform prior
%-------------------------------------------------------------------------%
if nargin < 2 || isempty(prior)
    prior = 1/M * ones(M,1);
end;
if size(prior,2) == 1
    prior = repmat(prior,[1 N]);
end;

% Subtract average log model evidence
%-------------------------------------------------------------------------%
LME = LME - repmat(mean(LME,1),[M 1]);

% Exponentiate log model evidences
%-------------------------------------------------------------------------%
exp_LME = exp(LME);

% Estimate posterior probabilities
%-------------------------------------------------------------------------%
post = exp_LME.*prior;
post = post./repmat(sum(post,1),[M 1]);