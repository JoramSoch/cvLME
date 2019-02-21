function [LBF12, BF12, PP1] = MS_LBF(LME1, LME2)
% _
% (Log) Bayes Factor between Two Models
% FORMAT [LBF12, BF12, PP1] = MS_LBF(LME1, LME2)
% 
%     LME1  - a 1 x N vector of log model evidences for model 1
%     LME2  - a 1 x N vector of log model evidences for model 2
% 
%     LBF12 - a 1 x N vector of log Bayes factors in favor of model 1
%     BF12  - a 1 x N vector of Bayes factors in favor of model 1
%     PP1   - a 1 x N vector of posterior probabilities of model 1
% 
% FORMAT [LBF12, BF12, PP1] = MS_LBF(LME1, LME2) calculates LBF and BF
% between two models using log model evidences LME1 and LME2 as well as
% the posterior probability of model one.
% 
% Author: Joram Soch, BCCN Berlin
% E-Mail: joram.soch@bccn-berlin.de
% Edited: 25/11/2016, 14:45


% Log Bayes factor
%-------------------------------------------------------------------------%
LBF12 = LME1 - LME2;

% Bayes factor
%-------------------------------------------------------------------------%
BF12 = exp(LBF12);

% Posterior probability
%-------------------------------------------------------------------------%
PP1 = BF12./(BF12+1);