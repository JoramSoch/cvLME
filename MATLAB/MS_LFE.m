function LFE = MS_LFE(LME, m2f)
% _
% Log Family Evidences for Model Families
% FORMAT LFE = MS_LFE(LME, m2f)
% 
%     LME - an M x N matrix of log model evidences
%     m2f - a  1 x M vector of family affiliations
% 
%     LFE - an F x N vector of log family evidences
% 
% FORMAT LFE = MSC_LFE(LME, m2f) computes the log family evidences LFE for
% a number of log model evidences LME where family affiliation is given by
% m2f, i.e. m2f(i) = j means the i-th model belongs to the j-th family.
% 
% Author: Joram Soch, BCCN Berlin
% E-Mail: joram.soch@bccn-berlin.de
% Edited: 21/02/2019, 06:50


% Number of data sets
%-------------------------------------------------------------------------%
M = size(LME,1);
N = size(LME,2);
F = max(m2f);

% Log family evidences
%-------------------------------------------------------------------------%
LFE = zeros(F,N);
for j = 1:F
    
    % Get models from family
    %---------------------------------------------------------------------%
    Mf = numel(find(m2f==j));
    
    % Set uniform prior
    %---------------------------------------------------------------------%
    prior = 1/Mf * ones(Mf,1);
    prior = repmat(prior,[1 N]);
    
    % Calculate LFEs
    %---------------------------------------------------------------------%
    LME_fam  = LME(m2f==j,:);
    LME_fam  = LME_fam + log(prior) + log(Mf);
    LME_max  = max(LME_fam,[],1);
    LME_diff = LME_fam - repmat(LME_max,[M 1]);
    LFE(j,:) = LME_max + log(mean(exp(LME_diff),1));
    
end;