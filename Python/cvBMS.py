# The cvBMS Unit
# _
# This module assembles methods for calculating the cross-validated log model
# evidence (cvLME) for different sorts of statistical models and to perform
# cross-validated Bayesian model seelection (cvBMS) based on these cvLMEs.
# 
# The general principles of the cvBMS unit are as follows:
# o To use the module, it is simply imported via
#       import cvBMS ,
#   e.g. at the beginning of the analysis script.
# o First, a model object is generated using a command like
#       model = cvBMS.<name-of-the-model-class>(Y, [X])
#   where Y are measured data and X are independent variables, if any.
# o Second, the cvLME method is applied using a command like
#       cvLME_model = model.cvLME(S=2)
#   where S is the number of subsets into which the data are partitioned.
# o Generally, this module applies mass-univariate analysis, i.e. columns of
#   the data matrix are treated as different measurements which are analyzed
#   separately, but using the same model.
# 
# Author: Joram Soch, BCCN Berlin
# E-Mail: joram.soch@bccn-berlin.de
# Edited: 21/02/2019, 06:40


# import packages
#-----------------------------------------------------------------------------#
import numpy as np


###############################################################################
# class: model space                                                          #
###############################################################################
class MS:
    """
    The model space class allows to perform model comparison, model selection
    and family inference over a space of candidate models. A model space is
    defined by a number of log model evidences (LMEs) from which measures
    such as log Bayes factors (LBFs) and posterior model probabilities (PPs)
    can be derived.
    
    Edited: 01/02/2019, 12:30
    """
    
    # initialize MS
    #-------------------------------------------------------------------------#
    def __init__(self, LME):
        """
        Initialize a Model Space:
            LME   - an M x N array of LMEs
            self  - a model space object
            o LME - the Mx N array of log model evidences
            o M   - the number of models in the model space
            o N   - the number of instances to be analyzed
        """
        self.LME = LME          # log model evidences
        self.M   = LME.shape[0] # number of models
        self.N   = LME.shape[1] # number of instances
        
    # function: log Bayes factor
    #-------------------------------------------------------------------------#
    def LBF(self, m1=1, m2=2):
        """
        Return Log Bayes Factor between Two Models:
            m1    - index of the first model to be compared (default: 1)
            m2    - index of the second model to be compared (default: 2)
            LBF12 - a 1 x N vector of log Bayes factors in favor of first model
        """
        LBF12 = self.LME[m1-1,:] - self.LME[m2-1,:]
        return LBF12
    
    # fucntion: Bayes factor
    #-------------------------------------------------------------------------#
    def BF(self, m1=1, m2=2):
        """
        Return Bayes Factor between Two Models:
            m1   - index of the first model to be compared (default: 1)
            m2   - index of the second model to be compared (default: 2)
            BF12 - a 1 x N vector of Bayes factors in favor of first model
        """
        BF12 = np.exp(self.LBF(m1, m2))
        return BF12
        
    # function: posterior model probabilities
    #-------------------------------------------------------------------------#
    def PP(self, prior=None):
        """
        Return Posterior Probabilities of Several Models:
            prior - an M x 1 vector
                    or M x N matrix of prior model probabilities (optional)
            post  - an M x N matrix of posterior model probabilities
        """
        
        # set uniform prior
        if prior is None:
            prior = 1/self.M * np.ones((self.M,1))
        if prior.shape[1] == 1:
            prior = np.tile(prior, (1, self.N))
            
        # subtract average LMEs
        LME = self.LME
        LME = LME - np.tile(np.mean(LME,0), (self.M, 1))
        
        # calculate PPs
        post = np.exp(LME) * prior
        post = post / np.tile(np.sum(post,0), (self.M, 1))
        
        # return PPs
        return post
    
    # function: log family evidences
    #-------------------------------------------------------------------------#
    def LFE(self, m2f):
        """
        Return Log Family Evidences for Model Families:
            m2f - a 1 x M vector specifying family affiliation, i.e.
                  m2f(i) = f -> i-th model belongs to j-th family
        """
                
        # get number of model families
        F = np.int(m2f.max())
        
        # calculate log family evidences
        #---------------------------------------------------------------------#
        LFE = np.zeros((F,self.N))
        for f in range(F):
            
            # get models from family
            mf = [i for i, m in enumerate(m2f) if m == (f+1)]
            Mf = len(mf)
            
            # set uniform prior
            prior = 1/Mf * np.ones((Mf,1))
            prior = np.tile(prior, (1, self.N))
            
            # calculate LFEs
            LME_fam  = self.LME[mf,:]
            LME_fam  = LME_fam + np.log(prior) + np.log(Mf)
            LME_max  = LME_fam.max(0)
            LME_diff = LME_fam - np.tile(LME_max, (Mf, 1))
            LFE[f,:] = LME_max + np.log(np.mean(np.exp(LME_diff),0))
            
        # return log family evidence
        return LFE
    