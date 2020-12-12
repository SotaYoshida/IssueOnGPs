# IssueOnGPs

We summarized the following issues:  

~・The GPy library possibly give wrong predictions~  
(I was wrong. GPy library sets the noise variance as 1.0 (!!!) by default.
This may cause substantial problems especially when one whitens data...)  
・The epsilon prescription in GPs (i.e., adding infinitesimal diagonal matrix to the covariance to recover positive semi-definiteness) may cause some problems.  
