# https://github.com/slawekj/ennet/blob/master/ennet/R/ennet.R

library(ennet)

call_ennet <- function(X, K=matrix(0,nrow(X),ncol(X)), Tf=1:ncol(X)) {
  ennet(E=X, K=K, Tf=Tf)
}