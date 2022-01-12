# https://github.com/aertslab/GENIE3/blob/master/vignettes/GENIE3.Rmd

library(doRNG)
library(GENIE3)

call_genie3 <- function(X, regulators=1:nrow(X)) {
  rownames(X) <- paste("Gene", 1:nrow(X), sep="")
  colnames(X) <- paste("Sample", 1:ncol(X), sep="")
  GENIE3(X, regulators=regulators, nCores=4)
}