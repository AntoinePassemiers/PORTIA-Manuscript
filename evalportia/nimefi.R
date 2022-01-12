source("nimefi_main.R")

call_nimefi <- function(expressionMatrix, predictorIndices=NULL) {

  expressionMatrix <- as.matrix(expressionMatrix)

  ngenes <- ncol(expressionMatrix)
  genenames <- vector(length=ngenes)
  for (i in 1:ngenes) {
    genenames[i] = sprintf("G%d", i)
  }
  colnames(expressionMatrix) <- genenames

  NIMEFI(expressionMatrix, predictorIndices=predictorIndices,
    outputFileName=NULL)
}