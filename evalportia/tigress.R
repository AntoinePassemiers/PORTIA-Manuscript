library(tigress)

call_tigress <- function(expdata, tflist=colnames(expdata), nsplit=1000) {
  tigress(expdata, tflist=tflist, nsplit=nsplit, allsteps=FALSE, verb=FALSE, usemulticore=TRUE)
}