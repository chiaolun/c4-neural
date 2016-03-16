require(data.table)
require(ggplot2)

dat = fread("results.csv")

dat[, params := as.factor(
          paste(optimizer
              , ifelse(batch_normalization==1, "batch_normalization", "")
              , ifelse(flat==1, "wide", "pyramid")
              , activation
              , nlayer
              , ncensor
              , trainsize))]

dat = dat[trainsize == 1000000]

qplot(epoch, val_acc, geom = "line"
    , col = optimizer
    , group = params
    , data = dat
    , main = "ADAM clearly works better")

dat = dat[optimizer == "adam"]

qplot(epoch, val_acc, geom = "line"
    , col = paste(ncensor)
    , group = params
    , data = dat
    , main = "Let's focus on ncensor = 3")

dat = dat[ncensor == 3]

qplot(epoch, val_acc, geom = "line"
    , col = as.factor(nlayer)
    , group = params
    , data = dat
    , main = "nlayer = 5 seems enough")

dat = dat[nlayer == 5]

qplot(epoch, val_acc, geom = "line"
    , col = as.factor(flat)
    , group = params
    , data = dat
    , main = "pyramid shape is fine, no need to use wide network")

dat = dat[flat == 0]

qplot(epoch, val_acc, geom = "line"
    , col = activation
    , group = params
    , data = dat
    , main = "relu works better")

require(knitr)
