library(plyr)
library(ggplot2)
library(tidyverse)
library(lubridate)

files = list.files("./datasets")

for (i in files) {
  ticker = read.csv(paste('./datasets/', i, sep = ""))
  symbol = strsplit(i,split = ".csv")
  print(symbol)
  ticker$Date = ymd(ticker$Date)
  ticker_plot = ggplot(data = ticker, aes(x = Date, y = Adj.Close)) +
    geom_line() +
    ggtitle(symbol)
  plot(ticker_plot)
}
  