data <- read.csv("train.csv", header=TRUE)
sums <- data["adult_males"] + data["subadult_males"] + data["adult_females"] + data["juveniles"]
write.table(sums, "sum.csv", row.names=FALSE, col.names=FALSE)

ours <- read.csv("our_counts.csv", header=FALSE)
our <- as.numeric(ours[,1])
sum <- as.numeric(sums[,1])

o <- our
o[o == 0] <- NA

diff = sum - o
diff[diff<=0] <- NA
sum(diff, na.rm=TRUE) / length(sum)

mean(sum)
