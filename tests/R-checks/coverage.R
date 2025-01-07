library(IRanges)
ir <- IRanges(c(1, 8, 14, 15, 19, 34, 40), width = c(12, 6, 6, 15, 6, 2, 7))

res1 <- coverage(ir)
paste(as.vector(res1), collapse=",")
ir <- IRanges(start=c(-2L, 6L, 9L, -4L, 1L, 0L, -6L, 10L),
              width=c( 5L, 0L, 6L,  1L, 4L, 3L,  2L,  3L))

res2 <- coverage(ir)
paste(as.vector(res2), collapse=",")

res3 <- coverage(ir, shift=7)
paste(as.vector(res3), collapse=",")

res4 <- coverage(ir, shift=7, width=27)
paste(as.vector(res4), collapse=",")

res5 <- coverage(ir, weight=10)
paste(as.vector(res5), collapse=",")
