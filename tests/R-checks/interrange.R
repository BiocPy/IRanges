ir <- IRanges(start=c(-2L, 6L, 9L, -4L, 1L, 0L, -6L, 10L),
              width=c( 5L, 0L, 6L,  1L, 4L, 3L,  2L,  3L))

res1 <- range(ir)

red1 <- reduce(ir)
red2 <- reduce(ir, drop.empty.ranges = TRUE)