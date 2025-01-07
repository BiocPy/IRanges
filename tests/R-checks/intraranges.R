starts = c(1, 20, 25, 25, 33)
widths = c(19, 5, 0, 8, 5)
x = IRanges(starts, width=widths)

shift(x, shift = 3)
shift(x, shift = c(-3, -3,-3,-3, -3))

starts = c(1, 20, 25, 33)
widths = c(19, 5, 8, 5)
x = IRanges(starts, width=widths)

narrow(x)
narrow(x, start = 4)
narrow(x, start = 4, width=2)
narrow(x, start = -4)
narrow(x, start = -4, width=2)
narrow(x, start = 4, width=-2) # err
narrow(x, start = 4, end=-2)
narrow(x, start = 4, end=2) # err
narrow(x, start = 4, end=20) # err
narrow(x, start = 4, end=10) # err
narrow(x, width = 2, end=3)
narrow(x, width=2, end=-3)
narrow(x, end=-3)
narrow(x, end=3)

resize(x, 200)
resize(x, 2, fix="end")
resize(x, 2, fix="center")


starts = c(2, 5, 1)
widths = c(2, 3, 3)
x = IRanges(starts, width=widths)

flank(x, 2)
flank(x, 2, start=FALSE)
flank(x, 2, both=TRUE)
flank(x, 2, start=FALSE, both=TRUE)
flank(x, -2, start=FALSE, both=TRUE)

starts = c(20, 21, 22, 23)
widths = c(3, 3, 3, 3)
x = IRanges(starts, width=widths)

promoters(x, 0, 0)
promoters(x, 0, 1)
promoters(x, 1, 0)

starts = c(2, 5, 1)
widths = c(2, 3, 3)
x = IRanges(starts, width=widths)

bounds = IRanges(c(0,5,3), width=c(11,2,7))
reflect(x, bounds)
bounds = IRanges(c(5), width=c(2))
reflect(x, bounds)


starts = c(1, 20, 25, 25, 33)
widths = c(19, 5, 0, 8, 5)
x = IRanges(starts, width=widths)

restrict(x, start=12, end=34)
restrict(x, start=20)
restrict(x, start=21)

threebands(x)
