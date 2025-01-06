x = IRanges(c(1, 5, -2, 0, 14), width=c(10, 5, 6, 12, 4))
y = IRanges(c(14, 0, -5, 6, 18), width=c(7, 3, 8, 3, 3))

union(x,y)
setdiff(x,y)
intersect(x,y)
intersect(y,x)
