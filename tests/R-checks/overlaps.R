subject = IRanges(c(2, 2, 10), width=c(1, 2, 3))
query = IRanges(c(1, 4), width=c(5, 4))

findOverlaps(query, subject)
findOverlaps(query, subject, maxgap = 0)
findOverlaps(query, subject, select="first")
findOverlaps(query, subject, select="last")
findOverlaps(query, subject, select="arbitrary")
findOverlaps(query, subject, type="start")
findOverlaps(query, subject, type="start", maxgap=1L)
findOverlaps(query, subject, type="end", select="first")
findOverlaps(query, subject, type="within", maxgap=1L)

query = IRanges(c(1, 3, 9), width=c(2, 5, 2))
subject = IRanges(c(3, 5, 12), width=c(1, 2, 1))

nearest(query, subject)
nearest(subject, query)
