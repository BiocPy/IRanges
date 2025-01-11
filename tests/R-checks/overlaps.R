subject = IRanges(c(2, 2, 10), width=c(1, 2, 3))
query = IRanges(c(1, 4, 9), width=c(5, 4, 2))

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
nearest(query, subject, select = "all")
nearest(subject, query, select="all")


query <- IRanges(c(1, 3, 9), c(3, 7, 10))
subject <- IRanges(c(3, 2, 10), c(3, 13, 12))
precede(query, subject)
precede(subject, query)

query <- IRanges(4:3, end=6)
query
subject <- IRanges(1:10, width=10:1)
subject

findOverlaps(query, subject)
findOverlaps(query, subject, select="all")
findOverlaps(query, subject, select="all", maxgap=0)

precede(query, subject)
precede(query, subject, select="all")

follow(query, subject)
follow(query, subject, select="all")

nearest(query, subject)
nearest(query, subject, select="all")
