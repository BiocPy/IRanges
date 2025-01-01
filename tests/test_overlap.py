from iranges import IRanges


def test_overlaps():
    # Create intervals
    ir = IRanges([1, 10, 20], [5, 5, 5])

    # Find overlaps
    query = IRanges([8, 15], [4, 8])
    overlaps = ir.find_overlaps(query)

    assert overlaps is not None
