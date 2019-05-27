def strip(xs):
    """
    WARNING: alters lengths of list, use carefully

    Returns a list with entries that `Truthy`
    """
    return [x for x in xs if x]
