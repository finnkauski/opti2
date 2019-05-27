def strip(xs):
    """
    WARNING: alters lengths of list, use carefully

    Returns a list with entries that `Truthy`
    """
    return [x for x in xs if x]


def unpack(xxs):
    """
    Unpack a list of lists

    Parameters
    ----------
    xxs : list[list]
        the list of lists to unpack

    Returns
    -------
    list
        unpacked list
    """
    return [x for xs in xxs for x in xs]
