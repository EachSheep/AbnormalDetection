def list_flatten(lst : list) -> list:
    """A function to flatten a list of lists
    Args：
        lst (list): a list of lists
    Returns:
        list: a flattened list
    """
    result = []
    def inner(lst):
        for item in lst:
            if isinstance(item,list):
                inner(item)
            else:
                result.append(item)
    inner(lst)
    return result

if __name__ == "__main__":
    lst = [[1,2,3,4],[1,2,3],[5,6]]
    a = flat_list(lst)
    print(a)
