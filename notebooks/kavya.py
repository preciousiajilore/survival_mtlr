def remove_duplicates(alist,case_sensitive=True):
    """ Remove duplicates from a list of strings. 
        Default to a filter that IS sensitive to case.
        i.e. 'dog' and 'Dog' are unique strings.
    """
    if not case_sensitive:
        alist = [i.lower() for i in alist] 
    return list(dict.fromkeys(alist))



list1 = ["kavya", "patel", "precious", "ajilore", "Kavya", "Patel"]

removelist = remove_duplicates(list1, case_sensitive = False)

print(removelist)
