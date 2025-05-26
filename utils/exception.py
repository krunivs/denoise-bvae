import traceback


def get_exception_traceback(exception):
    """
    get exception traceback
    :param exception: (Exception)
    :return: (str) error traceback
    """
    exception_traceback = traceback.TracebackException.from_exception(exception)
    return ''.join(exception_traceback.format())