import sys


def log(log_filename, mode='a'):
    def wrapper(func):
        def logged_func(*args, **kwargs):
            temp = sys.stdout
            sys.stdout = open(log_filename, mode)
            return_val = func(*args, **kwargs)
            sys.stdout.close()
            sys.stdout = temp
            return return_val
        return logged_func
    return wrapper


