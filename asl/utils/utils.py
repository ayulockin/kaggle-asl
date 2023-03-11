import re
import string
import random


def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    """
    Generate random string that acts as unique id.
    """
    return ''.join(random.choice(chars) for _ in range(size))


def natural_keys(text):
    """
    Key to be passed to sorted.
    """
    def atoi(text):
        return int(text) if text.isdigit() else text
    
    return [atoi(c) for c in re.split(r'(\d+)', text)]


