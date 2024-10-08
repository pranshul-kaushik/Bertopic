import os
import shutil
from datetime import datetime

from src.utils.constants import TEMP_FOLDER


def get_timestamp():
    """
    Return the current timestamp as a string in the format
    "%Y%m%d_%H%M%S".
    """
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def extract_corpus(fileobj):
    """
    Reads a file object and returns its contents as a list of strings.

    Copies the file to a temporary location on disk, then reads it line by line
    into a list.

    Parameters
    ----------
    fileobj : File-like object
        The file to read.

    Returns
    -------
    lines : List of str
        The contents of the file as a list of strings.
    """
    path = TEMP_FOLDER + get_timestamp() + os.path.basename(fileobj)
    shutil.copyfile(fileobj.name, path)

    with open(path, "r") as f:
        lines = f.readlines()

    os.remove(path)

    return lines, fileobj.name
