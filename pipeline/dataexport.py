#!/usr/bin/env python3
import pandas as pd
import numpy as np
import csv

from . import CellAttr, rsetattr, rgetattr

def export_csv(list_CellAttr, attr, filename):
    """
    Exports a chosen CellAttr attribute from a list of CellAttr objects to a
    CSV file.  Default attribute is fluo_processed.
    """
    if attr is None:
        attr = 'fluo_processed'
    with open(filename, mode = 'w') as exportfobj:
        exportfwriter = csv.writer(exportfobj, delimiter = ',',
                                        quotechar = '"',
                                        quoting = csv.QUOTE_MINIMAL)
        for cell in list_CellAttr:
            # if attribute is not iterable, put it into a list so that I can
            # write it
            try:
                iter(rgetattr(cell, attr))
            except Exception:
                exportfwriter.writerow([rgetattr(cell, attr)])
            else:
                exportfwriter.writerow(rgetattr(cell, attr))

