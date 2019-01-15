'''
Author: Luke Prince
Date: 11 January 2019
'''

import os

class Analysis(object):
    def __init__(self, data_path):
        self.cellids = [dirname[0][2:] for dirname in os.walk(data_path) if x[0].startswith('./1')]
celldirs.sort()