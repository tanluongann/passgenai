import sys
import os
import random
import time
import json

from learners.GraphLearner import GraphLearner

try:
    folder = sys.argv[1]
except:
    folder = 'data/simplelist'

try:
    max_passwords = int(sys.argv[2])
except:
    max_passwords = 500

print('Generating from %s' % folder)

gl = GraphLearner(folder=folder, max_passwords=max_passwords)
gl.generate()

print('Generated %s passwords' % len(gl.generated))