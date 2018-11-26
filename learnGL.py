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

print('Learning from %s' % folder)

gl = GraphLearner()
gl.set_folder(folder)
gl.load_data()
gl.learn()
gl.save_model()
