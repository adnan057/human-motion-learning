import numpy as np
import pickle
#import cPickle as pickle
import time
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.image import extract_patches_2d
import matplotlib.pyplot as plt

patches = pickle.load(open('data/patches_10000.p','rb'))
