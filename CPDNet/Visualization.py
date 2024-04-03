

import argparse
import subprocess
import tensorflow as tf
import numpy as np
from datetime import datetime
import json
import os
import sys
import datetime
import time
import collections
import ioUtil



Train_examples = ioUtil.load_examples(FLAGS.train_hdf5, FLAGS.domain_A, FLAGS.domain_B, 'names')