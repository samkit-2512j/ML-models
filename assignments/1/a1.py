import os
import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './')))
from a1_knn import *
from a1_lr import *

if __name__=='__main__':
    run='lr' # 'knn' or 'lr'
    if run=='knn':
        knn_main()

    if run=='lr':
        lr_main()
