from os.path import join
from os import listdir as ls
import json
import random
from itertools import chain
from glob import glob
import os
from pathlib import Path
import shutil
from labellists import *

random.seed(100)

rootpath = '/data/keshav/360/finalEgok360'

def savefile(x, fname):
    with open(fname, 'w') as f:
        f.write(json.dumps(x))

def openfile(fname):
    with open(fname,'rb') as f:
        return json.load(f)



if __name__ == '__main__':
    videos = glob(join(rootpath,'images/*/*/*'))

    random.shuffle(videos)

    class_counts = [*map(lambda x:x.split('/')[-3]+'/'+x.split('/')[-2],videos)]
    from collections import Counter

    class_counts = dict(Counter(class_counts))

    keys = list(class_counts.keys())
    emptval = [0 for i in range(len(keys))]

    test_dict = dict(zip(keys,emptval))
    val_dict = dict(zip(keys,emptval))

    def countlimit(j):
        if j<40:
            return 1
        else:
            return int(j*0.05)

    limit = {i:countlimit(j) for i,j in class_counts.items()}

    print(limit)

    print(test_dict)


    train = []
    test = []
    val = []

    def getsample(x):
        p = Path(x)
        vidname = p.name
        superclass = p.parent.parent.name
        subclass = p.parent.name
        cls = superclass + '/' + subclass
        label = labellist[cls]

        u = x.replace('/images/','/flows/u/')
        v = x.replace('/images/', '/flows/v/')

        assert os.path.exists(u)
        assert os.path.exists(v)
        assert os.path.exists(x)

        sample = {
            'imgs':x,
            'u':u,
            'v':v,
            'cls':cls,
            'superclass':superclass,
            'vidname':vidname,
            'label':label
        }

        return sample


    for vid in videos:
        cls = vid.split('/')[-3]+'/'+vid.split('/')[-2]
        sample = getsample(vid)
        if test_dict.get(cls)<limit.get(cls):
            test.append(sample)
            test_dict[cls] += 1
            continue
        if val_dict.get(cls)<limit.get(cls):
            val.append(sample)
            val_dict[cls]+=1
            continue

        train.append(sample)


    print(len(train))
    print(len(test))
    print(len(val))
    print(len(train+test+val))
    print(len(videos))

    input()
    print(train)
    input()
    print(test)
    input()
    print(val)


    savefile(train,'./data_list/train.txt')
    savefile(test,'./data_list/test.txt')
    savefile(val,'./data_list/val.txt')
