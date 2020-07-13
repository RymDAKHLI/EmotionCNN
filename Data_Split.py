import argparse
import os
import random

def main():
    parser = argparse.ArgumentParser(description='Split data')
    parser.add_argument('--split-training', default=0.8, type=float)
    parser.add_argument('--split-testing', default=0.2, type=float)
    args = parser.parse_args()
    split(args)

def split(args):
    traindir = 'train'
    testdir = 'test'
    # Find subjects
    available = {}
    for (dirpath, dirnames, filenames) in os.walk(traindir):
        for name in [n for n in filenames \
                     if n.lower().endswith('.jpg')]:
            filen = os.path.join(dirpath, name)
            available[name[0:3]] = True
    # Split subjects
    subjects = list(available.keys())
    random.shuffle(subjects)
    traini = int(len(subjects)*args.split_training)
    testi = int(len(subjects)*args.split_testing)

    # Verify our sets have elements
    if min(traini, testi) < 1:
        raise ValueError('Size of one of the sets is zero')

    test = subjects[:testi]
    train = subjects[testi:]
   

    os.makedirs(testdir, exist_ok=True)  
    # Move to new directories
    for (dirpath, dirnames, filenames) in os.walk(traindir):
        for filen in [f for f in filenames if f.lower().endswith('.jpg')]:
            path = dirpath.split('/')[1:]
            if filen[0:3] in train:
                pass
                # Should be already there
                #os.renames(os.path.join(dirpath, filen), 
                #          os.path.join(traindir, *dirnames[1:], filen)
            elif filen[0:3] in test:
                os.renames(os.path.join(dirpath, filen), 
                          os.path.join(testdir, *path, filen))

if __name__ == '__main__':
    main()