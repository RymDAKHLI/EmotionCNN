import os
from multiprocessing import Pool
import sys
import os
import dlib
from skimage import io
import argparse
import math
import numpy as np
import skimage.transform

def transform(args, files):

    detector = dlib.get_frontal_face_detector()
    if args.window:
        win = dlib.image_window()
    progress = 1
    count = len(files)
    for line in files:
        print("Processing file: {} {}/{}".format(line, progress, count))
        progress += 1
        img = io.imread(line)
        dets, scores, idx = detector.run(img, 1, args.threshold)
        print("Number of faces detected: {}".format(len(dets)))
        if args.ignore_multi and len(dets) > 1:
            print("Skipping image with more then one face")
            continue
            
        if len(dets) == 0:
            print('Skipping image as no faces found')
            continue

        d = dets[0]

        (ymax, xmax, _) = img.shape
        g = args.grow
        l, t, r, b = max(d.left()-g, 0), max(d.top()-g, 0), \
                     min(d.right()+g, xmax), min(d.bottom()+g, ymax)
        # Proportion check
        if ((r-l)*(b-t))/(xmax * ymax) < args.min_proportion:
            print('Image proportion too small, skipping')

        if args.window:
            win.clear_overlay()
            win.set_image(img)
            win.add_overlay(dets)
            dlib.hit_enter_to_continue()
        
        img = img[np.arange(t, b),:,:]
        img = img[:, np.arange(l, r), :]
        if args.resize:
            img = skimage.transform.resize(img, 
                  (args.row_resize, args.col_resize))
        io.imsave(args.o + '/' + os.path.basename(line), img)

def main():
    parser = argparse.ArgumentParser(description="Preprocesses photos to" +
        " their face detected version")
    # Input/Output
    parser.add_argument('-o')
    # Lower threshold is more lossy
    parser.add_argument('--threshold', default=0.0, type=float)
    parser.add_argument('--window', default=False, type=bool)
    parser.add_argument('--ignore-multi', default=True, type=bool)
    parser.add_argument('--grow', default=10, type=int)
    parser.add_argument('--resize', default=True, type=bool)
    parser.add_argument('--row-resize', default=512, type=int)
    parser.add_argument('--col-resize', default=512, type=int)
    parser.add_argument('--min-proportion', default=0.1, type=float)

    args = parser.parse_args()

    files=[line.strip() for line in sys.stdin]

    transform(args, files)
if __name__ == '__main__':
    main()





class AttributeDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def _inner(arg):
    (k, v, emotion, outdir, args) = arg
    # Output
    args['o'] = outdir + '/' + emotion[k]
    face_detector.transform(AttributeDict(args), v)


def get_folders_kdef():
    fname = 'KDEF'
    outdir = 'train'
    emotion = {
        'AF': "afraid",
        'AN': "angry",
        'DI': "disgusted",
        'HA': "happy",
        'NE': "neutral",
        'SA': "sad",
        'SU': "surprised"
    }
    emotion_contents = dict(emotion)
    for key in emotion_contents.keys():
        emotion_contents[key] = []

    for folder in emotion.values():
        os.makedirs(outdir + '/' + folder, exist_ok=True)
    for (dirpath, dirnames, filenames) in os.walk(fname):
        for filen in [f for f in filenames if '.JPG' in f]:
            filepath = os.path.join(dirpath, filen)
            emotioncur = filen[4:6]
            # Skip Full right and full left images
            if filen[6:8] in ['FR', 'FL']:
                continue
            try:
                emotion_contents[emotioncur].append(filepath)
            except Exception as e:
                print('Warning: Odd file {}\nFailed due to {}'.format(
                      filepath, e))
                continue

    args = {}
    args['threshold'] = 0.0
    args['window'] = False
    args['ignore_multi'] = True
    args['grow'] = 10 
    args['resize'] = True
    args['row_resize'] = 512
    args['col_resize'] = 512
    args['min_proportion'] = 0.1

    eles = list(emotion_contents.items())

    with Pool(8) as p:
        p.map(_inner, [(k, v, emotion, outdir, args) for k, v in eles])

def main():
    get_folders_kdef()

if __name__ == '__main__':
    main()
