import Face_Detector
import os
from multiprocessing import Pool

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
