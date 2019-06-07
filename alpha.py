'''Experiment with alpha-blending on the Rett videos

June 2019'''

%matplotlib


import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib as mpl
from itertools import tee


video = 'videos/12.mp4'


vc = cv2.VideoCapture(video)


def grabFrames(start, end=None):
    '''
    Extract a frame and convert it to LAB float32 format
    '''
    if not end:
        end = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
    vc.set(cv2.CAP_PROP_POS_FRAMES, start)
    for fno in range(start, end):
        rval, im = vc.read()
        if not rval:
            break
        im = cv2.cvtColor(im.astype(np.float32) / 255.0, cv2.COLOR_BGR2RGB)
        yield im


frames = [f for i, f in enumerate(grabFrames(116, 126)) if i % 2 == 0]


def show(im, num=1, **kwargs):
    '''Show images actual size unless it is tiny
    '''
    height, width = im.shape[:2]
    if height > 50 and width > 50:
        dpi = 100
        margin = 50
        figsize = ((width + 2 * margin) / dpi,
                   (height + 2 * margin) / dpi)  # inches
        left = margin / dpi / figsize[0]  # axes ratio
        bottom = margin / dpi / figsize[1]

        fig = plt.figure(num=num, figsize=figsize, dpi=dpi)
        fig.subplots_adjust(left=left, bottom=bottom,
                            right=1.0 - left, top=1.0 - bottom)
    else:
        plt.figure(num)

    args = dict(kwargs)
    if 'title' in args:
        del args['title']

    if len(im.shape) == 2:
        args['cmap'] = 'gray'

    plt.imshow(im, **args)
    if 'title' in kwargs:
        plt.title(kwargs['title'])


def findContours(before, after):
    '''
    Return the contours of changes
    '''
    d = (before - after)[:, :, 0]
    _, t = cv2.threshold(d, 0.3, 255, cv2.THRESH_BINARY)
    t = t.astype(np.uint8)
    # dilate a bit to fill in noise
    t = cv2.dilate(t, np.ones((3, 3), dtype=np.uint8), iterations=1)
    # erode to get back
    t = cv2.erode(t, np.ones((3, 3), dtype=np.uint8), iterations=1)

    im2, contours, hierarchy = cv2.findContours(t, cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    # filter out tiny ones
    minArea = np.pi * 4**2
    contours = [contour for contour in contours
                if minArea < cv2.contourArea(contour)]

    return contours


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


f = 1
for a, b in pairwise(frames):
    contours = findContours(a, b)
    o = b.copy()
    cv2.drawContours(o, contours, -1, (1, 0, 0), 2)
    show(o, num=f)
    f += 1
