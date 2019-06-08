'''Train logistic regression to id pixels in dots

Focus on the perimeter in this version.

Gary Bishop June 2019'''

import pandas as pd
import cv2
import numpy as np
import math
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pylab as plt


def grabFrame(vc, start):
    '''
    Extract a frame and convert it to RGB float32 format
    '''
    vc.set(cv2.CAP_PROP_POS_FRAMES, start)
    rval, im = vc.read()
    assert rval
    im = cv2.cvtColor(im.astype(np.float32) / 255.0, cv2.COLOR_BGR2RGB)
    return im


# get the dots I manually located
dots = pd.read_csv('dots.csv')
dots.columns = ['video', 'fno', 'x', 'y', 'r']


# accumulate the features, labels, etc
features = []
labels = []
coords = []  # coordinate for plotting the errors
weights = []  # weight for each pixel

# group them by video
for video, group in dots.groupby('video'):
    # open the video
    vc = cv2.VideoCapture(video)
    # process each dot
    for dot in group.itertuples():
        # grab the frame before
        f0 = grabFrame(vc, dot.fno - 2)
        # grab the frame after
        f1 = grabFrame(vc, dot.fno)
        # get the center and radius
        xc, yc, r = dot.x, dot.y, dot.r
        # scan out a box with about 1/2 the pixels inside the dot
        side = math.ceil(1.25 * r)
        ylo = max(0, int(yc - side))
        yhi = min(f0.shape[0] - 1, int(math.ceil(yc + side)))
        xlo = max(0, int(xc - side))
        xhi = min(f0.shape[1] - 1, int(math.ceil(xc + side)))
        for y in range(ylo, yhi):
            for x in range(xlo, xhi):
                p0 = f0[y, x, :]
                p1 = f1[y, x, :]
                # try to eliminate effects of brighness
                # p0 = p0 / np.sum(p0)
                # p1 = p1 / np.sum(p0)
                # change in brightness
                change = np.sqrt(np.sum((p0 - p1) ** 2))
                # cosine of angle between them
                ca = np.dot(p0, p1) / (np.linalg.norm(p0) *
                                       np.linalg.norm(p1))
                # distance from the center of the dot
                d = np.sqrt((x - xc) ** 2 + (y - yc) ** 2)
                # weight pixels near the edge lower
                weight = 1
                # weight = (r - d) ** 2
                # label is 1 if on the edge
                label = abs(d - r) < 0.5
                # try some powers
                p = np.array([1, 2, 3])
                # accumulate into a list, make into an array later
                features.append(np.concatenate([p0,
                                                p1,
                                                f1[y, x + 1, :],
                                                f1[y, x - 1, :],
                                                f1[y + 1, x, :],
                                                f1[y - 1, x, :],
                                                # p1 - p0,
                                                p1 < p0,
                                                # np.outer(p0, p1).ravel(),
                                                # change ** p,
                                                # ca ** p,
                                                ]))
                labels.append(label)
                weights.append(weight)
                # for debugging
                coords.append([dot.fno, x, y])
features = np.array(features)
labels = np.array(labels)
weights = np.array(weights)
coords = np.array(coords)

# build and test the model
model = LogisticRegression()
x_train, x_test, y_train, y_test, w_train, w_test = train_test_split(features,
                                                    labels,
                                                    weights,
                                                    stratify=labels,
                                                    test_size=0.25)
model.fit(x_train, y_train, w_train)
print('score on test data')
print(model.score(x_test, y_test, w_test))


# where are the wrong ones
wrong = coords[labels != model.predict(features)]
print('number wrong per frame')
print(pd.DataFrame(wrong).groupby(0).count())


print('score on all data')
print(model.score(features, labels))


def showErrors(fno):
    predictions = model.predict(features)
    probs = model.predict_proba(features)
    iswrong = labels != predictions
    isf= coords[:, 0] == fno
    isb = iswrong & isf
    f = grabFrame(vc, fno)
    c = coords[isb]
    plt.imshow(f)
    plt.plot(c[:, 1], c[:, 2], '+')


wrong = features[iswrong]
