''' Find center and size of dots for training

Gary Bishop June 2019
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os.path as osp
import sys


if len(sys.argv) > 1:
    video = sys.argv[1]
else:
    video = osp.join('videos', '12.mp4')

vc = cv2.VideoCapture(video)

# bash their mappings
keys = [
    'fullscreen',
    'home',
    'back',
    'forward',
    'pan',
    'zoom',
    'save',
    'quit',
    'grid',
    'yscale',
    'xscale',
    'all_axes'
]
for key in keys:
    plt.rcParams['keymap.' + key] = ''


def grabFrame(vc, start):
    '''
    Extract a frame and convert it to LAB float32 format
    '''
    vc.set(cv2.CAP_PROP_POS_FRAMES, start)
    rval, im = vc.read()
    assert rval
    im = cv2.cvtColor(im.astype(np.float32) / 255.0, cv2.COLOR_BGR2RGB)
    return im


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


# frames = [f for i, f in enumerate(grabFrames(0)) if i % 2 == 0]


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
        fig = plt.figure(num)

    args = dict(kwargs)
    if 'title' in args:
        del args['title']

    if len(im.shape) == 2:
        args['cmap'] = 'gray'

    plt.imshow(im, **args)
    if 'title' in kwargs:
        plt.title(kwargs['title'])

    return fig


class Handler:
    '''Handle events on the figure'''
    def __init__(self, vc, name):
        self.vc = vc
        self.fno = 0
        self.nframes = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame = grabFrame(self.vc, 0)
        self.fig = show(self.frame)
        self.xlim = (-0.5, self.frame.shape[1] - 0.5)
        self.ylim = (self.frame.shape[0] - 0.5, -0.5)
        self.ax = self.fig.axes[0]
        self.center = [0, 0]
        self.radius = 10
        self.cid = self.fig.canvas.mpl_connect('button_press_event',
                                               self.onclick)
        self.kid = self.fig.canvas.mpl_connect('key_press_event',
                                               self.onpress)
        self.circ = None
        self.dots = []
        self.name = name

    def __del__(self):
        print('cleanup')
        self.fig.canvas.mpl_disconnect(self.cid)
        self.fig.canvas.mpl_disconnect(self.kid)

    def onclick(self, event):
        '''handle click events'''
        print('%s click: button=%d, xdata=%f, ydata=%f' %
              ('double' if event.dblclick else 'single', event.button,
               event.xdata, event.ydata))
        self.center = (int(event.xdata) + 0.5, int(event.ydata) + 0.5)
        self.showCircle()
        self.update()

    def onpress(self, event):
        '''handle keyboard events'''
        if event.key == 'h':
            self.showCircle(dx=-1)
        elif event.key == 'j':
            self.showCircle(dy=1)
        elif event.key == 'k':
            self.showCircle(dy=-1)
        elif event.key == 'l':
            self.showCircle(dx=1)
        elif event.key == 'H':
            self.showCircle(dx=-10)
        elif event.key == 'J':
            self.showCircle(dy=10)
        elif event.key == 'K':
            self.showCircle(dy=-10)
        elif event.key == 'L':
            self.showCircle(dx=10)
        elif event.key == 'ctrl+h':
            self.showCircle(dx=-0.1)
        elif event.key == 'ctrl+j':
            self.showCircle(dy=0.1)
        elif event.key == 'ctrl+k':
            self.showCircle(dy=-0.1)
        elif event.key == 'ctrl+l':
            self.showCircle(dx=0.1)
        elif event.key == 'a':
            self.showCircle(dr=1)
        elif event.key == 's':
            self.showCircle(dr=-1)
        elif event.key == 'right':
            self.showFrame(df=1)
        elif event.key == 'left':
            self.showFrame(df=-1)
        elif event.key == 'z':
            plt.xlim(self.center[0] - 2 * self.radius,
                     self.center[0] + 2 * self.radius)
            plt.ylim(self.center[1] + 2 * self.radius,
                     self.center[1] - 2 * self.radius)
        elif event.key == 'Z':
            plt.xlim(*self.xlim)
            plt.ylim(*self.ylim)
        elif event.key == 'r':
            self.record()
        else:
            print('press', event.key)

        self.update()

    def showCircle(self, dx=0, dy=0, dr=0):
        '''Draw a circle on the graph'''
        self.center = self.center[0] + dx, self.center[1] + dy
        self.radius += dr
        if not self.circ:
            self.circ = mpl.patches.Circle(self.center, self.radius,
                                           color=(1, 0, 0), fill=False)
            self.ax.add_artist(self.circ)
        else:
            self.circ.center = self.center
            self.circ.radius = self.radius

    def showFrame(self, df=0):
        '''Show the current frame'''
        self.fno = max(0, min(self.nframes - 1, self.fno + df * 2))
        self.frame = grabFrame(self.vc, self.fno)
        self.fig.axes[0].get_images()[0].set_data(self.frame)
        plt.title(f'frame {self.fno}')

    def update(self):
        '''update the figure'''
        self.fig.canvas.draw()

    def record(self):
        '''record the current dot'''
        with open('dots.csv', 'at') as fp:
            line = (f'{self.name},'
                    f'{self.fno},'
                    f'{self.center[0]:.1f},'
                    f'{self.center[1]:.1f},'
                    f'{self.radius:.1f}\n')
            fp.write(line)


handler = None
handler = Handler(vc, video)


plt.show()
