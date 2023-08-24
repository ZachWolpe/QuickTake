
'''
========================================================
Launch a WebCam and prediction in the video feed.

: zach wolpe, 24 Aug 23
: zach.wolpe@medibio.com.au
========================================================
'''

from quicktake import QuickTake

if __name__ == "__main__":
    qt = QuickTake(verbose=True)
    qt.launchStream()