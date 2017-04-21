from numpy import linalg as la
from PIL import Image
import numpy as np


def loadImage(filename):

    im = Image.open(filename)

    im = im.convert("L")

    data = im.getdata()
    data = np.matrix(data)

    return data.reshape((512, 512))


def encry_watermark(alpha=1.0):

    data = loadImage("lena.jpg")

    watermk = loadImage("watermk.png")

    U, s, V = la.svd(data)
    S = np.diag(s)
    L = S + alpha * watermk
    U1, s1, V1 = la.svd(L)
    S1 = np.diag(s1)
    aw = np.dot(U, np.dot(S1, V))
    im = Image.fromarray(aw)
    im.show()
    im.convert('L').save("wmklena{:.1f}.jpg".format(alpha))

    return U1, V1, alpha, S


def decry_watermark(alpha):

    U1, V1, alpha, S = encry_watermark(alpha)
    data = loadImage("wmklena{:.1f}.jpg".format(alpha))
    Up, sp, Vp = la.svd(data)
    Sp = np.diag(sp)
    F = np.dot(U1, np.dot(Sp, V1))
    We = (F - S) / alpha
    im = Image.fromarray(We)
    im.show()
    im.convert('L').save("decryed_wmk{:.1f}.jpg".format(alpha))

decry_watermark(1.0)
decry_watermark(0.5)
decry_watermark(0.2)

