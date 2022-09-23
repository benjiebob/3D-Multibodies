import numpy as np
import copy
import io

import matplotlib.pyplot as plt

from PIL import Image

import torch

from visdom import Visdom

viz = None

def get_visdom_env(cfg):
    if len(cfg.visdom_env)==0:
        visdom_env = cfg.exp_dir
    else:
        visdom_env = cfg.visdom_env
    return visdom_env

def get_visdom_connection(server='http://localhost',port=8097): 
    global viz
    if viz is None:    
        viz = Visdom(server=server,port=port)
    return viz

def denorm_image_trivial(im):
    im = im - im.min()
    im = im / (im.max()+1e-7)
    return im

def ensure_im_width(img,basewidth):
    # basewidth = 300
    # img = Image.open('somepic.jpg')
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((basewidth,hsize), Image.ANTIALIAS)
    return img

def denorm_image_trivial(im):
    im = im - im.min()
    im = im / (im.max()+1e-7)
    return im


def fig2data(fig, size=None):
    """Convert a Matplotlib figure to a numpy array

    Based on the ICARE wiki.

    Args:
        fig (matplotlib.Figure): a figure to be converted

    Returns:
        (ndarray): an array of RGB values
    """
    # TODO(samuel): convert figure to provide a tight fit in image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    im = Image.open(buf).convert('RGB')
    if size:
        im = im.resize(size)
    # fig.canvas.draw()
    # import ipdb ; ipdb.set_trace()
    # # w,h = fig.canvas.get_width_height()
    # # buf = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8)
    # buf.shape = (h, w, 3)
    # return buf
    return np.array(im)
