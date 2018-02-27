from scipy.misc import imread, imsave
import torch
from torch.autograd import Variable
from net_canny import Net
from matplotlib import pyplot as plt
import matplotlib
import numpy as np


def canny(raw_img, use_cuda=False):
    img = torch.from_numpy(raw_img.transpose((2, 0, 1)))
    batch = torch.stack([img]).float()

    net = Net(threshold=3.0)
    if use_cuda:
        net.cuda()
    net.eval()

    data = Variable(batch)
    if use_cuda:
        data = Variable(batch).cuda()

    blurred_img, grad_mag, grad_orientation, thin_edges, thresholded, early_threshold = net(data)

    imsave('gradient_magnitude.png',grad_mag.data.cpu().numpy()[0,0])
    imsave('thin_edges.png', thresholded.data.cpu().numpy()[0, 0])
    imsave('thresholded.png', early_threshold.data.cpu().numpy()[0, 0])

    plt.imshow(thresholded.data.cpu().numpy()[0, 0], interpolation='None', cmap='gray')
    plt.show()

if __name__ == '__main__':
    img = imread('fb_profile.jpg') / 255.0

    # img = np.array([    [0, 0, 0, 0, 0],
    #                     [0, 0, 0, 0, 0],
    #                     [0, 0, 0, 0, 0],
    #                     [0, 0, 1, 0, 0],
    #                     [0, 0, 0, 0, 0]     ])
    # img = np.stack([img,img,img]).transpose([1,2,0])


    canny(img, use_cuda=False)