from scipy.misc import imread, imsave
import torch
from torch.autograd import Variable
from net_canny import Net


def canny(raw_img, use_cuda=False):
    img = torch.from_numpy(raw_img.transpose((2, 0, 1)))
    batch = torch.stack([img, img]).float()

    net = Net(threshold=3.0, use_cuda=use_cuda)
    if use_cuda:
        net.cuda()
    net.eval()

    data = Variable(batch)
    if use_cuda:
        data = Variable(batch).cuda()

    blurred_img, grad_mag, grad_orientation, thin_edges, thresholded, early_threshold = net(data)

    for i in range(batch.shape[0]):
        imsave('gradient_magnitude_{}.png'.format(i),grad_mag[i].data.cpu().numpy()[0])
        imsave('thin_edges_{}.png'.format(i), thresholded[i].data.cpu().numpy()[0])
        imsave('final_{}.png'.format(i), (thresholded[i].data.cpu().numpy()[0] > 0.0).astype(float))
        imsave('thresholded_{}.png'.format(i), early_threshold[i].data.cpu().numpy()[0])


if __name__ == '__main__':
    img = imread('fb_profile.jpg') / 255.0

    # canny(img, use_cuda=False)
    canny(img, use_cuda=True)
