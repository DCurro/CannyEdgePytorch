import torch
import torch.nn as nn
import numpy as np
from scipy.signal import gaussian


class Net(nn.Module):
    def __init__(self, threshold=10.0):
        super(Net, self).__init__()

        self.threshold = threshold

        filter_size = 5
        generated_filters = gaussian(filter_size,std=1.0).reshape([1,filter_size])

        self.gaussian_filter_horizontal = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1,filter_size), padding=(0,filter_size/2))
        self.gaussian_filter_horizontal.weight.data.copy_(torch.from_numpy(generated_filters))
        self.gaussian_filter_vertical = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(filter_size,1), padding=(filter_size/2,0))
        self.gaussian_filter_vertical.weight.data.copy_(torch.from_numpy(generated_filters.T))

        sobel_filter = np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])

        self.sobel_filter_horizontal = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=sobel_filter.shape, padding=sobel_filter.shape[0]/2)
        self.sobel_filter_horizontal.weight.data.copy_(torch.from_numpy(sobel_filter))
        self.sobel_filter_vertical = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=sobel_filter.shape, padding=sobel_filter.shape[0]/2)
        self.sobel_filter_vertical.weight.data.copy_(torch.from_numpy(sobel_filter.T))

        # filters were flipped manually
        filter_0 = np.array([   [ 0, 0, 0],
                                [ 0, 1,-1],
                                [ 0, 0, 0]])

        filter_45 = np.array([  [ 0, 0, 0],
                                [ 0, 1, 0],
                                [ 0, 0,-1]])

        filter_90 = np.array([  [ 0, 0, 0],
                                [ 0, 1, 0],
                                [ 0,-1, 0]])

        filter_135 = np.array([ [ 0, 0, 0],
                                [ 0, 1, 0],
                                [-1, 0, 0]])

        filter_180 = np.array([ [ 0, 0, 0],
                                [-1, 1, 0],
                                [ 0, 0, 0]])

        filter_225 = np.array([ [-1, 0, 0],
                                [ 0, 1, 0],
                                [ 0, 0, 0]])

        filter_270 = np.array([ [ 0,-1, 0],
                                [ 0, 1, 0],
                                [ 0, 0, 0]])

        filter_315 = np.array([ [ 0, 0,-1],
                                [ 0, 1, 0],
                                [ 0, 0, 0]])

        self.angle_filters_0_135 = [filter_0, filter_45, filter_90, filter_135]
        self.angle_filters_180_315 = [filter_180, filter_225, filter_270, filter_315]

        self.directional_filter_0 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=filter_0.shape, padding=filter_0.shape[0] / 2)
        self.directional_filter_1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=filter_0.shape, padding=filter_0.shape[0] / 2)


    def forward(self, img):
        img_r = img[:,0:1]
        img_g = img[:,1:2]
        img_b = img[:,2:3]

        blur_horizontal = self.gaussian_filter_horizontal(img_r)
        blurred_img_r = self.gaussian_filter_vertical(blur_horizontal)
        blur_horizontal = self.gaussian_filter_horizontal(img_g)
        blurred_img_g = self.gaussian_filter_vertical(blur_horizontal)
        blur_horizontal = self.gaussian_filter_horizontal(img_b)
        blurred_img_b = self.gaussian_filter_vertical(blur_horizontal)

        blurred_img = torch.stack([blurred_img_r,blurred_img_g,blurred_img_b],dim=1)
        blurred_img = torch.stack([torch.squeeze(blurred_img)])

        grad_x_r = self.sobel_filter_horizontal(blurred_img_r)
        grad_y_r = self.sobel_filter_vertical(blurred_img_r)
        grad_x_g = self.sobel_filter_horizontal(blurred_img_g)
        grad_y_g = self.sobel_filter_vertical(blurred_img_g)
        grad_x_b = self.sobel_filter_horizontal(blurred_img_b)
        grad_y_b = self.sobel_filter_vertical(blurred_img_b)

        # COMPUTE THICK EDGES

        grad_mag = torch.sqrt(grad_x_r**2 + grad_y_r**2)
        grad_mag += torch.sqrt(grad_x_g**2 + grad_y_g**2)
        grad_mag += torch.sqrt(grad_x_b**2 + grad_y_b**2)
        grad_orientation = (torch.atan2(grad_y_r+grad_y_g+grad_y_b, grad_x_r+grad_x_g+grad_x_b) * (180.0/3.14159))
        grad_orientation[grad_orientation<0] = 360+grad_orientation
        grad_orientation = grad_orientation % 180.0
        grad_orientation =  torch.round( grad_orientation / 45.0 ) * 45.0

        # THIN EDGES (NON-MAX SUPPRESSION)

        is_maxes = []

        for angle_idx, angle_pair in enumerate([(0.0,180.0), (45.0,225.0), (90.0,270.0), (135.0,315.0)]):
            angle_0 = angle_pair[0]
            angle_1 = angle_pair[1]

            angle_filter_0 = self.angle_filters_0_135[angle_idx]
            self.directional_filter_0.weight.data.copy_(torch.from_numpy(angle_filter_0))
            diff_0 = self.directional_filter_0(grad_mag)

            angle_filter_1 = self.angle_filters_180_315[angle_idx]
            self.directional_filter_1.weight.data.copy_(torch.from_numpy(angle_filter_1))
            diff_1 = self.directional_filter_1(grad_mag)

            diff_0[grad_orientation!=angle_0] = 0.0
            diff_1[grad_orientation!=angle_1] = 0.0

            diff = diff_0+diff_1

            is_maxes += [diff>0.0]

        is_max = torch.sum(torch.stack(is_maxes),dim=0)

        thin_edges = grad_mag.clone()
        thin_edges[is_max==0] = 0.0

        # THRESHOLD

        thresholded = thin_edges.clone()
        thresholded[thin_edges<self.threshold] = 0.0

        assert grad_mag.size() == grad_orientation.size() == thin_edges.size() == thresholded.size()

        early_threshold = grad_mag.clone()
        early_threshold[grad_mag<self.threshold] = 0.0

        return blurred_img, grad_mag, grad_orientation, thin_edges, thresholded, early_threshold


if __name__ == '__main__':
    Net()