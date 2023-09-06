import torch
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import torch.optim as optim
import torch.nn as nn
import cv2
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import DataLoader





class GradCAM(object):
    def __init__(self, model, layers, image,ind,device):
        self.model = model
        self.gradients = -1
        self.activations = -1
        self.layers = layers
        self.img = image
        self.device = device
        self.index = ind
        self.img_list = list() 

        def forward_hook(module, input, output):
            self.activations = output
            return None

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]
            return None 


        self.layers.register_forward_hook(forward_hook)
        self.layers.register_backward_hook(backward_hook)

    def ImagePreprocess(self):

        img = self.img
        resize_size = [256]
        crop_size = [224]
        mean = [0.48235, 0.45882, 0.40784]
        std = [0.00392156862745098, 0.00392156862745098, 0.00392156862745098]
        transform = transforms.Compose([
        transforms.Resize(resize_size, interpolation=InterpolationMode.BILINEAR),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        ])

        img_bf = transform(img)
        
        ##Keep the IMG before standardization
        nor = transforms.Normalize(mean=mean, std=std)

        img_af = nor(img_bf)

        img_af = img_af.unsqueeze(0)    # C*H*W --> B*C*H*W
        img_af = img_af.to(self.device)

        return img_bf, img_af


    def forward(self, input):
        b, c, h, w = input.size()
        
        self.model.eval()

        prot = self.model(input)


        self.model.zero_grad()

        #index = torch.argmax(prot)
        score = prot[:, self.index]

        score.backward(retain_graph=False)

        b, k, h, w = self.gradients.size()
        
        ##The gradient matrix is changed into a row by row, and then the average is obtained in the row direction, which is equivalent to gap, [1,512]
        alpha = self.gradients.view(b, k, -1).mean(2)

        ##It is transformed into the form of [1, 512, 1, 1]
        weights = alpha.view(b, k, 1, 1)

        ##Add the multiplied [1, 512, 14, 14] on the first dimension (512) and keep the first dimension
        ## [1, 1, 14, 14]
        saliency_map = (weights*self.activations).sum(1, keepdim=True)
        saliency_map = F.relu(saliency_map)

        ##Get the same size as the input
        _, _, h, w = input.size()

        ##Sample the feature map to the same size
        saliency_map = F.upsample(saliency_map, size=(h, w), mode='bilinear', align_corners=False)

        ##Normalization
        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data

        mask = saliency_map.cpu().data.numpy()

        return mask

    def HeatMap(self, mask, img):
        
        ##When making Heatmap, the number of channels obtained is in the back
        heatmap = cv2.applyColorMap(np.uint8(255*mask.squeeze()), cv2.COLORMAP_JET)

        ##Turn to torch's numpy and adjust the number of channels to the front
        heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float().div(255)

        ##In the channel dimension, the three separated RGB matrixes are BGR to RGB
        b, g, r = heatmap.split(1)

        heatmap = torch.cat([r, g, b])

        result = heatmap+img.cpu()

        result = result.div(result.max()).squeeze()

        return img, heatmap, result

    def __call__(self):

        ##Image preprocessing
        img_bf, img_af = self.ImagePreprocess()

        ##Forward propagation and backward propagation
        mask = self.forward(img_af)

        ##Generate heat map
        img, heatmap, result = self.HeatMap(mask, img_bf)
        img = img.numpy().transpose(1,2,0)
        heatmap = heatmap.numpy().transpose(1,2,0)
        result = result.numpy().transpose(1,2,0)
        return img,heatmap,result,self.index
    
    # def plot(self):
    #     img, heatmap, result, index = self.__call__()
    #     fig = plt.figure(figsize=(10, 7))
    #     rows = 1
    #     columns = 3
    #     fig.add_subplot(rows, columns, 1)
    #     plt.imshow(img.numpy().transpose(1,2,0))
    #     plt.axis('off')
    #     plt.title("Raw image")

    #     fig.add_subplot(rows, columns, 2)
    #     plt.imshow(heatmap.numpy().transpose(1,2,0))
    #     plt.axis('off')
    #     plt.title("Heatmap")

    #     fig.add_subplot(rows, columns, 3)
    #     plt.imshow(result.numpy().transpose(1,2,0))
    #     plt.axis('off')
    #     plt.title("Result")

    #     plt.show()
