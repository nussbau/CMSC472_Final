import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define CNN-PP module for feature extraction
class CNN_PP(nn.Module):
    def __init__(self, num_outputs=16):
        super(CNN_PP, self).__init__()
        self.conv_layers = nn.Sequential(
            # 5 convolutional blocks with Leaky ReLU
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.01, inplace=True)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 128),  # This value might need to be adjusted based on the actual output of the conv layers
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(128, num_outputs)  # Output 15 parameters for the DIP module
        )

    def forward(self, x):
        # Downsample input to 256x256 before applying conv layers
        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x
    
    # Define the DIPModule with filter methods
class DIPModule(nn.Module):
    def create_gaussian_kernel(self, kernel_size=5, sigma=2):
        x_cord = torch.arange(kernel_size)
        x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

        mean = (kernel_size - 1) / 2.
        variance = sigma ** 2.

        gaussian_kernel = (1. / (2. * np.pi * variance)) * torch.exp(
            -torch.sum((xy_grid - mean) ** 2., dim=-1) /
            (2 * variance)
        )
        gaussian_kernel /= torch.sum(gaussian_kernel)

        # Adjust to [3, 1, 5, 5] for applying independently to each input channel
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(3, 1, 1, 1)  # Now [3, 1, 5, 5]

        return gaussian_kernel.to(dtype=torch.float32)


    def __init__(self):
        super(DIPModule, self).__init__()
        self.gaussian_kernel = self.create_gaussian_kernel().to(device)    


    def defog(self, img, w):
        dcp = -torch.nn.functional.max_pool2d(-img, kernel_size=3, stride=1, padding=1)
        AC = torch.topk(torch.flatten(dcp, 2), 1000, dim=-1)[0].mean(-1, keepdim=True).unsqueeze(-1)
        A = torch.topk(torch.flatten(dcp, 1), 10)[0].mean(-1).view((-1, 1, 1, 1)) 
        t = 1 - w[:, None, None, None]*(dcp/AC).min(1, keepdim=True)[0]
        return (img - (A*(1 - t)))/t

    def wb(self, img, rgb):
        rgb = F.sigmoid(rgb/3)*3
        return img * rgb.view(-1, 3, 1, 1)

    def gamma(self, img, gamma):
        img = F.relu(img)
        gamma = torch.abs(gamma)
        return torch.pow(img, gamma.view(-1, 1, 1, 1))

    def tone(self, img, t):
        num_tones = t.shape[0]
        j = torch.arange(num_tones, device=img.device).float()  # Shape [num_tones]

        # Expand 'img' to add a tone dimension: [batch, channels, height, width, num_tones]
        img_expanded = img.unsqueeze(-1).expand(-1, -1, -1, -1, num_tones)

        # Calculate tone transformation
        tone_transform = (1 / t.sum()) * torch.clamp(num_tones * img_expanded - j, 0, 1).sum(dim=4)

        return tone_transform

    def contrast(self, img, alpha, beta):
        return alpha.view(-1, 1, 1, 1)*img + beta.view(-1, 1, 1, 1)

    def sharpen(self, img, lambd):
        filtered_img = F.conv2d(img, self.gaussian_kernel, padding='same', groups=3)
        return img + lambd.view(-1, 1, 1, 1) * (img - filtered_img)

    def forward(self, img, parameters):
        img = self.defog(img, parameters[:, 0])
        img = self.wb(img, parameters[:, 1:4])
        img = self.gamma(img, parameters[:, 4])
        img = self.tone(img, parameters[:, 5:-4])
        img = self.contrast(img, parameters[:, -4], parameters[:, -3])        
        img = self.sharpen(img, parameters[:, -2])
        return img
    
class IntegratedModel(nn.Module):
    def __init__(self):
        super(IntegratedModel, self).__init__()        
        self.cnn_pp = CNN_PP(num_outputs=16)
        self.dip_module = DIPModule()

    def forward(self, x):
        parameters = self.cnn_pp(x)
        processed_img = self.dip_module(x, parameters)
        return processed_img, parameters
    

if __name__ == '__main__':
    from PIL import Image
    from torchvision import transforms
    from os import listdir
    from random import shuffle

    img_paths = listdir("datasets/underwater/images/train/")
    shuffle(img_paths)

    for img_path in img_paths:
        in_image = transforms.ToTensor()(Image.open(f"datasets/underwater/images/train/{img_path}")).unsqueeze(0).to(device)
        model = torch.load('DIP_Model_Underwater.pth')
        #optim = torch.optim.Adam(model.parameters(), lr=3e-4)
        
        '''for i in range(1000):
            optim.zero_grad()
            out = model(in_image)
            loss = ((in_image - out)**2).mean()
            
            loss.backward()
            print(loss)
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optim.step()'''
        
        import matplotlib.pyplot as plt
        model.eval()
        out, parameters = model(in_image)
        out -= out.min()
        #out = (out-out.min())/(1.15*(out.max()-out.min()))
        f, axarr = plt.subplots(2)
        '''axarr[0].imshow(in_image.cpu().detach().squeeze().permute((1, 2, 0)).numpy())
        axarr[1].imshow(out.cpu().detach().squeeze().permute((1, 2, 0)).numpy())
        plt.show()'''
        Image.fromarray((out.cpu().detach().squeeze().permute((1, 2, 0)).numpy()*255).astype(np.uint8)).save("results_dip/" + img_path.replace('.jpg', '') + "_dip.jpg")
        
        
