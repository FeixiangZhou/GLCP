import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from skimage.morphology import skeletonize_3d,skeletonize
import torch.nn.functional as F
import random
from scipy.spatial.distance import cdist

from sklearn.cluster import DBSCAN
import random
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt
from nnunetv2.training.loss.soft_skeleton import SoftSkeletonize
from skimage import io
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


output_folder_vis_results = '/media/fxzhou/DATADRIVE1/project/nnUNet-2/nnUNet_results/Dataset106_STARE/vis_results/'
os.makedirs(output_folder_vis_results, exist_ok=True)

    
class UNet2D_refine(nn.Module):

    def __init__(self, num_classes=2, ):
        super(UNet2D_refine, self).__init__()
        use_bias = True
        # self.conv11 = nn.Conv2d(1, 8, kernel_size=3, padding=1, bias=use_bias)
        # self.conv12 = nn.Conv2d(8, 8, kernel_size=3, padding=1, bias=use_bias)
        self.down1 = nn.Conv2d(8, 16, kernel_size=3, padding=1, stride=2, bias=use_bias)
        self.conv21 = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=use_bias)
        self.down2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2, bias=use_bias)
        self.conv31 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=use_bias)
        self.down3 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2, bias=use_bias)
        self.conv41 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=use_bias)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv32 = nn.Conv2d(96, 32, kernel_size=3, padding=1, bias=use_bias)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv22 = nn.Conv2d(48, 16, kernel_size=3, padding=1, bias=use_bias)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv13 = nn.Conv2d(24, 8, kernel_size=3, padding=1, bias=use_bias)
        # self.conv14 = nn.Conv2d(8, num_classes, kernel_size=1, padding=0, bias=use_bias)

    def forward(self, x):
        # x1 = F.relu(self.conv11(x))
        # x1 = F.relu(self.conv12(x1))
        x1=x
        x2 = self.down1(x1)
        x2 = F.relu(self.conv21(x2))
        x3 = self.down2(x2)
        x3 = F.relu(self.conv31(x3))
        x4 = self.down3(x3)
        x4 = F.relu(self.conv41(x4))

        x3 = torch.cat([self.up3(x4), x3], dim=1)
        x3 = F.relu(self.conv32(x3))
        x2 = torch.cat([self.up2(x3), x2], dim=1)
        x2 = F.relu(self.conv22(x2))
        x1 = torch.cat([self.up1(x2), x1], dim=1)
        x1 = F.relu(self.conv13(x1))
        # x = self.conv14(x1)
        return x

# Basic block: Conv2d -> InstanceNorm2d -> LeakyReLU
class ConvDropoutNormReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dropout=False):
        super(ConvDropoutNormReLU, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        ]
        if dropout:
            layers.insert(0, nn.Dropout2d(0.3))
        self.all_modules = nn.Sequential(*layers)

    def forward(self, x):
        return self.all_modules(x)


# Stacked Conv Block: Multiple ConvDropoutNormReLU layers
class StackedConvBlocks(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=2, stride=1):
        super(StackedConvBlocks, self).__init__()
        layers = []
        layers.append(ConvDropoutNormReLU(in_channels, out_channels, stride=stride))
        for _ in range(1, num_layers):
            layers.append(ConvDropoutNormReLU(out_channels, out_channels))
        self.convs = nn.Sequential(*layers)

    def forward(self, x):
        return self.convs(x)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


# Encoder block: Downsample using Conv2d with stride > 1
class PlainConvEncoder(nn.Module):
    def __init__(self, in_channels):
        super(PlainConvEncoder, self).__init__()
        self.stages = nn.Sequential(
            StackedConvBlocks(in_channels, 32, stride=1),
            StackedConvBlocks(32, 64, stride=2),
            StackedConvBlocks(64, 128, stride=2),
            StackedConvBlocks(128, 256, stride=2),
            StackedConvBlocks(256, 512, stride=2),

            StackedConvBlocks(512, 512, stride=2),
            StackedConvBlocks(512, 512, stride=2),
            StackedConvBlocks(512, 512, stride=2)


        )

    def forward(self, x):
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features


class PlainConvEncoder_refine(nn.Module):
    def __init__(self, in_channels):
        super(PlainConvEncoder_refine, self).__init__()
        self.stages = nn.Sequential(
            StackedConvBlocks(in_channels, 32, stride=1),
            StackedConvBlocks(32, 64, stride=2),
            # StackedConvBlocks(64, 128, stride=2),
            # StackedConvBlocks(128, 256, stride=2),
            # StackedConvBlocks(256, 512, stride=2),
        )

    def forward(self, x):
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features

    

# Decoder block: Upsample and concatenate with encoder feature maps
class UNetDecoder(nn.Module):
    def __init__(self):
        super(UNetDecoder, self).__init__()

        self.upconv10 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.conv10 = StackedConvBlocks(1024, 512)

        self.upconv11 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.conv11 = StackedConvBlocks(1024, 512)

        self.upconv12 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.conv12 = StackedConvBlocks(1024, 512)


        self.upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv1 = StackedConvBlocks(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = StackedConvBlocks(256, 128)

        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv3 = StackedConvBlocks(128, 64)

        self.upconv4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv4 = StackedConvBlocks(64, 32)

    def forward(self, x, encoder_features):

        x = self.upconv10(x)
        x = torch.cat([x, encoder_features[6]], dim=1)
        x = self.conv10(x)

        x = self.upconv11(x)
        x = torch.cat([x, encoder_features[5]], dim=1)
        x = self.conv10(x)

        x = self.upconv12(x)
        x = torch.cat([x, encoder_features[4]], dim=1)
        x = self.conv12(x)


        x = self.upconv1(x)
        x = torch.cat([x, encoder_features[3]], dim=1)
        x = self.conv1(x)

        x = self.upconv2(x)
        x = torch.cat([x, encoder_features[2]], dim=1)
        x = self.conv2(x)

        x = self.upconv3(x)
        x = torch.cat([x, encoder_features[1]], dim=1)
        x = self.conv3(x)

        x = self.upconv4(x)
        x = torch.cat([x, encoder_features[0]], dim=1)
        x = self.conv4(x)

        return x
    



class Unet2D(nn.Module):
    def __init__(self, in_channels=3, out_channels=2):
        super(Unet2D, self).__init__()
        self.counter = 0  

        self.encoder = PlainConvEncoder(in_channels)
        self.decoder = UNetDecoder()
        # self.decoder_skeleton = UNetDecoder()
        # self.decoder_criticalregion = UNetDecoder()

        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)
        self.final_conv_skeleton = nn.Conv2d(32, 2, kernel_size=1)
        self.final_conv_criticalregion = nn.Conv2d(32, 2, kernel_size=1)


       
        self.final_conv_refine = nn.Conv2d(32, out_channels, kernel_size=1)

        # self.satt = SpatialAttention( kernel_size=3)

        self.refine_conv_seg = nn.Conv2d(2, 32, kernel_size=1)
        # self.m_skeletonize = SoftSkeletonize(num_iter=10)

    def forward(self, x, mask=None, train=None):
        self.counter += 1

        # Encoding path
        x_ori = x
        encoder_features = self.encoder(x)
        # Bottleneck feature map
        bottleneck = encoder_features[-1]
        # Decoding path
        x = self.decoder(bottleneck, encoder_features)
        x_decoder = x

        x_skeleton = self.final_conv_skeleton(x)
        x_criticalregion = self.final_conv_criticalregion(x)
        x = self.final_conv(x)

        if mask is not None:
            y_pred_fore = x[:, 1:]
            y_pred_fore = torch.max(y_pred_fore, dim=1, keepdim=True)[0]  # C foreground channels -> 1 channel
            y_pred_binary = torch.cat([x[:, :1], y_pred_fore], dim=1)
            y_prob_binary = torch.softmax(y_pred_binary, 1)
            y_pred_prob = y_prob_binary[:, 1]  # [2,32,32,32]
            y_pred_hard = (y_pred_prob > 0.5).float()  # [2,32,32,32]

            # skeleton_batch_soft = self.m_skeletonize(y_pred_hard.unsqueeze(1))#[2, 1, 32, 32, 32]
            # print("skeleton_batch_soft-----", torch.min(skeleton_batch_soft), torch.max(skeleton_batch_soft))
            # save_binary_array_as_image_skimage(skeleton_batch_soft.squeeze(1), "skeleton_batch_soft", self.counter)

            skeleton_batch, endpoints_batch = compute_skeleton_and_endpoints(y_pred_hard)
            # print("skeleton_batch-----", torch.min(skeleton_batch), torch.max(skeleton_batch))
            # save_binary_array_as_image_skimage(skeleton_batch, "skeleton_batch",self.counter)
            # skeleton_batch--- torch.Size([2, 32, 32, 32])
            # endpoints_batch--- torch.Size([2, 1, 32, 32, 32])

            # for gt
            mask_gt = mask.gt(0).squeeze(1).float()
            skeleton_batch_gt, endpoints_batch_gt = compute_skeleton_and_endpoints(mask_gt)

            # skeleton consistency
            prob_skeleton = y_prob_binary.clone()  # [2, 2, 32, 32, 32]
            # prob_skeleton[:, 0, :, :][skeleton_batch == 0] = 0.7311  # ([2, 2, 32, 32, 32])
            # prob_skeleton[:, 1, :, :][skeleton_batch == 0] = 0.2689  # ([2, 2, 32, 32, 32])

            prob_skeleton[:, 0, :, :][skeleton_batch == 0] = 1 
            prob_skeleton[:, 1, :, :][skeleton_batch == 0] = 0 

            # prob_skeleton[:, 0, :, :] = prob_skeleton[:, 0, :, :] * (skeleton_batch != 0).float() + 0.7311  * (skeleton_batch == 0).float()
            # prob_skeleton[:, 1, :, :] = prob_skeleton[:, 1, :, :] * (skeleton_batch != 0).float() +  0.2689 * (skeleton_batch == 0).float()

            #one
            # prob_skeleton= y_pred_prob * (skeleton_batch != 0).float()

            # two
            # prob_skeleton_0 = y_pred_prob * (skeleton_batch_soft.squeeze(1))
            # prob_skeleton_1 = 1 - prob_skeleton_0
            # prob_skeleton = torch.stack([prob_skeleton_1, prob_skeleton_0], dim=1)

            endpoint_features_eachbatch = []
            endpoint_mask_eachbatch = []
            mask_criticalregion_eachbatch = []

            for i in range(y_pred_hard.size(0)):
                endpoints_np = endpoints_batch[i, 0].cpu().numpy()  # [32, 32, 32]
                coords = np.argwhere(endpoints_np == 1)
                # print("len(coords)---", len(coords))
                # print("coords---", coords.shape)
                # print("coords---", coords)


                endpoints_gt_np = endpoints_batch_gt[i, 0].cpu().numpy()  # [32, 32, 32]
                coords_gt = np.argwhere(endpoints_gt_np == 1)
                # print("coords_gt---", coords_gt.shape) 
                # print("coords_gt---", coords_gt) 

                mask_criticalregion = torch.zeros_like(mask[i, :, :, :])

                if len(coords) > 0:
                    if len(coords_gt) > 0:
                        selected_endpoints = endpoints_select(coords, coords_gt)
                        selected_endpoints_FP = endpoints_select(coords_gt, coords)

                        if  (len(selected_endpoints)==0 and len(selected_endpoints_FP)==0  ):
                            selected_endpoints = np.array([]) 

                        elif (len(selected_endpoints)==0 and len(selected_endpoints_FP) > 0  ):
                            selected_endpoints = selected_endpoints_FP
                        elif  (len(selected_endpoints) > 0 and len(selected_endpoints_FP) == 0  ):
                            selected_endpoints = selected_endpoints
                        elif  (len(selected_endpoints) > 0 and len(selected_endpoints_FP) > 0  ):
                            print("selected_endpoints  ---", selected_endpoints.shape)
                            print("selected_endpoints_FP  ---", selected_endpoints_FP.shape)
                            selected_endpoints = np.concatenate((selected_endpoints,selected_endpoints_FP),0)


                        if(len(selected_endpoints) > 0):
                            print("selected_endpoints before cluster  ---", selected_endpoints.shape) #[496,3]
                            selected_endpoints = DBSCAN_2d(selected_endpoints, endpoints_np)
                            print("selected_endpoints after cluster  ---", selected_endpoints.shape) #[496,3]
                            # print("selected_endpoints after cluster  ---", selected_endpoints) 


                            # if (len(selected_endpoints) >=5): 
                            #     indices = np.random.choice(selected_endpoints.shape[0], size=5, replace=False)
                            #     selected_endpoints = selected_endpoints[indices]
                            
                        else:
                            print("no selected_endpoints after endpoints_select  ---")
                            mask_criticalregion_eachbatch.append(mask_criticalregion)
                            continue
                    else:
                        selected_endpoints = coords
                else:
                    selected_endpoints = coords_gt
                    # mask_criticalregion_eachbatch.append(mask_criticalregion)
                    # continue


                endpoint_features_eachsample = []
                endpoint_mask_eachsample = []

                for (y, x1) in selected_endpoints:
                    prob_map = x[i, :, :, :]
                    skeleton_map = x_skeleton[i, :, :, :]
                    imag_map = x_ori[i, :, :, :]
                    mask_map = mask[i, :, :, :]

                    scale_factor = 16

                    wy = y_pred_hard.size(1) // scale_factor
                    wx = y_pred_hard.size(2) // scale_factor

                    inter = 1

                    ymin, ymax = max(0, y - wy // 2), min(y_pred_hard.size(1), y + wy // 2)
                    xmin, xmax = max(0, x1 - wx // 2), min(y_pred_hard.size(2), x1 + wx // 2)


                    if ymax - ymin < wy:
                        if ymin == 0:
                            ymax = min(ymin + wy, y_pred_hard.size(1))
                        else:
                            ymin = max(0, ymax - wy)

                    if xmax - xmin < wx:
                        if xmin == 0:
                            xmax = min(xmin + wx, y_pred_hard.size(2))
                        else:
                            xmin = max(0, xmax - wx)

                    prob_region = prob_map[:,  ymin:ymax:inter, xmin:xmax:inter]  
                    prob_region = prob_region.unsqueeze(0)  

                    skeleton_region = skeleton_map[:, ymin:ymax:inter, xmin:xmax:inter]
                    skeleton_region = skeleton_region.unsqueeze(0) 

                    img_region = imag_map[:, ymin:ymax:inter, xmin:xmax:inter] 
                    img_region = img_region.unsqueeze(0) 

                    mask_region = mask_map[:, ymin:ymax:inter, xmin:xmax:inter]
                    mask_region = mask_region.unsqueeze(0) 

                    # feature fusion
                    # prob_region = prob_region + 0* skeleton_region
                    prob_region = torch.cat((img_region, prob_region), 1)

                    mask_criticalregion[:, ymin:ymax, xmin:xmax] = 1
                    # print("mask_criticalregion----", mask_criticalregion.shape)  # torch.Size([1， 1，10，10,10])
                    # print("mask_criticalregion----", mask_criticalregion.sum())

                    endpoint_features_eachsample.append(prob_region)
                    endpoint_mask_eachsample.append(mask_region)

                if (len(endpoint_features_eachsample) > 0):
                    endpoint_features_eachsample = torch.cat(endpoint_features_eachsample)
                    # print("endpoint_features_eachsample----", endpoint_features_eachsample.shape)  #[5, 64, 10, 10, 10]
                    endpoint_mask_eachsample = torch.cat(endpoint_mask_eachsample)

                    endpoint_features_eachbatch.append(endpoint_features_eachsample)
                    endpoint_mask_eachbatch.append(endpoint_mask_eachsample)
                    mask_criticalregion_eachbatch.append(mask_criticalregion)

            if (len(endpoint_features_eachbatch) > 0):
                endpoint_features_eachbatch = torch.cat(endpoint_features_eachbatch)
                print("endpoint_features_eachbatch----",endpoint_features_eachbatch.shape)  # torch.Size([ 10, 64, 10, 10, 10])
                endpoint_mask_eachbatch = torch.cat(endpoint_mask_eachbatch)
                print("endpoint_mask_eachbatch----", endpoint_mask_eachbatch.shape)  # torch.Size([ 10, 1, 10, 10, 10])


                mask_criticalregion_eachbatch = torch.stack(mask_criticalregion_eachbatch)
                # print("mask_criticalregion_eachbatch----", mask_criticalregion_eachbatch.shape)  # torch.Size([ 2, 1, 32, 32, 32])

            '''
            refine module
            '''

            # prob_map_refine = torch.cat((x, x_criticalregion), 1)
            # prob_map_refine = torch.cat((x, x_criticalregion, x_skeleton), 1)


            x_refine= self.refine_conv_seg(x)
            # x_skeleton_refine= self.refine_conv_seg(x_skeleton)
            # x_criticalregion= self.refine_conv_break(x_criticalregion)


            probs_weight = torch.softmax(x_criticalregion, dim=1)
            probs_weight = probs_weight[:,1].unsqueeze(1)
            prob_map_refine = x_refine * probs_weight


            skeleton_weight = torch.softmax(x_skeleton, dim=1)
            skeleton_weight = skeleton_weight[:,1].unsqueeze(1)
            skeleton_map_refine = x_refine * skeleton_weight

            endpoint_features_refine = prob_map_refine + skeleton_map_refine  + x_refine 

            endpoint_features_refine = self.final_conv_refine(endpoint_features_refine)
            print("endpoint_features_eachbatch_after_refine----",endpoint_features_refine.shape)  # torch.Size([ 10, 2, 10, 10, 10])


            x_criticalregion_train = x_criticalregion.clone()
            x_criticalregion_train = torch.softmax(x_criticalregion_train, 1)  # [batchsize, 2， 32,32,32】
            x_criticalregion_np = x_criticalregion_train.detach().cpu().numpy()
            print('coords_criticalregion_train max----', np.max(x_criticalregion_np[:, 1, :, :], axis=(1, 2)))

            if isinstance(mask_criticalregion_eachbatch, list):  # all coords in a batch is 0
                print("************mask_criticalregion_eachbatch is list*****************")
                mask_criticalregion_eachbatch = torch.stack(mask_criticalregion_eachbatch)
            print("mask_criticalregion_eachbatch----", mask_criticalregion_eachbatch.shape)  # torch.Size([ 10, 1, 32, 32, 10])

            # return x, x_skeleton, x_criticalregion, mask_criticalregion_eachbatch, endpoint_features_refine, endpoint_mask_eachbatch, prob_skeleton
            return x, x_skeleton, x_criticalregion, mask_criticalregion_eachbatch, endpoint_features_refine, skeleton_batch_gt, prob_skeleton

        else:
            print("----------------------------test-------------------------------")
            #vis output------------------
            folder_name_ske = 'output_ske/'
            folder_name_seg= 'output_seg/'
            folder_name_criticalregion = 'output_critical/'

            folder_name_criticalregion_visatt = 'output_critical_att/'
            folder_name_ske_visatt = 'output_ske_att/'
            folder_name_ske_crit_visatt = 'output_ske_crit_att/'

            folder_name_crit_pred_visatt = 'output_crit_pred_att/'


            y_pred_fore = x[:, 1:]
            y_pred_fore = torch.max(y_pred_fore, dim=1, keepdim=True)[0]  # C foreground channels -> 1 channel
            y_pred_binary = torch.cat([x[:, :1], y_pred_fore], dim=1)
            y_prob_binary = torch.softmax(y_pred_binary, 1)
            y_pred_prob = y_prob_binary[:, 1]  # [2,32,32,32]
            y_pred_hard = (y_pred_prob > 0.5).float()  # [2,32,32,32]



            x_skeleton_prob = torch.softmax(x_skeleton, 1)
            x_skeleton_prob = x_skeleton_prob[:, 1]  # [1,32,32,32]
            x_skeleton_img = (x_skeleton_prob > 0.5).float() 
            # save_binary_array_as_image_skimage(x_skeleton_img, folder_name_ske, "output_skeleton",self.counter)
            save_binary_array_as_image_skimage(y_pred_hard, folder_name_seg, "output_seg",self.counter)

            x_criticalregion_prob = torch.softmax(x_criticalregion, 1)
            x_criticalregion_prob = x_criticalregion_prob[:, 1] 
            x_criticalregion_img = (x_criticalregion_prob > 0.5).float() 
            # save_binary_array_as_image_skimage(x_criticalregion_img, folder_name_criticalregion, "output_criticalregion",self.counter)



            # x_criticalregion = torch.softmax(x_criticalregion, 1)
            # x_criticalregion = x_criticalregion[:,1].unsqueeze(1)

            # prob_map_refine = torch.cat((x, x_criticalregion), 1) 
            # prob_map_refine = torch.cat((x, x_criticalregion, x_skeleton), 1)
            # prob_map_refine = x + x_criticalregion
            x_refine= self.refine_conv_seg(x)


            probs_weight = torch.softmax(x_criticalregion, dim=1)
            probs_weight = probs_weight[:,1].unsqueeze(1) #[1,1, 512,512]
            prob_map_refine = x_refine * probs_weight
            # print("probs_weight_crit----",probs_weight.shape)  # torch.Size([ 1, 1, 512, 512])
            # vis_attention(probs_weight[0][0], folder_name_criticalregion_visatt, "output_criticalregion_att", self.counter)


            skeleton_weight = torch.softmax(x_skeleton, dim=1)
            skeleton_weight = skeleton_weight[:,1].unsqueeze(1)
            skeleton_map_refine = x_refine * skeleton_weight
            # vis_attention(skeleton_weight[0][0], folder_name_ske_visatt, "output_ske_att", self.counter)
            # vis_attention_crit_ske_image(skeleton_weight[0][0],probs_weight[0][0], x_ori[0], folder_name_ske_crit_visatt, "output_ske_crit_att", self.counter)
            # vis_attention_crit_ske_image(y_pred_hard[0],probs_weight[0][0], x_ori[0], folder_name_crit_pred_visatt, "output_crit_pred_att", self.counter)

            endpoint_features_refine = prob_map_refine + skeleton_map_refine + x_refine
            endpoint_features_refine = self.final_conv_refine( endpoint_features_refine)
            # print("endpoint_features_eachbatch_after_refine----",endpoint_features_refine.shape)  # torch.Size([ 10, 2, 10, 10, 10])

            # return x
            return endpoint_features_refine

 

def compute_skeleton_and_endpoints(batch_binary_output):
    skeleton_batch = torch.zeros_like(batch_binary_output)
    # print("skeleton_batch---",skeleton_batch.shape)

    for i in range(batch_binary_output.size(0)):  
        binary_np = batch_binary_output[i].cpu().numpy()  
        skeleton_np = skeletonize(binary_np)
        skeleton_np[skeleton_np == 255] = 1
        skeleton_batch[i] = torch.tensor(skeleton_np).to(batch_binary_output.device)  
    # print("skeleton_batch-----", torch.min(skeleton_batch), torch.max(skeleton_batch))

    endpoints_batch = detect_endpoints(skeleton_batch)  # [bs,32,32,32]
    # print("endpoints_batch---",endpoints_batch.shape)

    return skeleton_batch, endpoints_batch


def detect_endpoints(skeleton):
    """
        skeleton (numpy.ndarray): 3D image of two skeletons, 1 bone, 0 others.
        endpoints (numpy.ndarray): 1 for endpoints
    """
    kernel = torch.ones((1, 1, 3, 3), dtype=torch.float32, device=skeleton.device)  # 3D卷积核
    kernel[:, :, 1, 1] = 0 

    skeleton = skeleton.unsqueeze(1)
    neighbors_count = F.conv2d(skeleton, kernel, padding=1)

    endpoints = (skeleton == 1) & ((neighbors_count == 1) | (neighbors_count == 0))
    endpoints = endpoints.float()
    return endpoints


def endpoints_select(pred, gt, model='train'):
    distances = cdist(pred, gt) 

    min_distances = distances.min(axis=1)
    # print('min_distances-----------', np.min(min_distances))

    mean_distance = min_distances.mean()
    std_distance = min_distances.std()

    if model=='train':
        threshold = mean_distance + std_distance
        # threshold = mean_distance
        # filtered_endpoints = np.array([]) if all(x <= 5 for x in min_distances) else pred[min_distances >= threshold]
        filtered_endpoints = np.array([]) if all(x <= 8 for x in min_distances) else pred[(min_distances >= threshold) & (min_distances > 8)]
        # filtered_endpoints = pred[min_distances >= threshold]
    else:
        threshold = mean_distance
        filtered_endpoints = pred[min_distances < 16] 
    print("mean dis:", mean_distance)
    print("std dis:", std_distance)
    return filtered_endpoints



def DBSCAN_2d(points_2d, img):
    # eps distance threshold，min_samples Minimum number of points per cluster
    size = img.shape
    max_len = max(size)
    # ep = (max_len / 16) / 4.0
    ep = (max_len/8.0) / 1.0    #STARE 512/8=64


    dbscan = DBSCAN(eps=ep, min_samples=1)
    clusters = dbscan.fit_predict(points_2d)

    clustered_points = {}
    for idx, label in enumerate(clusters):
        if label == -1:
            continue
        if label not in clustered_points:
            clustered_points[label] = []
        clustered_points[label].append(points_2d[idx])

    # for cluster_id, points in clustered_points.items():
    #     print(f"cluster {cluster_id}: {points}")

    selected_points = []
    for cluster_id, points in clustered_points.items():
        if len(points) > 1:
            selected_point = random.choice(points)
            selected_points.append(selected_point)

        else:
            selected_points.append(points[0])

    selected_points = np.array(selected_points)
    # print(selected_points)
    return  selected_points


def distance_tranformation(input_image):
    distance_maps = []

    for i in range(input_image.shape[0]):
        image_numpy = input_image[i].cpu().numpy()

        distance_map = distance_transform_edt(image_numpy)

        distance_map_tensor = torch.tensor(distance_map, dtype=torch.float32).to(input_image.device)
        distance_map_normalized = (distance_map_tensor - distance_map_tensor.min()) / (distance_map_tensor.max() - distance_map_tensor.min()
        )

        distance_maps.append(distance_map_normalized)

    distance_maps = torch.stack(distance_maps, dim=0)

    return distance_maps




def save_binary_array_as_image_skimage(skeleton_batch_soft,floder_name, file_name, count):
  
    output_folder_vis_results = '/media/fxzhou/DATADRIVE1/project/nnUNet-2/nnUNet_results/Dataset106_STARE/vis_results/'
    os.makedirs(output_folder_vis_results, exist_ok=True)
    os.makedirs(output_folder_vis_results+floder_name, exist_ok=True)

    if skeleton_batch_soft.is_cuda:
        skeleton_batch_soft = skeleton_batch_soft.cpu()
    
    skeleton_batch_soft = (skeleton_batch_soft > 0.5).float()
    
    np_array = skeleton_batch_soft.numpy()
    
    if len(np_array.shape) == 3:
        for i in range(np_array.shape[0]):
            #  0-255 
            frame = (np_array[i] * 255).astype(np.uint8)  
            io.imsave(output_folder_vis_results + floder_name + f"{file_name}_{count}_frame_{i}.png", frame, check_contrast=False)  # 保存每一帧

            # plt.figure()
            # plt.imshow(np_array[i], cmap='plasma')
            # plt.colorbar()  
            # plt.axis('off')
            # plt.savefig(output_folder_vis_results + floder_name + f"{file_name}_{count}_frame_{i}_att.png")

    elif len(np_array.shape) == 2:
        frame = (np_array * 255).astype(np.uint8) 
        io.imsave(file_name, frame) 
    else:
        raise ValueError("input shape should be [H, W] or [N, H, W]")
    
    

def vis_attention(skeleton_atten, floder_name, file_name, count):
    print("skeleton_atten------", skeleton_atten.shape) #(512,512)
    output_folder_vis_results = '/media/fxzhou/DATADRIVE1/project/nnUNet-2/nnUNet_results/Dataset106_STARE/vis_results/'
    os.makedirs(output_folder_vis_results, exist_ok=True)
    os.makedirs(output_folder_vis_results+floder_name, exist_ok=True)

    if skeleton_atten.is_cuda:
        skeleton_atten = skeleton_atten.cpu()
    
    np_array = skeleton_atten.numpy()
    np_array2 = np.random.rand(512, 512)  

    # np_array[np_array >0.5] =1.0
    # np_array[np_array <=0.5] =0.01
    
    unique  = np.unique(np_array)
    print(np.max(unique), np.min(unique))

    
    if len(np_array.shape) == 3:
            plt.figure()
            plt.imshow(np_array[i], cmap='plasma',vmin=0, vmax=1)
            plt.colorbar()  
            plt.axis('off') 
            plt.savefig(output_folder_vis_results + floder_name + f"{file_name}_{count}_frame_{i}.png")

    elif len(np_array.shape) == 2:
        plt.figure()
        # plt.imshow(np_array, cmap='plasma',vmin=0, vmax=1)
        # plt.imshow(np_array, cmap='viridis',vmin=0, vmax=1)
        # plt.colorbar() 
        # plt.axis('off')  
        # plt.savefig(output_folder_vis_results + floder_name + f"{file_name}_{count}_frame.png")

        colormap = plt.cm.plasma  
        rgb_image = colormap(np_array)
        plt.imsave(output_folder_vis_results + floder_name + f"{file_name}_{count}_frame.png", rgb_image)
        plt.close()  


    else:
        raise ValueError("input shape must be [H, W] or [N, H, W]")


def vis_attention_crit_ske_image(skeleton_atten, crit_atten,image, floder_name, file_name, count):
    print("skeleton_atten------", skeleton_atten.shape) #(512,512)
    print("image------", image.shape) #(512,512)

    output_folder_vis_results = '/media/fxzhou/DATADRIVE1/project/nnUNet-2/nnUNet_results/Dataset106_STARE/vis_results/'
    os.makedirs(output_folder_vis_results, exist_ok=True)
    os.makedirs(output_folder_vis_results+floder_name, exist_ok=True)

    if skeleton_atten.is_cuda:
        skeleton_atten = skeleton_atten.cpu()
        crit_atten = crit_atten.cpu()
        image = image.cpu()

    
    np_array = skeleton_atten.numpy()
    np_array_crit = crit_atten.numpy()
    image = image.numpy()


    
    if len(np_array.shape) == 3:
            plt.figure()
            plt.imshow(np_array[i], cmap='plasma')
            plt.colorbar()  
            plt.axis('off') 
            plt.savefig(output_folder_vis_results + floder_name + f"{file_name}_{count}_frame_{i}.png")

    elif len(np_array.shape) == 2:
        plt.figure()
        # plt.imshow(np_array, cmap='plasma')
        # plt.imshow(np_array_crit, cmap='plasma', alpha=0.5 )
        # plt.colorbar()  
        # plt.axis('off') 
        # plt.savefig(output_folder_vis_results + floder_name + f"{file_name}_{count}_frame_0.png")

        colormap = plt.cm.plasma  

        rgb_image = colormap(np_array)
        rgb_image2 = colormap(np_array_crit)
        rgb_image = 0.5 * rgb_image + 0.5 * rgb_image2
        plt.imsave(output_folder_vis_results + floder_name + f"{file_name}_{count}_frame.png", rgb_image)
        plt.close()  
    else:
        raise ValueError("input shape must be [H, W] or [N, H, W]")
    