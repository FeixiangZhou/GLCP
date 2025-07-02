import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from skimage.morphology import skeletonize_3d
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


# Basic block: Conv3d -> InstanceNorm3d -> LeakyReLU
class ConvDropoutNormReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dropout=False):
        super(ConvDropoutNormReLU, self).__init__()
        layers = [
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        ]
        if dropout:
            layers.insert(0, nn.Dropout3d(0.3))
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


# Encoder block: Downsample using Conv3d with stride > 1
class PlainConvEncoder(nn.Module):
    def __init__(self, in_channels):
        super(PlainConvEncoder, self).__init__()
        self.stages = nn.Sequential(
            StackedConvBlocks(in_channels, 32, stride=1),
            StackedConvBlocks(32, 64, stride=2),
            StackedConvBlocks(64, 128, stride=2),
            StackedConvBlocks(128, 256, stride=2),
            StackedConvBlocks(256, 320, stride=2)

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
        self.upconv1 = nn.ConvTranspose3d(320, 256, kernel_size=2, stride=2)
        self.conv1 = StackedConvBlocks(512, 256)

        self.upconv2 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.conv2 = StackedConvBlocks(256, 128)

        self.upconv3 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.conv3 = StackedConvBlocks(128, 64)

        self.upconv4 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.conv4 = StackedConvBlocks(64, 32)

    def forward(self, x, encoder_features):
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


class Unet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=2):
        super(Unet3D, self).__init__()
        self.counter = 0  
        self.encoder = PlainConvEncoder(in_channels)
        self.decoder = UNetDecoder()

        self.final_conv = nn.Conv3d(32, out_channels, kernel_size=1)
        self.final_conv_skeleton = nn.Conv3d(32, 2, kernel_size=1)
        self.final_conv_criticalregion = nn.Conv3d(32, 2, kernel_size=1)
        
        self.final_conv_refine = nn.Conv3d(32, out_channels, kernel_size=1)
        self.refine_conv_seg = nn.Conv3d(out_channels, 32, kernel_size=1)
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

        # Final convolution to get output
        x_skeleton = self.final_conv_skeleton(x)
        x_criticalregion = self.final_conv_criticalregion(x)
        x = self.final_conv(x)
        # skeleton_batch_soft = self.m_skeletonize(y_pred_hard.unsqueeze(1))#[2, 1, 32, 32, 32]

        if mask is not None:
            y_pred_fore = x[:, 1:]
            y_pred_fore = torch.max(y_pred_fore, dim=1, keepdim=True)[0]  # C foreground channels -> 1 channel
            y_pred_binary = torch.cat([x[:, :1], y_pred_fore], dim=1)
            y_prob_binary = torch.softmax(y_pred_binary, 1)
            y_pred_prob = y_prob_binary[:, 1]  # [2,32,32,32]
            y_pred_hard = (y_pred_prob > 0.5).float()  # [2,32,32,32]

            skeleton_batch, endpoints_batch = compute_skeleton_and_endpoints(y_pred_hard)

            #for gt
            mask_gt = mask.gt(0).squeeze(1).float()
            skeleton_batch_gt, endpoints_batch_gt = compute_skeleton_and_endpoints(mask_gt)

            # skeleton consistency
            prob_skeleton = y_prob_binary.clone()# [2, 2, 32, 32, 32]
            
            # prob_skeleton[:, 0, :, :,:][skeleton_batch == 0] = 0.7311  #([2, 2, 32, 32, 32])
            # prob_skeleton[:, 1, :, :,:][skeleton_batch == 0] = 0.2689  #([2, 2, 32, 32, 32])

            prob_skeleton[:, 0, :, :][skeleton_batch == 0] = 1 # ([2, 2, 32, 32, 32])
            prob_skeleton[:, 1, :, :][skeleton_batch == 0] = 0 # ([2, 2, 32, 32, 32])

            # prob_skeleton[:, 0, :, :] = prob_skeleton[:, 0, :, :] * (skeleton_batch != 0).float() + 0.7311  * (skeleton_batch == 0).float()
            # prob_skeleton[:, 1, :, :] = prob_skeleton[:, 1, :, :] * (skeleton_batch != 0).float() +  0.2689 * (skeleton_batch == 0).float()

            #one
            # prob_skeleton= y_pred_prob * (skeleton_batch != 0).float()

            #two
            # prob_skeleton_0 = y_pred_prob * (skeleton_batch_soft.squeeze(1))
            # prob_skeleton_1 = 1 - prob_skeleton_0
            # prob_skeleton = torch.stack([prob_skeleton_1, prob_skeleton_0], dim=1)


            endpoint_features_eachbatch = []
            endpoint_mask_eachbatch = []
            mask_criticalregion_eachbatch = []


            for i in range(y_pred_hard.size(0)):
                # print("batch---",i)
                endpoints_np = endpoints_batch[i, 0].cpu().numpy() 
                coords = np.argwhere(endpoints_np == 1)
          
                endpoints_gt_np = endpoints_batch_gt[i, 0].cpu().numpy()  
                coords_gt = np.argwhere(endpoints_gt_np == 1)
          
                mask_criticalregion = torch.zeros_like(mask[i, :, :, :, :])


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
                            print("selected_endpoints before cluster  ---", selected_endpoints.shape) 
                            selected_endpoints = DBSCAN_2d(selected_endpoints, endpoints_np)
                            print("selected_endpoints after cluster  ---", selected_endpoints.shape) #

                        else:
                            print("no selected_endpoints after endpoints_select  ---")
                            mask_criticalregion_eachbatch.append(mask_criticalregion)
                            continue
                    else:
                        selected_endpoints = coords
                        #for CCA data
                        # mask_criticalregion_eachbatch.append(mask_criticalregion)
                        # continue
                else:
                    selected_endpoints = coords_gt
                    if(len(selected_endpoints) <= 0):
                        mask_criticalregion_eachbatch.append(mask_criticalregion)
                        continue

                endpoint_features_eachsample = []
                endpoint_mask_eachsample = []


                for (z, y, x1) in selected_endpoints:
                    prob_map = x[i, :, :, :, :]
                    skeleton_map = x_skeleton[i, :, :, :, :]
                    imag_map = x_ori[i, :, :, :, :]
                    mask_map = mask[i,:,:,:,:]
                    scale_factor = 8   


                    wz = y_pred_hard.size(1) // scale_factor  
                    wy = y_pred_hard.size(2) // scale_factor 
                    wx = y_pred_hard.size(3) // scale_factor         

                    inter=1
                
                    zmin, zmax = max(0, z - wz//2), min(y_pred_hard.size(1), z + wz//2)
                    ymin, ymax = max(0, y - wy//2), min(y_pred_hard.size(2), y + wy//2)
                    xmin, xmax = max(0, x1 - wx//2), min(y_pred_hard.size(3), x1 + wx//2)

                
                    if zmax - zmin < wz:
                        if zmin == 0:
                            zmax = min(zmin + wz, y_pred_hard.size(1))
                        else:
                            zmin = max(0, zmax - wz)

                    if ymax - ymin < wy:
                        if ymin == 0:
                            ymax = min(ymin + wy, y_pred_hard.size(2))
                        else:
                            ymin = max(0, ymax - wy)

                    if xmax - xmin < wx:
                        if xmin == 0:
                            xmax = min(xmin + wx, y_pred_hard.size(3))
                        else:
                            xmin = max(0, xmax - wx)

                    prob_region = prob_map[:, zmin:zmax:inter,  ymin:ymax:inter, xmin:xmax:inter] 
                    prob_region = prob_region.unsqueeze(0) 


                    skeleton_region = skeleton_map[:, zmin:zmax:inter,  ymin:ymax:inter, xmin:xmax:inter]
                    skeleton_region = skeleton_region.unsqueeze(0)  

                    img_region = imag_map[:, zmin:zmax:inter, ymin:ymax:inter, xmin:xmax:inter]  
                    img_region = img_region.unsqueeze(0)  

                    mask_region = mask_map[:, zmin:zmax:inter,  ymin:ymax:inter, xmin:xmax:inter]
                    mask_region = mask_region.unsqueeze(0) 


                    #feature fusion
                    prob_region = torch.cat((img_region, prob_region), 1)

                    mask_criticalregion[:,zmin:zmax, ymin:ymax, xmin:xmax] = 1
                    endpoint_features_eachsample.append(prob_region)
                    endpoint_mask_eachsample.append(mask_region)

                if(len(endpoint_features_eachsample) > 0):
                    endpoint_features_eachsample = torch.cat(endpoint_features_eachsample)
                    endpoint_mask_eachsample = torch.cat(endpoint_mask_eachsample)

                    endpoint_features_eachbatch.append(endpoint_features_eachsample)
                    endpoint_mask_eachbatch.append(endpoint_mask_eachsample)
                    mask_criticalregion_eachbatch.append(mask_criticalregion)


            if (len(endpoint_features_eachbatch) > 0):
                endpoint_features_eachbatch = torch.cat(endpoint_features_eachbatch)
                endpoint_mask_eachbatch = torch.cat(endpoint_mask_eachbatch)
                mask_criticalregion_eachbatch = torch.stack(mask_criticalregion_eachbatch) # torch.Size([ bs, 1, 32, 32, 32])

            '''
            refine module
            '''

            x_refine= self.refine_conv_seg(x)
            probs_weight = torch.softmax(x_criticalregion, dim=1)
            probs_weight = probs_weight[:,1].unsqueeze(1)
            prob_map_refine = x_refine * probs_weight


            skeleton_weight = torch.softmax(x_skeleton, dim=1)
            skeleton_weight = skeleton_weight[:,1].unsqueeze(1)
            skeleton_map_refine = x_refine * skeleton_weight

            endpoint_features_refine = prob_map_refine + skeleton_map_refine  + x_refine 

            endpoint_features_refine = self.final_conv_refine(endpoint_features_refine)


            x_criticalregion_train = x_criticalregion.clone()
            x_criticalregion_train = torch.softmax(x_criticalregion_train, 1) #[batchsize, 2， 32,32,32】
            x_criticalregion_np = x_criticalregion_train.detach().cpu().numpy()

            if isinstance(mask_criticalregion_eachbatch, list):  # all coords in a batch is 0
                mask_criticalregion_eachbatch = torch.stack(mask_criticalregion_eachbatch)
            return x, x_skeleton, x_criticalregion, mask_criticalregion_eachbatch, endpoint_features_refine, skeleton_batch_gt, prob_skeleton
        else:
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

            x_refine= self.refine_conv_seg(x)
    

            probs_weight = torch.softmax(x_criticalregion, dim=1)
            probs_weight = probs_weight[:,1].unsqueeze(1)
            prob_map_refine = x_refine * probs_weight

            skeleton_weight = torch.softmax(x_skeleton, dim=1)
            skeleton_weight = skeleton_weight[:,1].unsqueeze(1)
            skeleton_map_refine = x_refine * skeleton_weight

            endpoint_features_refine = prob_map_refine + skeleton_map_refine + x_refine
            endpoint_features_refine = self.final_conv_refine(endpoint_features_refine)
            
            return endpoint_features_refine 


def compute_skeleton_and_endpoints(batch_binary_output):
    skeleton_batch = torch.zeros_like(batch_binary_output)

    for i in range(batch_binary_output.size(0)): 
        binary_np = batch_binary_output[i].cpu().numpy()  
        skeleton_np = skeletonize_3d(binary_np)
        skeleton_np[skeleton_np == 255] = 1
        skeleton_batch[i] = torch.tensor(skeleton_np).to(batch_binary_output.device) 


    endpoints_batch = detect_endpoints(skeleton_batch)  # [bs,32,32,32]
    return skeleton_batch, endpoints_batch


def detect_endpoints(skeleton):
    # Define a 3D neighborhood convolution kernel to calculate the number of neighbors for each pixel
    kernel = torch.ones((1, 1, 3, 3, 3), dtype=torch.float32, device=skeleton.device)  
    kernel[:,:,1, 1, 1] = 0  # The center itself is not considered a neighbor
    skeleton = skeleton.unsqueeze(1)
    neighbors_count = F.conv3d(skeleton, kernel, padding=1)
    endpoints = (skeleton == 1) & ((neighbors_count == 1) | (neighbors_count == 0))
    endpoints =  endpoints.float()
    return endpoints



def endpoints_select(pred, gt, model='train'):
    distances = cdist(pred, gt)  

    min_distances = distances.min(axis=1)
    mean_distance = min_distances.mean()
    std_distance = min_distances.std()

    if model=='train':
        threshold = mean_distance + std_distance
        # threshold = mean_distance
        # filtered_endpoints = np.array([]) if all(x <= 5 for x in min_distances) else pred[min_distances >= threshold]
        filtered_endpoints = np.array([]) if all(x <= 5 for x in min_distances) else pred[(min_distances >= threshold) & (min_distances > 5)]
        # filtered_endpoints = pred[min_distances >= threshold]

    else:
        threshold = mean_distance
        filtered_endpoints = pred[min_distances < threshold]
    print("mean dis:", mean_distance)
    print("std dis:", std_distance)
    return filtered_endpoints


def DBSCAN_2d(points_2d, img):

    size = img.shape
    max_len = max(size)
    # ep = (max_len/8.0) / 2.0
    ep = (max_len/8.0) / 1.0


    dbscan = DBSCAN(eps=ep, min_samples=1)
    clusters = dbscan.fit_predict(points_2d)

    clustered_points = {}
    for idx, label in enumerate(clusters):
        if label == -1:
            continue
        if label not in clustered_points:
            clustered_points[label] = []
        clustered_points[label].append(points_2d[idx])

    for cluster_id, points in clustered_points.items():
        print(f"簇 {cluster_id}: {points}")

    selected_points = []
    for cluster_id, points in clustered_points.items():
        if len(points) > 1:
            selected_point = random.choice(points)
            selected_points.append(selected_point)

        else:
            selected_points.append(points[0])

    selected_points = np.array(selected_points)
    return  selected_points