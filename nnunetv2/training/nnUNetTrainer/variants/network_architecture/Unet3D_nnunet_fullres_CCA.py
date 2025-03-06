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


output_folder_vis_results = '/media/fxzhou/DATADRIVE1/project/nnUNet-2/nnUNet_results/Dataset106_STARE/vis_results/'
os.makedirs(output_folder_vis_results, exist_ok=True)




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
        # self.decoder_skeleton = UNetDecoder()
        # self.decoder_criticalregion = UNetDecoder()



        self.final_conv = nn.Conv3d(32, out_channels, kernel_size=1)
        self.final_conv_skeleton = nn.Conv3d(32, 2, kernel_size=1)
        self.final_conv_criticalregion = nn.Conv3d(32, 2, kernel_size=1)


        # self.encoder_refine = PlainConvEncoder(in_channels=3)
        # self.decoder_refine = UNetDecoder()
        
        self.final_conv_refine = nn.Conv3d(32, out_channels, kernel_size=1)

        self.refine_conv_seg = nn.Conv3d(out_channels, 32, kernel_size=1)
        # self.m_skeletonize = SoftSkeletonize(num_iter=10)




    def forward(self, x, mask=None, train=None):
        # print("train-------------", train)
        # print("mask is None-------------", mask is None)
        # Encoding path
        x_ori = x
        encoder_features = self.encoder(x)
        # Bottleneck feature map
        bottleneck = encoder_features[-1]
        # Decoding path
        x = self.decoder(bottleneck, encoder_features)
        # x_skeleton = self.decoder_skeleton(bottleneck, encoder_features)
        # x_criticalregion = self.decoder_criticalregion(bottleneck, encoder_features)



        # Final convolution to get output
        x_skeleton = self.final_conv_skeleton(x)
        x_criticalregion = self.final_conv_criticalregion(x)
        x = self.final_conv(x)




        y_pred_fore = x[:, 1:]
        y_pred_fore = torch.max(y_pred_fore, dim=1, keepdim=True)[0]  # C foreground channels -> 1 channel
        y_pred_binary = torch.cat([x[:, :1], y_pred_fore], dim=1)
        y_prob_binary = torch.softmax(y_pred_binary, 1)
        y_pred_prob = y_prob_binary[:, 1]  # [2,32,32,32]
        y_pred_hard = (y_pred_prob > 0.5).float()  # [2,32,32,32]


        # skeleton_batch_soft = self.m_skeletonize(y_pred_hard.unsqueeze(1))#[2, 1, 32, 32, 32]
        # print("skeleton_batch_soft-----", torch.min(skeleton_batch_soft), torch.max(skeleton_batch_soft))

        skeleton_batch, endpoints_batch = compute_skeleton_and_endpoints(y_pred_hard)
        # skeleton_batch--- torch.Size([2, 32, 32, 32])
        # endpoints_batch--- torch.Size([2, 1, 32, 32, 32])

        if mask is not None:
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


            # two
            # prob_skeleton_0 = y_pred_prob * (skeleton_batch_soft.squeeze(1))
            # prob_skeleton_1 = 1 - prob_skeleton_0
            # prob_skeleton = torch.stack([prob_skeleton_1, prob_skeleton_0], dim=1)




            endpoint_features_eachbatch = []
            endpoint_mask_eachbatch = []
            mask_criticalregion_eachbatch = []


            for i in range(y_pred_hard.size(0)):
                # print("batch---",i)
                endpoints_np = endpoints_batch[i, 0].cpu().numpy()  # [32, 32, 32]
                coords = np.argwhere(endpoints_np == 1)
                print("len(coords)---", len(coords))
                # print("coords---", coords.shape) 
                # print("coords---", coords) 



                endpoints_gt_np = endpoints_batch_gt[i, 0].cpu().numpy()  # [32, 32, 32]
                coords_gt = np.argwhere(endpoints_gt_np == 1)
                print("len(coords_gt)---", len(coords_gt))
                # print("coords_gt---", coords_gt.shape) 
                # print("coords_gt---", coords_gt)

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
                        # selected_endpoints = coords
                        
                        #for CCA data
                        mask_criticalregion_eachbatch.append(mask_criticalregion)
                        continue
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


                    wz = y_pred_hard.size(1) // 8  
                    wy = y_pred_hard.size(2) // 8 
                    wx = y_pred_hard.size(3) // 8         

                    inter=1
                    
                    # center = (wz // 2, wy // 2, wx // 2)
                    # criticalregion_prob = create_3d_probability_map(center, window_size=(wz, wy, wx), mask= mask)

                    zmin, zmax = max(0, z - wz//2), min(y_pred_hard.size(1), z + wz//2)
                    ymin, ymax = max(0, y - wy//2), min(y_pred_hard.size(2), y + wy//2)
                    xmin, xmax = max(0, x1 - wx//2), min(y_pred_hard.size(3), x1 + wx//2)



                    #  let zmax - zmin is 10
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
                    prob_region = prob_region.unsqueeze(0)  # add bs
                    # print("prob_region----", prob_region.shape)  # torch.Size([1， 64，10，10,10])


                    skeleton_region = skeleton_map[:, zmin:zmax:inter,  ymin:ymax:inter, xmin:xmax:inter]
                    skeleton_region = skeleton_region.unsqueeze(0)  

                    img_region = imag_map[:, zmin:zmax:inter, ymin:ymax:inter, xmin:xmax:inter]  
                    img_region = img_region.unsqueeze(0)  

                    mask_region = mask_map[:, zmin:zmax:inter,  ymin:ymax:inter, xmin:xmax:inter]
                    mask_region = mask_region.unsqueeze(0) 
                    # print("mask_region----", mask_region.shape)  # torch.Size([1， 1，10，10,10])


                    #feature fusion
                    prob_region = torch.cat((img_region, prob_region), 1)

                    mask_criticalregion[:,zmin:zmax, ymin:ymax, xmin:xmax] = 1
                    # mask_criticalregion[:,zmin:zmax, ymin:ymax, xmin:xmax] = criticalregion_prob.unsqueeze(0)

                    # print("mask_criticalregion----", mask_criticalregion.shape)  # torch.Size([1， 1，10，10,10])
                    # print("mask_criticalregion----", mask_criticalregion.sum())

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
                print("endpoint_features_eachbatch----", endpoint_features_eachbatch.shape)  # torch.Size([ 10, 64, 10, 10, 10])
                endpoint_mask_eachbatch = torch.cat(endpoint_mask_eachbatch)
                print("endpoint_mask_eachbatch----", endpoint_mask_eachbatch.shape)  # torch.Size([ 10, 1, 10, 10, 10])

                mask_criticalregion_eachbatch = torch.stack(mask_criticalregion_eachbatch) # torch.Size([ bs, 1, 32, 32, 32])


                ## select 8 samples for training
                # num_samples = x_ori.shape[0] * 2
                # indices = torch.randperm(endpoint_features_eachbatch.size(0))[:num_samples]  
                # endpoint_features_eachbatch = endpoint_features_eachbatch[indices] 
                # endpoint_mask_eachbatch = endpoint_mask_eachbatch[indices] 
                # print("endpoint_features_eachbatch sample----", endpoint_features_eachbatch.shape)  
                # print("endpoint_mask_eachbatch sample----", endpoint_mask_eachbatch.shape)  



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
            # probs_weight_np = probs_weight.detach().cpu().numpy()
            # probs_weight_np[probs_weight_np < 0.5] = 1
            # probs_weight_tensor  = torch.tensor(probs_weight_np).to(x.device)
            prob_map_refine = x_refine * probs_weight


            skeleton_weight = torch.softmax(x_skeleton, dim=1)
            skeleton_weight = skeleton_weight[:,1].unsqueeze(1)
            skeleton_map_refine = x_refine * skeleton_weight

            # x_att_refine = self.satt(x_refine) * x_refine

            endpoint_features_refine = prob_map_refine + skeleton_map_refine  + x_refine 
            # endpoint_features_refine = prob_map_refine  + x_refine 
            # endpoint_features_refine = self.unet_refine(endpoint_features_refine)



            # encoder_features_local = self.encoder_refine(prob_map_refine)
            # bottleneck_local = encoder_features_local[-1]
            # endpoint_features_refine = self.decoder_refine(bottleneck_local, encoder_features_local)


            endpoint_features_refine = self.final_conv_refine(endpoint_features_refine)
            print("endpoint_features_eachbatch_after_refine----",endpoint_features_refine.shape)  # torch.Size([ bs, 2, 32, 32, 32])



            x_criticalregion_train = x_criticalregion.clone()
            x_criticalregion_train = torch.softmax(x_criticalregion_train, 1) #[batchsize, 2， 32,32,32】
            x_criticalregion_np = x_criticalregion_train.detach().cpu().numpy()
            print('coords_criticalregion_train max----', np.max(x_criticalregion_np[:,1,:,:,:],axis=(1, 2, 3)))


            if isinstance(mask_criticalregion_eachbatch, list):  # all coords in a batch is 0
                print("************mask_criticalregion_eachbatch is list*****************")
                mask_criticalregion_eachbatch = torch.stack(mask_criticalregion_eachbatch)
            print("mask_criticalregion_eachbatch----",mask_criticalregion_eachbatch.shape)  # torch.Size([ 10, 1, 32, 32, 10])

         

            return x, x_skeleton, x_criticalregion, mask_criticalregion_eachbatch, endpoint_features_refine, skeleton_batch_gt, prob_skeleton
        else:
            print("-----------------------------test-------------------------------")
            # x_criticalregion = torch.softmax(x_criticalregion, 1)
            # x_criticalregion = x_criticalregion[:,1].unsqueeze(1)

            # prob_map_refine = torch.cat((x, x_criticalregion), 1) 
            # prob_map_refine = torch.cat((x, x_criticalregion, x_skeleton), 1)
            # prob_map_refine = x + x_criticalregion

            x_refine= self.refine_conv_seg(x)
    

            probs_weight = torch.softmax(x_criticalregion, dim=1)
            probs_weight = probs_weight[:,1].unsqueeze(1)
            # probs_weight_np = probs_weight.detach().cpu().numpy()
            # probs_weight_np[probs_weight_np < 0.5] = 1
            # probs_weight_tensor  = torch.tensor(probs_weight_np).to(x.device)
            prob_map_refine = x_refine * probs_weight


            skeleton_weight = torch.softmax(x_skeleton, dim=1)
            skeleton_weight = skeleton_weight[:,1].unsqueeze(1)
            skeleton_map_refine = x_refine * skeleton_weight

            # x_att_refine = self.satt(x_refine) * x_refine

            endpoint_features_refine = prob_map_refine + skeleton_map_refine + x_refine
            # endpoint_features_refine = prob_map_refine + x_refine

            # endpoint_features_refine = self.unet_refine(endpoint_features_refine) 
 
             # encoder_features_local = self.encoder_refine(prob_map_refine)
            # bottleneck_local = encoder_features_local[-1]
            # endpoint_features_refine = self.decoder_refine(bottleneck_local, encoder_features_local)

            endpoint_features_refine = self.final_conv_refine(endpoint_features_refine)
            # print("endpoint_features_eachbatch_after_refine----",endpoint_features_refine.shape)  # torch.Size([ 10, 2, 10, 10, 10])
            
            return x
            # return endpoint_features_refine 




# 计算骨架和端点的函数
def compute_skeleton_and_endpoints(batch_binary_output):
    skeleton_batch = torch.zeros_like(batch_binary_output)
    # print("skeleton_batch---",skeleton_batch.shape)

    for i in range(batch_binary_output.size(0)):  # 对于每个样本
        binary_np = batch_binary_output[i].cpu().numpy()  # 转换为 NumPy 数组
        # print("binary_np----", np.sum(binary_np))
        skeleton_np = skeletonize_3d(binary_np)
        skeleton_np[skeleton_np == 255] = 1
        skeleton_batch[i] = torch.tensor(skeleton_np).to(batch_binary_output.device)  # 转换回张量
    # print("skeleton_batch-----", torch.min(skeleton_batch), torch.max(skeleton_batch))


    endpoints_batch = detect_endpoints(skeleton_batch)  # [bs,32,32,32]
    # print("endpoints_batch---",endpoints_batch.shape)

    return skeleton_batch, endpoints_batch


# detect_endpoints函数，检测骨骼图像的端点
def detect_endpoints(skeleton):
    """
    检测3D骨架图像中的端点。

    参数:
        skeleton (numpy.ndarray): 3D二值骨架图像，骨架为1，其余为0。

    返回:
        endpoints (numpy.ndarray): 3D数组，端点处为1，其余为0。
    """
    # 定义3D邻域卷积核，用于计算每个像素的邻居数量
    kernel = torch.ones((1, 1, 3, 3, 3), dtype=torch.float32, device=skeleton.device)  # 3D卷积核
    kernel[:,:,1, 1, 1] = 0  # 中心点自己不算邻居

    skeleton = skeleton.unsqueeze(1)

    # 计算每个像素的邻居数
    neighbors_count = F.conv3d(skeleton, kernel, padding=1)

    # 端点是那些在骨架中且邻居数为1的点
    endpoints = (skeleton == 1) & ((neighbors_count == 1) | (neighbors_count == 0))
    endpoints =  endpoints.float()
    return endpoints



def endpoints_select(pred, gt, model='train'):
    # 计算 B 中每个点到 A 中所有点的欧式距离
    distances = cdist(pred, gt)  # 距离矩阵，维度为 (5, 10)

    # 找到每个 B 点到 A 中最近点的距离
    min_distances = distances.min(axis=1)
    # print('min_distances-----------', min_distances)

    # 计算最近距离的平均值和方差
    mean_distance = min_distances.mean()
    std_distance = min_distances.std()

    # 筛选出距离大于平均值和方差之和的 B 中坐标
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
    # 使用 DBSCAN 进行聚类
    # eps 是距离阈值，min_samples 是每个簇的最小点数
    size = img.shape
    max_len = max(size)
    # ep = (max_len/8.0) / 2.0
    ep = (max_len/8.0) / 1.0


    dbscan = DBSCAN(eps=ep, min_samples=1)
    clusters = dbscan.fit_predict(points_2d)

    # 将点按簇分类
    clustered_points = {}
    for idx, label in enumerate(clusters):
        if label == -1:
            # 如果 label 为 -1，表示为噪声点，不属于任何簇
            continue
        if label not in clustered_points:
            clustered_points[label] = []
        clustered_points[label].append(points_2d[idx])

    # 输出每个簇的点
    # for cluster_id, points in clustered_points.items():
    #     print(f"簇 {cluster_id}: {points}")

    selected_points = []
    for cluster_id, points in clustered_points.items():
        if len(points) > 1:
            # 如果该簇的点数大于 1，随机选择一个点
            selected_point = random.choice(points)
            selected_points.append(selected_point)

        else:
            selected_points.append(points[0])

    selected_points = np.array(selected_points)
    # print(selected_points)

    return  selected_points


def distance_tranformation(input_image):
    # 初始化距离变换结果
    distance_maps = []

    # 对每个 batch 的样本单独处理
    for i in range(input_image.shape[0]):
        # 提取单个样本，转到 CPU 并转为 NumPy 数组
        image_numpy = input_image[i].cpu().numpy()
        
        # 计算距离变换
        distance_map = distance_transform_edt(image_numpy)
        
        # 将结果转为 PyTorch 张量并归一化
        distance_map_tensor = torch.tensor(distance_map, dtype=torch.float32).to(input_image.device)
        distance_map_normalized = (distance_map_tensor - distance_map_tensor.min()) / (
            distance_map_tensor.max() - distance_map_tensor.min()
        )
        
        # 保存到列表
        distance_maps.append(distance_map_normalized)

    # 堆叠结果到一个张量
    distance_maps = torch.stack(distance_maps, dim=0)

    return distance_maps

def create_3d_probability_map(center, window_size=(10, 6, 6), mask=None):
    """
    根据中心点生成一个3D概率地图，中心概率为1，向外衰减。

    参数：
    - center: (cx, cy, cz)，即中心点的坐标
    - window_size: 窗口大小 (depth, height, width)，默认为(10, 6, 6)

    返回：
    - prob_map: 生成的3D概率地图，大小为 (depth, height, width)
    """
    cz, cy, cx = center
    depth, height, width = window_size

    # 创建一个网格，范围为 [-half_depth, half_depth] 和 [-half_height, half_height] 和 [-half_width, half_width]
    # z, y, x = np.ogrid[-half_depth:half_depth + 1, -half_height:half_height + 1, -half_width:half_width + 1]
    z, y, x = np.ogrid[-0:depth, -0:height, -0:width]



    # 计算每个点到中心点的欧几里得距离
    # distance = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    distance = np.sqrt((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2)

    # 使用高斯分布公式计算概率：exp(-distance^2 / (2 * sigma^2))
    sigma = 8  # 高斯分布的标准差
    prob_map = np.exp(-distance ** 2 / (2 * sigma ** 2))
    # print('prob_map-----------', np.max(prob_map))
    # print('prob_map-----------', prob_map)

    prob_map_tensor = torch.tensor(prob_map, dtype=torch.float32).to(mask.device)

    return prob_map_tensor




def update_prob(coordinates, probabilities, prob_map):

    # 找到唯一坐标和索引
    unique_coords, indices = torch.unique(coordinates, dim=0, return_inverse=True)

    # 初始化更新后的概率张量
    updated_probabilities = torch.zeros((unique_coords.size(0), probabilities.size(1)))
    updated_probabilities = updated_probabilities.to(prob_map.device)

    # 遍历每个唯一坐标，取概率平均值
    for i in range(unique_coords.size(0)):
        mask = (indices == i)  # 找到对应坐标的位置
        updated_probabilities[i] = probabilities[mask].mean(dim=0)

    # 遍历每个坐标，将概率值替换到 map 的对应位置
    for i, (z, y, x) in enumerate(unique_coords):
        prob_map[0, :, z, y, x] = updated_probabilities[i]
    
    
    print('unique_coords-----', unique_coords.shape)

    return prob_map

    # print("唯一坐标:")
    # print("更新后的概率:")
    # print(updated_probabilities)

# 1. 模拟的随机数据集类
class RandomMedicalImageDataset(Dataset):
    def __init__(self, num_samples, image_size):
        self.num_samples = num_samples
        self.image_size = image_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 生成随机的3D图像数据 (C, D, H, W)
        image = torch.randn(self.image_size).float()

        # 生成随机的3D标签数据（0或1，模拟二分类的掩码）
        mask = torch.randint(0, 2, self.image_size[1:]).long()  # 二分类标签

        return image, mask


criterion_BCE = nn.BCEWithLogitsLoss()

# 2. 训练函数
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for i, (images, masks) in enumerate(dataloader):
        images, masks = images.to(device), masks.to(device)


        # 前向传播
        outputs, output_criticalregion, mask_criticalregion_eachbatch, outputs_skeleton, endpoint_features_eachbatch, endpoint_mask_eachbatch, prob_skeleton = model(images, masks)
        print("outputs--------", outputs.shape) #[bs, 2, 32, 32, 32])
        print("masks--------", masks.shape) #[bs, 32, 32, 32])

        # 计算损失
        loss = criterion(outputs, masks)
        loss_criticalregion = criterion_BCE(output_criticalregion, mask_criticalregion_eachbatch)
        print("loss_criticalregion:", loss_criticalregion.item())


        loss_con = consistency_constraint_loss_mse(outputs_skeleton, prob_skeleton)
        print("Consistency Constraint Loss:", loss_con.item())
        loss = loss + loss_criticalregion + loss_con


        if (len(endpoint_features_eachbatch) > 0):
            loss_refine= criterion(endpoint_features_eachbatch, endpoint_mask_eachbatch)
            loss = loss + loss_refine



        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(dataloader)


# 3. 验证函数
def validate(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)

    return 0

# 5. 主函数
def main():
    # 超参数
    num_epochs = 10
    batch_size = 2
    learning_rate = 1e-4
    image_size = (1, 32, 32, 32)  # 模拟的图像尺寸 (C, D, H, W)

    # 准备随机生成的数据集
    train_dataset = RandomMedicalImageDataset(num_samples=10, image_size=image_size)
    val_dataset = RandomMedicalImageDataset(num_samples=2, image_size=image_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # 定义模型、损失函数和优化器
    model = Unet3D(in_channels=1, out_channels=2).to(device)
    criterion = nn.CrossEntropyLoss()  # 使用交叉熵损失进行二分类
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练循环
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        # val_loss = validate(model, val_loader, criterion, device)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}")


        # 可选：保存模型
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"unet3d_epoch_{epoch + 1}.pth")
            print(f"Model saved at epoch {epoch + 1}")
    # val_loss = validate(model, val_loader, criterion, device)


if __name__ == "__main__":
    main()