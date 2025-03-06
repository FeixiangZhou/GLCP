import multiprocessing
import os
from copy import deepcopy
from multiprocessing import Pool
from typing import Tuple, List, Union, Optional

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import subfiles, join, save_json, load_json, \
    isfile
from nnunetv2.configuration import default_num_processes
from nnunetv2.imageio.base_reader_writer import BaseReaderWriter
from nnunetv2.imageio.reader_writer_registry import determine_reader_writer_from_dataset_json, \
    determine_reader_writer_from_file_ending
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
# the Evaluator class of the previous nnU-Net was great and all but man was it overengineered. Keep it simple
from nnunetv2.utilities.json_export import recursive_fix_for_json_export
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from skimage.morphology import skeletonize, skeletonize_3d, binary_erosion

from scipy.spatial.distance import directed_hausdorff
from scipy.spatial.distance import cdist
import monai
from nnunetv2.evaluation.BettiMatching import BettiMatching

from scipy.ndimage import distance_transform_edt
from skimage import morphology, measure

from scipy.spatial import cKDTree
import cv2
from scipy.ndimage import label, generate_binary_structure
import scipy

from skimage.morphology import skeletonize_3d, cube, binary_dilation



# pool = Pool(8)


def label_or_region_to_key(label_or_region: Union[int, Tuple[int]]):
    return str(label_or_region)


def key_to_label_or_region(key: str):
    try:
        return int(key)
    except ValueError:
        key = key.replace('(', '')
        key = key.replace(')', '')
        splitted = key.split(',')
        return tuple([int(i) for i in splitted if len(i) > 0])


def save_summary_json(results: dict, output_file: str):
    """
    stupid json does not support tuples as keys (why does it have to be so shitty) so we need to convert that shit
    ourselves
    """
    results_converted = deepcopy(results)
    # convert keys in mean metrics
    results_converted['mean'] = {label_or_region_to_key(k): results['mean'][k] for k in results['mean'].keys()}
    # convert metric_per_case
    for i in range(len(results_converted["metric_per_case"])):
        results_converted["metric_per_case"][i]['metrics'] = \
            {label_or_region_to_key(k): results["metric_per_case"][i]['metrics'][k]
             for k in results["metric_per_case"][i]['metrics'].keys()}
    # sort_keys=True will make foreground_mean the first entry and thus easy to spot
    save_json(results_converted, output_file, sort_keys=True)


def load_summary_json(filename: str):
    results = load_json(filename)
    # convert keys in mean metrics
    results['mean'] = {key_to_label_or_region(k): results['mean'][k] for k in results['mean'].keys()}
    # convert metric_per_case
    for i in range(len(results["metric_per_case"])):
        results["metric_per_case"][i]['metrics'] = \
            {key_to_label_or_region(k): results["metric_per_case"][i]['metrics'][k]
             for k in results["metric_per_case"][i]['metrics'].keys()}
    return results


def labels_to_list_of_regions(labels: List[int]):
    return [(i,) for i in labels]


def region_or_label_to_mask(segmentation: np.ndarray, region_or_label: Union[int, Tuple[int, ...]]) -> np.ndarray:
    if np.isscalar(region_or_label): #True
        return segmentation == region_or_label
    else:
        mask = np.zeros_like(segmentation, dtype=bool)
        for r in region_or_label:
            mask[segmentation == r] = True
    return mask


def compute_tp_fp_fn_tn(mask_ref: np.ndarray, mask_pred: np.ndarray, ignore_mask: np.ndarray = None):
    if ignore_mask is None:
        use_mask = np.ones_like(mask_ref, dtype=bool)
    else:
        use_mask = ~ignore_mask
    tp = np.sum((mask_ref & mask_pred) & use_mask)
    fp = np.sum(((~mask_ref) & mask_pred) & use_mask)
    fn = np.sum((mask_ref & (~mask_pred)) & use_mask)
    tn = np.sum(((~mask_ref) & (~mask_pred)) & use_mask)
    return tp, fp, fn, tn

def cl_score(v, s):
    """[this function computes the skeleton volume overlap]

    Args:
        v ([bool]): [image]
        s ([bool]): [skeleton]

    Returns:
        [float]: [computed skeleton volume intersection]
    """
    # print("-------np.sum(s)",np.sum(s))
    if np.sum(s)==0:
        return np.nan
    else:
        return np.sum(v * s) / np.sum(s)
    # return (np.sum(v * s)+ 1e-5) / (np.sum(s)+ 1e-5)



def clDice(v_p, v_l):
    """[this function computes the cldice metric]

    Args:
        v_p ([bool]): [predicted image]
        v_l ([bool]): [ground truth image]

    Returns:
        [float]: [cldice metric]
    """
    if len(v_p.shape) == 2:
        tprec = cl_score(v_p, skeletonize(v_l))
        tsens = cl_score(v_l, skeletonize(v_p))
    elif len(v_p.shape) == 3:
        tprec = cl_score(v_p, skeletonize_3d(v_l))
        tsens = cl_score(v_l, skeletonize_3d(v_p))
    return 2 * tprec * tsens / (tprec + tsens)

def hausdorff_distance_single(seg, label):
    # segmentation = seg.squeeze(1)
    # mask = label.squeeze(1)
    segmentation = seg
    mask = label

    non_zero_seg = np.transpose(np.nonzero(segmentation))
    non_zero_mask = np.transpose(np.nonzero(mask))
    h_dist = max(directed_hausdorff(non_zero_seg, non_zero_mask)[0],
                 directed_hausdorff(non_zero_mask, non_zero_seg)[0])
    return h_dist


def compute_nsd(pred, gt, tolerance=1.0):
    """
    计算二分类分割的 Normalized Surface Distance (NSD)
    
    参数:
        pred (np.ndarray): 二值分割结果 (0/1)，形状为 [H, W] 或 [D, H, W]
        gt (np.ndarray): 二值真值 (0/1)，形状与 pred 相同
        tolerance (float): 距离阈值
    
    返回:
        float: Normalized Surface Distance (NSD)
    """
    pred = (pred > 0.5).astype(np.uint8)
    gt = (gt > 0.5).astype(np.uint8)

    pred_boundary = pred - binary_erosion(pred)
    gt_boundary = gt - binary_erosion(gt)

    gt_distances = distance_transform_edt(1 - gt_boundary)  
    pred_distances = distance_transform_edt(1 - pred_boundary) 

    # 找到在阈值范围内的预测边界点
    pred_within_tolerance = pred_boundary * (gt_distances <= tolerance)

    # 计算 NSD
    num_within_tolerance = pred_within_tolerance.sum()  # 在阈值范围内的预测边界点数
    total_gt_boundary_points = gt_boundary.sum()  # 真值边界点总数

    # 归一化结果
    nsd = num_within_tolerance / total_gt_boundary_points if total_gt_boundary_points > 0 else 0.0
    return nsd


def compute_hausdorff_distance_fast(pred, gt):
    """
    快速计算 95% Hausdorff Distance (HD95)

    参数:
        pred (np.ndarray): 二值分割结果 (0/1)，形状为 [H, W] 或 [D, H, W]
        gt (np.ndarray): 二值真值 (0/1)，形状与 pred 相同
    
    返回:
        float: 95% Hausdorff Distance
    """
    # 获取非零点坐标
    pred_points = np.argwhere(pred > 0)
    gt_points = np.argwhere(gt > 0)

    # 如果任一边为空，返回无穷大
    # print("len(pred_points) ",len(pred_points) == 0)
    # print("len(gt_points) ",len(gt_points) == 0)

    if len(pred_points) == 0 or len(gt_points) == 0:
        # return float('inf')
        return np.nan

    # 使用 KD-Tree 进行快速最近邻查找
    tree_pred = cKDTree(pred_points)
    tree_gt = cKDTree(gt_points)

    # 计算 pred -> gt 的最近距离
    distances_pred_to_gt, _ = tree_gt.query(pred_points)
    # 计算 gt -> pred 的最近距离
    distances_gt_to_pred, _ = tree_pred.query(gt_points)

    # 计算 95% 分位数
    hd95_pred_to_gt = np.percentile(distances_pred_to_gt, 95)
    hd95_gt_to_pred = np.percentile(distances_gt_to_pred, 95)

    # 返回两者的最大值
    return max(hd95_pred_to_gt, hd95_gt_to_pred)


def compute_hausdorff_distance(pred, gt):
    """
    计算 Hausdorff Distance (HD)

    参数:
        pred (np.ndarray): 二值分割结果 (0/1)，形状为 [H, W] 或 [D, H, W]
        gt (np.ndarray): 二值真值 (0/1)，形状与 pred 相同
    
    返回:
        float: Hausdorff Distance
    """
    # 获取分割边界点坐标
    pred_points = np.argwhere(pred > 0)  # 分割结果的前景点坐标
    gt_points = np.argwhere(gt > 0)      # 真值的前景点坐标

    if len(pred_points) == 0 or len(gt_points) == 0:
        # return float('inf')
        return np.nan

    # 计算从 pred 到 gt 的最大最小距离
    distances_pred_to_gt = cdist(pred_points, gt_points)  # 所有点对的欧几里得距离
    min_distances_pred_to_gt = distances_pred_to_gt.min(axis=1)  # pred 到 gt 的最短距离
    # max_min_dist_pred_to_gt = min_distances_pred_to_gt.max()  # 最大值

    # 计算从 gt 到 pred 的最大最小距离
    distances_gt_to_pred = cdist(gt_points, pred_points)  # 所有点对的欧几里得距离
    min_distances_gt_to_pred = distances_gt_to_pred.min(axis=1)  # gt 到 pred 的最短距离
    # max_min_dist_gt_to_pred = min_distances_gt_to_pred.max()  # 最大值

     # 计算 95% 分位数距离
    hd95_pred_to_gt = np.percentile(min_distances_pred_to_gt, 95)  # pred 到 gt 的 95% 分位数
    hd95_gt_to_pred = np.percentile(min_distances_gt_to_pred, 95)  # gt 到 pred 的 95% 分位数


    # Hausdorff Distance 是两者的最大值
    hd = max(hd95_pred_to_gt, hd95_gt_to_pred)
    return hd



def compute_betti(y_scores, y_true, relative=True, comparison='union', filtration='superlevel', construction='V'):
    y_scores = post_process_output(y_scores)
    y_true = post_process_label(y_true)
    # y_scores = prun(y_scores,4)
    # y_true = prun(y_true,4)
    BM = BettiMatching(y_scores, y_true, relative=relative, comparison=comparison, filtration=filtration,
                       construction=construction)

    return [BM.loss(dimensions=[0, 1]), BM.loss(dimensions=[0]), BM.loss(dimensions=[1]), BM.Betti_number_error(
        threshold=0.5, dimensions=[0, 1]), BM.Betti_number_error(threshold=0.5, dimensions=[0]), BM.Betti_number_error(
        threshold=0.5, dimensions=[1])]


def filter_small_holes_func(mask_label, label_num, is_fore=True):
    left_num = 0
    filter_num = 800  # Background, small holes. #20
    if is_fore:  # Foreground, small line segments.
        filter_num = 100  #30
    for i in range(1, label_num + 1):
        if np.sum(mask_label == i) > filter_num:
            left_num += 1
    # print("left_num----",left_num)
    return left_num

def get_betti_own(x, is_show=False, filter_small_holes=False):  # binary_image  foreground 1， background 0
    # The 0th Betti number 𝑏0 represents the number of connected components, is equivalent to counting the number of connected components in the foreground.
    # The 1st Betti number 𝑏1 represents the number of holes, is equivalent to counting the number of connected components in the background.
    # the 2nd Betti number 𝑏2 represents the number of cavities.
    if len(x.shape)==2:
        mask_label_0, label_num_0 = measure.label(x, connectivity=2, background=0, return_num=True)  # label foreground connected regions
        mask_label_1, label_num_1 = measure.label(x, connectivity=2, background=1,return_num=True)  # label background connected regions
    else:
        mask_label_0, label_num_0 = measure.label(x, connectivity=2, background=0, return_num=True)  # label foreground connected regions
        mask_label_1, label_num_1 = measure.label(x, connectivity=1, background=1,return_num=True)  # label background connected regions
    if is_show:  # show case

        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1), plt.imshow(x, cmap='plasma'), plt.axis("off")
        plt.subplot(1, 3, 2), plt.imshow(mask_label_0, cmap='plasma'), plt.axis("off")
        plt.subplot(1, 3, 3), plt.imshow(mask_label_1, cmap='plasma'), plt.axis("off")
        plt.show()

    if filter_small_holes:
        label_num_0_filter = filter_small_holes_func(mask_label_0, label_num_0, is_fore=True)
        label_num_1_filter = filter_small_holes_func(mask_label_1, label_num_1, is_fore=False)
        return label_num_0_filter, label_num_1_filter
    return label_num_0, label_num_1



def compute_bettis_own(pred, label, filter_small_holes=False):
    if len(pred.shape)==2:
        filter_small_holes = True
        pred = post_process_output(pred) # for 2D
        label = post_process_label(label)
        # pred = prun(pred, 4)
        # label = prun(label, 4)

    else:
        pred = post_process_output_3d(pred) # for 2D
        label = post_process_label_3d(label)

      
    label_betti0, label_betti1 = get_betti_own(label, filter_small_holes=filter_small_holes)

    pred_betti0, pred_betti1 = get_betti_own(pred, filter_small_holes=filter_small_holes)
    # print("label---label_betti0, label_betti1:",  label_betti0, label_betti1  )
    # print("pred---pred_betti0, pred_betti1:",  pred_betti0, pred_betti1  )

    betti0_error = abs(label_betti0 - pred_betti0)
    betti1_error = abs(label_betti1 - pred_betti1)
    return betti0_error + betti1_error, betti0_error, betti1_error



def compute_betti_numbers_3d(binary_image):
    """
    """
    # 生成一个6-邻域结构，用于定义3D空间的连通性。
    structure = generate_binary_structure(3, 1)  # 使用6邻域
    # print("binary_image.shape, structure---------------",  binary_image.shape, structure.shape )

    # Betti-0: 计算前景的连通区域数
    _, betti_0 =  scipy.ndimage.label(binary_image, structure=structure)

    # Betti-1: 计算孔洞数
    inverted_image = np.logical_not(binary_image)
    labeled_holes, betti_1 =  scipy.ndimage.label(inverted_image, structure=structure)

    return betti_0, betti_1


def compute_betti_error_3d(pred, gt):
    """

    """
    # 计算真实标签的Betti数
    B0_true, B1_true = compute_betti_numbers_3d(gt)

    # 计算预测结果的Betti数
    B0_pred, B1_pred = compute_betti_numbers_3d(pred)

    # 计算Betti误差
    betti_error = abs(B0_true - B0_pred) + abs(B1_true - B1_pred)

    return betti_error,  abs(B0_true - B0_pred),  abs(B1_true - B1_pred)

def compute_metrics(reference_file: str, prediction_file: str, image_reader_writer: BaseReaderWriter,
                    labels_or_regions: Union[List[int], List[Union[int, Tuple[int, ...]]]],
                    ignore_label: int = None) -> dict:
    
    # print("---------reference_file", reference_file)
    # print("---------prediction_file", prediction_file)
    # print("---------labels_or_regions", labels_or_regions)
    # print("---------ignore_label", ignore_label)

    # load images
    seg_ref, seg_ref_dict = image_reader_writer.read_seg(reference_file)
    seg_pred, seg_pred_dict = image_reader_writer.read_seg(prediction_file)
    # spacing = seg_ref_dict['spacing']

    ignore_mask = seg_ref == ignore_label if ignore_label is not None else None
    # print("---------ignore_mask", ignore_mask) #None
    # print("---------seg_ref", seg_ref.shape) #[1, 1, 605, 700)


    results = {}
    results['reference_file'] = reference_file
    results['prediction_file'] = prediction_file
    results['metrics'] = {}
    for r in labels_or_regions:
        results['metrics'][r] = {}
        mask_ref = region_or_label_to_mask(seg_ref, r)
        mask_pred = region_or_label_to_mask(seg_pred, r)
        
        mask_ref_background = 1 - mask_ref
        mask_ref_onehot = np.concatenate([mask_ref_background, mask_ref], axis=1)   

        mask_pred_background = 1 - mask_pred
        mask_pred_onehot = np.concatenate([mask_pred_background, mask_pred], axis=1)   
        
    

        # print("---------mask_ref", mask_ref.shape, np.max(mask_ref),  np.min(mask_ref)) #2D-[1, 1, 605, 700) true, false   3D-(1, 31, 62, 88) 
        # print("---------mask_pred", mask_pred.shape) #[1, 1, 605, 700)

        tp, fp, fn, tn = compute_tp_fp_fn_tn(mask_ref, mask_pred, ignore_mask)
        if tp + fp + fn == 0:
            results['metrics'][r]['Dice'] = np.nan
            results['metrics'][r]['IoU'] = np.nan
            results['metrics'][r]['Acc'] = np.nan
        else:
            results['metrics'][r]['Dice'] = 2 * tp / (2 * tp + fp + fn)
            results['metrics'][r]['IoU'] = tp / (tp + fp + fn)
            results['metrics'][r]['Acc'] = (tp + tn) / (fp + fn + tp + tn)

        if len(np.squeeze(mask_ref).shape) == 2:
            mask_pred = mask_pred[0,0]
            mask_ref = mask_ref[0,0]
        if len(np.squeeze(mask_ref).shape) == 3:
            mask_pred = mask_pred[0]
            mask_ref = mask_ref[0]
              
        results['metrics'][r]['clDice'] = clDice(mask_pred, mask_ref)
        results['metrics'][r]['HD'] = compute_hausdorff_distance_fast(mask_pred, mask_ref)
        results['metrics'][r]['NSD'] = compute_nsd(mask_pred, mask_ref)
        # hd = monai.metrics.compute_hausdorff_distance(mask_pred, mask_ref, percentile=95, directed=True)
        # results['metrics'][r]['HD2'] = hd[0][0].numpy().item()
        # print("---------HD:",  results['metrics'][r]['HD'])
        
        if len(mask_pred.shape)==2:
            # print("-----compute_bettis_own_2d----")
            Betti_error = compute_bettis_own(mask_pred, mask_ref)
        elif len(mask_pred.shape)==3:
            # print("-----compute_betti_error_3d----")
            # Betti_error = compute_bettis_own(mask_pred, mask_ref)
            Betti_error = compute_betti_error_3d(mask_pred, mask_ref)
        # print("-----compute_betti_error_3d finish----")


        results['metrics'][r]['Betti'] =  Betti_error[0]
        results['metrics'][r]['Betti_0'] =  Betti_error[1]
        results['metrics'][r]['Betti_1'] =  Betti_error[2]


        results['metrics'][r]['FP'] = fp
        results['metrics'][r]['TP'] = tp
        results['metrics'][r]['FN'] = fn
        results['metrics'][r]['TN'] = tn
        results['metrics'][r]['n_pred'] = fp + tp
        results['metrics'][r]['n_ref'] = fn + tp
    return results


def compute_metrics_on_folder(folder_ref: str, folder_pred: str, output_file: str,
                              image_reader_writer: BaseReaderWriter,
                              file_ending: str,
                              regions_or_labels: Union[List[int], List[Union[int, Tuple[int, ...]]]],
                              ignore_label: int = None,
                              num_processes: int = default_num_processes,
                              chill: bool = True) -> dict:
    """
    output_file must end with .json; can be None
    """
    # print("---------regions_or_labels", regions_or_labels)
    # print("---------ignore_label", ignore_label)
    # print("---------num_processes", num_processes)
    # print("---------chill", chill) #true
    # print("---------cpu_count",  os.cpu_count())



    if output_file is not None:
        assert output_file.endswith('.json'), 'output_file should end with .json'
    files_pred = subfiles(folder_pred, suffix=file_ending, join=False)
    files_ref = subfiles(folder_ref, suffix=file_ending, join=False)
    if not chill:
        present = [isfile(join(folder_pred, i)) for i in files_ref]
        assert all(present), "Not all files in folder_pred exist in folder_ref"
    files_ref = [join(folder_ref, i) for i in files_pred]
    files_pred = [join(folder_pred, i) for i in files_pred]
    with multiprocessing.get_context("spawn").Pool(num_processes) as pool: #-----------
        # for i in list(zip(files_ref, files_pred, [image_reader_writer] * len(files_pred), [regions_or_labels] * len(files_pred), [ignore_label] * len(files_pred))):
        #     compute_metrics(*i)
        results = pool.starmap(
            compute_metrics,
            list(zip(files_ref, files_pred, [image_reader_writer] * len(files_pred), [regions_or_labels] * len(files_pred),
                     [ignore_label] * len(files_pred)))
        )

    # mean metric per class
    metric_list = list(results[0]['metrics'][regions_or_labels[0]].keys())
    means = {}
    for r in regions_or_labels:
        means[r] = {}
        for m in metric_list:
            means[r][m] = np.nanmean([i['metrics'][r][m] for i in results])

    # foreground mean
    foreground_mean = {}
    for m in metric_list:
        values = []
        for k in means.keys():
            if k == 0 or k == '0':
                continue
            values.append(means[k][m])
        foreground_mean[m] = np.mean(values)

    [recursive_fix_for_json_export(i) for i in results]
    recursive_fix_for_json_export(means)
    recursive_fix_for_json_export(foreground_mean)
    result = {'metric_per_case': results, 'mean': means, 'foreground_mean': foreground_mean}
    if output_file is not None:
        save_summary_json(result, output_file)
    return result
    # print('DONE')


def compute_metrics_on_folder2(folder_ref: str, folder_pred: str, dataset_json_file: str, plans_file: str,
                               output_file: str = None,
                               num_processes: int = default_num_processes,
                               chill: bool = False):
    dataset_json = load_json(dataset_json_file)
    # get file ending
    file_ending = dataset_json['file_ending']

    # get reader writer class
    example_file = subfiles(folder_ref, suffix=file_ending, join=True)[0]
    rw = determine_reader_writer_from_dataset_json(dataset_json, example_file)()

    # maybe auto set output file
    if output_file is None:
        output_file = join(folder_pred, 'summary.json')

    lm = PlansManager(plans_file).get_label_manager(dataset_json)
    compute_metrics_on_folder(folder_ref, folder_pred, output_file, rw, file_ending,
                              lm.foreground_regions if lm.has_regions else lm.foreground_labels, lm.ignore_label,
                              num_processes, chill=chill)


def compute_metrics_on_folder_simple(folder_ref: str, folder_pred: str, labels: Union[Tuple[int, ...], List[int]],
                                     output_file: str = None,
                                     num_processes: int = default_num_processes,
                                     ignore_label: int = None,
                                     chill: bool = False):
    example_file = subfiles(folder_ref, join=True)[0]
    file_ending = os.path.splitext(example_file)[-1]
    rw = determine_reader_writer_from_file_ending(file_ending, example_file, allow_nonmatching_filename=True,
                                                  verbose=False)()
    # maybe auto set output file
    if output_file is None:
        output_file = join(folder_pred, 'summary.json')
    compute_metrics_on_folder(folder_ref, folder_pred, output_file, rw, file_ending,
                              labels, ignore_label=ignore_label, num_processes=num_processes, chill=chill)


def evaluate_folder_entry_point():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('gt_folder', type=str, help='folder with gt segmentations')
    parser.add_argument('pred_folder', type=str, help='folder with predicted segmentations')
    parser.add_argument('-djfile', type=str, required=True,
                        help='dataset.json file')
    parser.add_argument('-pfile', type=str, required=True,
                        help='plans.json file')
    parser.add_argument('-o', type=str, required=False, default=None,
                        help='Output file. Optional. Default: pred_folder/summary.json')
    parser.add_argument('-np', type=int, required=False, default=default_num_processes,
                        help=f'number of processes used. Optional. Default: {default_num_processes}')
    parser.add_argument('--chill', action='store_true', help='dont crash if folder_pred doesnt have all files that are present in folder_gt')
    args = parser.parse_args()
    compute_metrics_on_folder2(args.gt_folder, args.pred_folder, args.djfile, args.pfile, args.o, args.np, chill=args.chill)


def evaluate_simple_entry_point():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('gt_folder', type=str, help='folder with gt segmentations')
    parser.add_argument('pred_folder', type=str, help='folder with predicted segmentations')
    parser.add_argument('-l', type=int, nargs='+', required=True,
                        help='list of labels')
    parser.add_argument('-il', type=int, required=False, default=None,
                        help='ignore label')
    parser.add_argument('-o', type=str, required=False, default=None,
                        help='Output file. Optional. Default: pred_folder/summary.json')
    parser.add_argument('-np', type=int, required=False, default=default_num_processes,
                        help=f'number of processes used. Optional. Default: {default_num_processes}')
    parser.add_argument('--chill', action='store_true', help='dont crash if folder_pred doesnt have all files that are present in folder_gt')

    args = parser.parse_args()
    compute_metrics_on_folder_simple(args.gt_folder, args.pred_folder, args.l, args.o, args.np, args.il, chill=args.chill)




# Post processing
def prun(image, kernel_size):
    """
    Remove small forks
    """
    label_map, num_label = measure.label(image, connectivity=1, background=1, return_num=True)
    result = np.zeros(label_map.shape)
    for i in range(1, num_label + 1):
        tmp = np.zeros(label_map.shape)
        tmp[label_map == i] = 1
        D_kernel = np.ones((kernel_size, kernel_size), np.uint8)
        dil = cv2.dilate(tmp, D_kernel)
        dst = cv2.erode(dil, D_kernel)
        result[dst == 1] = 255
    result = 255 - result
    result[result == 255] = 1
    result = np.uint8(result)
    return result


def post_process_label(in_img: np.ndarray, prun=False) -> np.ndarray:
    out_img = morphology.skeletonize(in_img, method="lee")
    out_img = morphology.dilation(out_img, morphology.square(3))
    if prun:
        out_img = prun(out_img, 4)  # 5
    return out_img


def post_process_output(in_img: np.ndarray, prun=False) -> np.ndarray:
    out_img = morphology.dilation(in_img, morphology.square(3))  # 2  3  #6
    out_img = morphology.skeletonize(out_img, method="lee")
    out_img = morphology.dilation(out_img, morphology.square(3))  # 5  3
    if prun:
        out_img = prun(out_img, 4)  # 5
    return out_img



def post_process_label_3d(in_img):
    """
    Perform 3D skeletonization and dilation on the input binary image.
    
    Parameters:
        in_img (numpy.ndarray): Input 3D binary image (values should be 0 or 1).
        
    Returns:
        numpy.ndarray: Processed 3D binary image after skeletonization and dilation.
    """
    # 3D skeletonization
    out_img = skeletonize_3d(in_img)
    
    # 3D dilation with a 3x3x3 cube structuring element
    out_img = binary_dilation(out_img, cube(3))
    
    return out_img

def post_process_output_3d(in_img):
    """
    Perform 3D skeletonization and dilation on the input binary image.
    
    Parameters:
        in_img (numpy.ndarray): Input 3D binary image (values should be 0 or 1).
        
    Returns:
        numpy.ndarray: Processed 3D binary image after skeletonization and dilation.
    """

    out_img = binary_dilation(in_img, cube(3))


    # 3D skeletonization
    out_img = skeletonize_3d(out_img)
    
    # 3D dilation with a 3x3x3 cube structuring element
    out_img = binary_dilation(out_img, cube(3))
    
    return out_img




if __name__ == '__main__':
    folder_ref = '/media/fabian/data/nnUNet_raw/Dataset004_Hippocampus/labelsTr'
    folder_pred = '/home/fabian/results/nnUNet_remake/Dataset004_Hippocampus/nnUNetModule__nnUNetPlans__3d_fullres/fold_0/validation'
    output_file = '/home/fabian/results/nnUNet_remake/Dataset004_Hippocampus/nnUNetModule__nnUNetPlans__3d_fullres/fold_0/validation/summary.json'
    image_reader_writer = SimpleITKIO()
    file_ending = '.nii.gz'
    regions = labels_to_list_of_regions([1, 2])
    ignore_label = None
    num_processes = 12
    compute_metrics_on_folder(folder_ref, folder_pred, output_file, image_reader_writer, file_ending, regions, ignore_label,
                              num_processes)
