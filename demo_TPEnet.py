# 2020/8/11
# Jungwon Kang


import os
import re
import pickle
import cv2
import json
import numpy as np
import copy
import sys
import nums_from_string
import torch


import PE_TPEnet
from helpers.utils import my_utils_img



###==================================================================================================================
### 1. set parameters
###==================================================================================================================
dim_instance_vectors = 16
flag_save_img        = 0
flag_save_data       = 0
data_in_use          = 1   # 0 for RailSem19 - 1 for others in which GT data is missing

###==================================================================================================================
### 2. init
###==================================================================================================================
obj_my_utils_img = my_utils_img.MyUtils_Image()

###==================================================================================================================
### 3. loop
###==================================================================================================================
dir_input           = "./sample_input_imgs/validation"
dir_weight          = "./net_weight/Mybest_7000.pkl"

list_fnames_img = os.listdir(dir_input)
list_fnames_img.sort(key=lambda f: int(re.sub('\D', '', f)))

PathExtractor = PE_TPEnet.PathExtraction_TPEnet(dim_instance_vectors, dir_weight)
for my_idx,fname_img_in in enumerate(list_fnames_img):
    if my_idx == 51:
        pass
    else:
        continue
    ##------------------------------------------------------------------------------------------------
    ### 3-1. read img from file
    ###------------------------------------------------------------------------------------------------
    full_fname_img_ori = os.path.join(dir_input, fname_img_in)

    print("Read Input Image from : {}".format(full_fname_img_ori))


    img_raw_rsz_uint8    = cv2.imread(full_fname_img_ori)
    # binary_seg_mask_json = json.load(open('./instance_segmentation_binary_json/rs' + f"{my_idx+6000:05d}" + '.txt', 'r'))
    binary_seg_mask_json = json.load(open('./instance_seg_bin_json_reduced/rs' + f"{my_idx + 6000:05d}" + '.txt', 'r'))

    ###------------------------------------------------------------------------------------------------
    ### 3-2. process
    ###------------------------------------------------------------------------------------------------
    jojo = PathExtractor.process(img_raw_rsz_uint8,binary_seg_mask_json)    # img_raw_rsz_uint8: sensor data
    # print(type(jojo))
    # print(jojo.shape)


    ###------------------------------------------------------------------------------------------------
    ### 3.3 PERFORMANCE METRICS CREATION
    ###------------------------------------------------------------------------------------------------
    if 0:
        img_idx = nums_from_string.get_nums(list_fnames_img[my_idx])[0]
        if data_in_use == 0 or (data_in_use == 1 and flag_ydhr == True):

            if flag_ydhr == True:
                dict_pathlabel_gt_this = list_pathlabel_gt_in[img_idx]


                gt_idx_time_this                  = dict_pathlabel_gt_this['idx_time_this']         # sequential index (0,1,2...)
                gt_fname_img_in_only              = dict_pathlabel_gt_this['fname_img_in_only']
                gt_raw_dict_xs_img_rail_LR        = dict_pathlabel_gt_this['dict_rail_pnt_x_img']
                gt_raw_dict_XYZ_pnt_in_cam_rail_L = dict_pathlabel_gt_this['dict_xyz_pnt_rail_left_in_cam']
                gt_raw_dict_XYZ_pnt_in_cam_rail_R = dict_pathlabel_gt_this['dict_xyz_pnt_rail_right_in_cam']

                gt_final_dict_xs_img_rail_LR, \
                gt_final_dict_XYZ_pnt_in_cam_rail_L,\
                gt_final_dict_XYZ_pnt_in_cam_rail_R = obj_helper_GT.get_gt_final(gt_raw_dict_xs_img_rail_LR,
                                                                                 gt_raw_dict_XYZ_pnt_in_cam_rail_L,
                                                                                 gt_raw_dict_XYZ_pnt_in_cam_rail_R)

                IoU_rail_region = 0
                IoU_rail        = 0
                IoU_background  = 0

            else:
                gt_final_dict_xs_img_rail_LR = json.load(open("railsem_jsons_test_modified2/railsem_jsons_test_modified" + str(my_idx) + ".json", 'r'))
                if num_seg_classes == 3:
                    gt_segmentation = cv2.imread("./rs19_val_modified/rs" + f"{my_idx+7000:05d}" + ".png", cv2.IMREAD_GRAYSCALE)
                if num_seg_classes == 19:
                    gt_segmentation = cv2.imread("./rs19_val/rs" + f"{my_idx + 7000:05d}" + ".png", cv2.IMREAD_GRAYSCALE)
                gt_segmentation = cv2.resize(gt_segmentation, (img_raw_rsz_uint8.shape[1], img_raw_rsz_uint8.shape[0]))

                evaluator_seg = evaluation_utils.eval_seg_object(gt_segmentation, labels_seg_predicted,
                                                                 image_height=img_raw_rsz_uint8.shape[0],
                                                                 image_width=img_raw_rsz_uint8.shape[1])

                if num_seg_classes == 3:
                    IoU_rail_region = evaluator_seg.calculate_IoU(class_this=0)
                    IoU_rail = evaluator_seg.calculate_IoU(class_this=1)
                    IoU_background = evaluator_seg.calculate_IoU(class_this=2)

                if num_seg_classes == 19:
                    IoU_rail_region = evaluator_seg.calculate_IoU(class_this=12)
                    IoU_rail = evaluator_seg.calculate_IoU(class_this=17)
                    IoU_background = evaluator_seg.calculate_IoU(class_this=3)

            ### 3.5.1 create evaluator objects
            evaluator_topolgy = evaluation_utils.eval_object_topology(gt_final_dict_xs_img_rail_LR, list_res_paths, image_height = img_raw_rsz_uint8.shape[0], image_width = img_raw_rsz_uint8.shape[1])
            # evaluator_seg = evaluation_utils.eval_seg_object(gt_segmentation, labels_seg_predicted, image_height = img_raw_rsz_uint8.shape[0], image_width = img_raw_rsz_uint8.shape[1])

            ### 3.5.2 annotate ground-truth rail area
            annotated_im, y_minimum = evaluator_topolgy.annotate_gt(final_im)

            ### 3.5.3 find correspondences between ground-truth and detected rail
            matching_mat, matched_ones = evaluator_topolgy.find_matches(6, y_minimum)

            ### 3.5.4 find true positives, false positives, and false negatives
            TP,FP,FN = evaluator_topolgy.performance_metrics_values_TP_level(matching_mat,matched_ones)
            path_level_prec, path_level_recall = evaluator_topolgy.performance_metrics_values_path_level(matched_ones, min_rate=0)
            all_pixel_prec, all_pixel_recall = evaluator_topolgy.performance_metrics_values_all_pixel_level(matching_mat,matched_ones)

            # evaluator_all_pixel_level = evaluation_utils.eval_object_all_pixel_level(gt_final_dict_xs_img_rail_LR, list_res_paths, image_height= img_raw_rsz_uint8.shape[0], image_width=img_raw_rsz_uint8.shape[1])
            # all_pixel_prec, all_pixel_recall   = evaluator_all_pixel_level.find_matches(6, y_minimum)

            ### 3.5.5 show performance evaluation results on the annotated image
            # image_showing_evaluation_res = evaluator_topolgy.create_final_result_on_annotated_image_V2(annotated_im, matching_mat, matched_ones)
            image_showing_evaluation_res = evaluator_topolgy.create_final_result_on_annotated_image_V1(final_im, matched_ones)
            # image_showing_evaluation_res = evaluator_topolgy.create_final_result_on_annotated_image_V0(final_im)

            # image_showing_evaluation_res = cv2.putText(image_showing_evaluation_res, 'Image index: %d' % img_idx, (400, 25),
            #                                            cv2.FONT_HERSHEY_SIMPLEX,
            #                                            0.75, (255, 0, 0), 1, cv2.LINE_AA)

            ### 3.5.6 measure segmentation IoU
            # if num_seg_classes == 3:
            #     IoU_rail_region = evaluator_seg.calculate_IoU(class_this=0)
            #     IoU_rail        = evaluator_seg.calculate_IoU(class_this=1)
            #     IoU_background  = evaluator_seg.calculate_IoU(class_this=2)
            #
            # if num_seg_classes == 19:
            #     IoU_rail_region = evaluator_seg.calculate_IoU(class_this=12)
            #     IoU_rail        = evaluator_seg.calculate_IoU(class_this=17)
            #     IoU_background  = evaluator_seg.calculate_IoU(class_this=3)

            if (TP+FP) > 0:
                # image_showing_evaluation_res = cv2.putText(image_showing_evaluation_res, 'TP Pixel-level Precision: %f' % (TP/(TP+FP)), (300, 25), cv2.FONT_HERSHEY_SIMPLEX,
                #                           0.75, (0, 0, 255), 1, cv2.LINE_AA)
                # image_showing_evaluation_res = cv2.putText(image_showing_evaluation_res, 'TP Pixel-level Recall: %f' % (TP/(TP+FN)), (300, 50), cv2.FONT_HERSHEY_SIMPLEX,
                #                           0.75, (0, 0, 255), 1, cv2.LINE_AA)
                res_eval.append(
                    {"id": img_idx, "TP": TP, "FP": FP, "FN": FN, "precision": (TP / (TP + FP)), "recall": (TP / (TP + FN)),
                     "IoU_rail_region": IoU_rail_region, "IoU_rail": IoU_rail, "IoU_background": IoU_background,
                     "path_level_prec": path_level_prec, "path_level_recall": path_level_recall,
                     "all_pixel_prec": all_pixel_prec, "all_pixel_recall": all_pixel_recall,
                     "time_net": dict_res_time["dtime_ab"], "time_pp": dict_res_time["dtime_bc"]})
            else:
                # image_showing_evaluation_res = cv2.putText(image_showing_evaluation_res, 'Precision: %f' % 0, (400, 50), cv2.FONT_HERSHEY_SIMPLEX,
                #                           0.75, (255, 0, 0), 1, cv2.LINE_AA)
                # image_showing_evaluation_res = cv2.putText(image_showing_evaluation_res, 'Recall: %f' % 0, (400, 75), cv2.FONT_HERSHEY_SIMPLEX,
                #                           0.75, (255, 0, 0), 1, cv2.LINE_AA)
                res_eval.append(
                    {"id": img_idx, "TP": TP, "FP": FP, "FN": FN, "precision": 0, "recall": 0,
                     "IoU_rail_region": IoU_rail_region, "IoU_rail": IoU_rail, "IoU_background": IoU_background,
                     "path_level_prec": path_level_prec, "path_level_recall": path_level_recall,
                     "all_pixel_prec": all_pixel_prec, "all_pixel_recall": all_pixel_recall,
                     "time_net": dict_res_time["dtime_ab"], "time_pp": dict_res_time["dtime_bc"]})
        else:
            # pass
            image_showing_evaluation_res = PathExtractor.show_final_path_on_ori_noGTdata(img_raw_rsz_uint8,list_res_paths)


    if flag_save_img == 1:
        cv2.imwrite("IMG/resluting_image_" + str(img_idx) + ".jpg", image_showing_evaluation_res)
        cv2.imwrite("SEG/resluting_image_" + str(img_idx) + ".bmp", img_res_seg)
        cv2.imwrite("CEN/resluting_image_" + str(img_idx) + ".png", img_res_centerness)



###------------------------------------------------------------------------------------------------
### 4. SAVE PERFORMANCE METRICS DATA
###------------------------------------------------------------------------------------------------
sum_prec = 0
sum_rec = 0

sum_path_prec = 0
sum_path_recall = 0

sum_all_prec = 0
sum_all_recall = 0

sum_IoU_rail_region = 0
sum_IoU_rail        = 0
sum_IoU_background  = 0

sum_time_net = 0
sum_time_pp  = 0
if flag_save_data == 1 and (data_in_use == 0 or (data_in_use == 1 and flag_ydhr == True)):
    with open('../../Performance Metrics/precision_1.txt', 'w') as f:
        for item in res_eval:
            prec = item["precision"]
            f.write('%f' % prec)
            f.write("\n")

            sum_prec = sum_prec + prec

            sum_IoU_rail_region += item["IoU_rail_region"]
            sum_IoU_rail        += item["IoU_rail"]
            sum_IoU_background  += item["IoU_background"]

            sum_path_prec   += item["path_level_prec"]
            sum_path_recall += item["path_level_recall"]

            sum_all_prec   += item["all_pixel_prec"]
            sum_all_recall += item["all_pixel_recall"]

            sum_time_net += item["time_net"]
            sum_time_pp  += item["time_pp"]

    with open('../../Performance Metrics/recall_1.txt', 'w') as f:
        for item in res_eval:
            rec = item["recall"]
            f.write('%f' % rec)
            f.write("\n")
            sum_rec = sum_rec + rec

    with open('../../Performance Metrics/TP_1.txt', 'w') as f:
        for item in res_eval:
            TP = item["TP"]
            f.write('%d' % TP)
            f.write("\n")

    with open('../../Performance Metrics/FP_1.txt', 'w') as f:
        for item in res_eval:
            FP = item["FP"]
            f.write('%d' % FP)
            f.write("\n")

    with open('../../Performance Metrics/FN_1.txt', 'w') as f:
        for item in res_eval:
            FN = item["FN"]
            f.write('%d' % FN)
            f.write("\n")


    avg_precision = sum_prec/len(res_eval)
    avg_recall = sum_rec/len(res_eval)

    mIoU_rail_region = sum_IoU_rail_region/len(res_eval)
    mIoU_rail        = sum_IoU_rail / len(res_eval)
    mIoU_background  = sum_IoU_background / len(res_eval)

    avg_path_precision = sum_path_prec / len(res_eval)
    avg_path_recall    = sum_path_recall / len(res_eval)

    avg_all_precision = sum_all_prec / len(res_eval)
    avg_all_recall    = sum_all_recall / len(res_eval)

    avg_time_net = sum_time_net / len(res_eval)
    avg_time_pp  = sum_time_pp / len(res_eval)

    print("TP PIXEL LEVEL [AVERAGE] PRECISION AND RECALL")
    print(avg_precision)
    print(avg_recall)
    print("SEGMENTATION PERFORMANCE")
    print(mIoU_rail_region)
    print(mIoU_rail)
    print(mIoU_background)
    print("ALL PIXEL LEVEL [AVERAGE] PRECISION AND RECALL")
    print(avg_all_precision)
    print(avg_all_recall)
    print("PATH LEVEL [AVERAGE] PRECISION AND RECALL")
    print(avg_path_precision)
    print(avg_path_recall)
    print("DURATION RESULTS")
    print(avg_time_net)
    print(avg_time_pp)


