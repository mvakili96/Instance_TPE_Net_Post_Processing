import numpy as np
import cv2
import PE_TPEnet
import my_args_TPEnet
import copy
import torch
import json
import torch.nn.functional as F
import torch.nn as nn
from scipy.signal import find_peaks



class eval_object_topology:

    def __init__(self, gt_data, detected_data, image_height = 540, image_width = 960, arch = None):
        self.gt_data = gt_data
        self.detected_data = detected_data
        self.image_height = image_height
        self.image_width = image_width
        self.tot_num_gt_paths = len(gt_data)
        self.tot_num_detected_paths = len(detected_data)

        ### set dataset to be used -> 0(YHDR), 1(NYC)
        DATASET_for_use = 0

        ### define args
        parser_oper = my_args_TPEnet.define_args_operation(DATASET_for_use, arch)
        parser_alg = my_args_TPEnet.define_args_algorithm(DATASET_for_use, arch)

        ### parse
        args_alg = parser_alg.parse_args()

        ### set values for some args
        args_alg = my_args_TPEnet.set_value_for_args_algorithm(DATASET_for_use, args_alg)

        # self.PathExtractor = PE_TPEnet.PathExtraction_TPEnet(args_alg)

    def find_matches(self, search_area_offset, y_min):
        if self.tot_num_detected_paths == 0:
            return [],[]
        matching_matrix = [[{"TP": None, "FP": None, "FN": None, "precision":None, "recall":None, "F1":None, "matched_detected":[], "unmatched_detected":[]} for y in range(self.tot_num_detected_paths)] for x in range(self.tot_num_gt_paths)]

        for cnt_gt in range(self.tot_num_gt_paths):
            gt_LR_rail_this = list(self.gt_data.values())[cnt_gt]
            for cnt_det in range(self.tot_num_detected_paths):
                TP = 0
                FN = 0
                FP = 0
                for LR_index in range(2):
                    bin_image = self.create_binary_image(cnt_det,LR_index,y_min)
                    for cnt_point in range(len(gt_LR_rail_this)):
                        ximg_gt = gt_LR_rail_this[cnt_point][LR_index]

                        if (ximg_gt < 0) or (ximg_gt > self.image_width):
                            continue

                        flag_match = 0
                        for i in range(cnt_point - search_area_offset, cnt_point + search_area_offset + 1):
                            for j in range(ximg_gt - search_area_offset, ximg_gt + search_area_offset + 1):
                                if (i < 270) or (i >= self.image_height) or (j < 0) or (j >= self.image_width):
                                    continue

                                if bin_image[i, j] != 0:
                                    bin_image[i, j] = 0.5
                                    flag_match = 1
                                    matching_matrix[cnt_gt][cnt_det]["matched_detected"].append([i,j])


                        if flag_match == 1:
                            TP = TP + 1
                        else:
                            FN = FN + 1
                    FP = FP + np.count_nonzero(bin_image == 1)
                    unmatched_loc = np.argwhere(bin_image == 1)
                    for item in unmatched_loc:
                        matching_matrix[cnt_gt][cnt_det]["unmatched_detected"].append(item)



                matching_matrix[cnt_gt][cnt_det]["TP"] = TP
                matching_matrix[cnt_gt][cnt_det]["FP"] = FP
                matching_matrix[cnt_gt][cnt_det]["FN"] = FN
                if TP > 0:
                    matching_matrix[cnt_gt][cnt_det]["precision"] = (TP/(TP+FP))
                    matching_matrix[cnt_gt][cnt_det]["recall"]    = (TP/(TP+FN))
                    matching_matrix[cnt_gt][cnt_det]["F1"]        = (2*TP/(2*TP + FN + FP))
                else:
                    matching_matrix[cnt_gt][cnt_det]["precision"] = 0
                    matching_matrix[cnt_gt][cnt_det]["recall"]    = 0
                    matching_matrix[cnt_gt][cnt_det]["F1"]        = 0

        matched_pairs = []
        matching_matrix_F1 = np.array([[matching_matrix[cnt_gt][cnt_det]["F1"] for cnt_det in range(self.tot_num_detected_paths)] for cnt_gt in range(self.tot_num_gt_paths)])
        for counter in range(max(self.tot_num_gt_paths,self.tot_num_detected_paths)):
            max_f1 = np.amax(matching_matrix_F1)
            max_index = np.argwhere(matching_matrix_F1 == max_f1)[0]
            if max_f1>0:
                matched_pairs.append([max_index,max_f1])
                matching_matrix_F1[max_index[0],:] = -1
                matching_matrix_F1[:, max_index[1]] = -1
        return matching_matrix,matched_pairs


    def create_binary_image(self, detected_path_index, L_or_R, y_min):
        binary_img = np.zeros((self.image_height, self.image_width))

        if L_or_R == 0:
            detected_3d = self.detected_data[detected_path_index]["polynomial"]["xy_left"]
            # detected_3d = self.detected_data[detected_path_index]["extracted"]["xy_left_img"]
        else:
            detected_3d = self.detected_data[detected_path_index]["polynomial"]["xy_right"]
            # detected_3d = self.detected_data[detected_path_index]["extracted"]["xy_right_img"]


        for idx_pnt in range(detected_3d.shape[0]):  # Loop for detected points (left) to create binary image
            x3d_detected = detected_3d[idx_pnt][0]
            y3d_detected = detected_3d[idx_pnt][1]

            # converted = self.PathExtractor.convert_one_point_from_world_to_img(x3d_detected, y3d_detected)
            # ximg_detected = converted[0]
            # yimg_detected = converted[1]

            ximg_detected_int = int(round(x3d_detected))
            yimg_detected_int = int(round(y3d_detected))

            if (ximg_detected_int < 0) or (ximg_detected_int >= self.image_width) or (yimg_detected_int < 0) or (yimg_detected_int >= self.image_height) or (yimg_detected_int < 270) or (yimg_detected_int < y_min):
                continue

            binary_img[yimg_detected_int, ximg_detected_int] = 1



        # if L_or_R == 0:
        #     detected_3d = self.detected_data[detected_path_index]["polynomial"]["xyz_left_3d"]
        # else:
        #     detected_3d = self.detected_data[detected_path_index]["polynomial"]["xyz_right_3d"]
        #
        # for idx_pnt in range(detected_3d.shape[0]):  # Loop for detected points (left) to create binary image
        #     x3d_detected = detected_3d[idx_pnt][0]
        #     y3d_detected = detected_3d[idx_pnt][1]
        #
        #     converted = self.PathExtractor.convert_one_point_from_world_to_img(x3d_detected, y3d_detected)
        #     ximg_detected = converted[0]
        #     yimg_detected = converted[1]
        #
        #     ximg_detected_int = int(round(ximg_detected))
        #     yimg_detected_int = int(round(yimg_detected))
        #
        #     if (ximg_detected_int < 0) or (ximg_detected_int >= self.image_width) or (yimg_detected_int < 0) or (yimg_detected_int >= self.image_height) or (yimg_detected_int < 270):
        #         continue
        #
        #     binary_img[yimg_detected_int, ximg_detected_int] = 1
        return binary_img


    def performance_metrics_values_TP_level(self,match_matrix,match_pairs):
        if len(match_pairs) == 0:
            TP = 0
            FP = 0
            FN = 0
            for cnt_gt in range(self.tot_num_gt_paths):
                gt_LR_rail_this = list(self.gt_data.values())[cnt_gt]
                for LR_index in range(2):
                    for cnt_point in range(len(gt_LR_rail_this)):
                        ximg_gt = gt_LR_rail_this[cnt_point][LR_index]
                        if (ximg_gt < 0) or (ximg_gt > self.image_width):
                            continue
                        FN += 1

            return TP, FP, FN

        TP = 0
        FP = 0
        FN = 0
        for item in match_pairs:
            TP_this = match_matrix[item[0][0]][item[0][1]]["TP"]
            TP += TP_this
            FP_this = match_matrix[item[0][0]][item[0][1]]["FP"]
            FP += FP_this
            FN_this = match_matrix[item[0][0]][item[0][1]]["FN"]
            FN += FN_this

        return TP,FP,FN


    def performance_metrics_values_all_pixel_level(self,match_matrix,match_pairs):
        if len(match_pairs) == 0:
            TP = 0
            FP = 0
            FN = 0
            for cnt_gt in range(self.tot_num_gt_paths):
                gt_LR_rail_this = list(self.gt_data.values())[cnt_gt]
                for LR_index in range(2):
                    for cnt_point in range(len(gt_LR_rail_this)):
                        ximg_gt = gt_LR_rail_this[cnt_point][LR_index]
                        if (ximg_gt < 0) or (ximg_gt > self.image_width):
                            continue
                        FN += 1

            return 0, 0

        TP = 0
        FP = 0
        FN = 0
        for item in match_pairs:
            TP_this = match_matrix[item[0][0]][item[0][1]]["TP"]
            TP += TP_this
            FP_this = match_matrix[item[0][0]][item[0][1]]["FP"]
            FP += FP_this
            FN_this = match_matrix[item[0][0]][item[0][1]]["FN"]
            FN += FN_this


        paired_gt = []
        for item in match_pairs:
            paired_gt.append(item[0][0])
        if self.tot_num_gt_paths > len(match_pairs):
            for counter in range(self.tot_num_gt_paths):
                if counter not in paired_gt:
                    FN += match_matrix[counter][0]["TP"] + match_matrix[counter][0]["FN"]

        paired_det = []
        for item in match_pairs:
            paired_det.append(item[0][1])
        if self.tot_num_detected_paths > len(match_pairs):
            for counter in range(self.tot_num_detected_paths):
                if counter not in paired_det:
                    FP += match_matrix[0][counter]["TP"] + match_matrix[0][counter]["FP"]

        if TP > 0:
            prec   = (TP / (TP + FP))
            recall = (TP / (TP + FN))
        else:
            prec = 0
            recall = 0

        return prec,recall





    def performance_metrics_values_path_level(self, match_pairs, min_rate):
        TP = 0
        for item in match_pairs:
            if item[1] >= min_rate:
                TP += 1
        FP = max(0, self.tot_num_detected_paths - TP)
        FN = max(0, self.tot_num_gt_paths - TP)
        if TP > 0:
            prec   = (TP / (TP + FP))
            recall = (TP / (TP + FN))
        else:
            prec = 0
            recall = 0

        return prec, recall




    def annotate_gt(self,image):
        img = copy.deepcopy(image)
        flag_found_y_min = 0
        for path_idx in range(self.tot_num_gt_paths):
            rail_points_gt = list(self.gt_data.values())[path_idx]
            for h_idx in range(self.image_height):
                ximg_left_gt  = rail_points_gt[h_idx][0]
                ximg_right_gt = rail_points_gt[h_idx][1]

                if (ximg_left_gt < 0) or (ximg_left_gt > self.image_width) or (ximg_right_gt < 0) or (ximg_right_gt > self.image_width) or h_idx < 270:
                    continue
                if flag_found_y_min == 0:
                    y_min = h_idx
                    flag_found_y_min = 1

                # cv2.circle(img, center=(ximg_left_gt, h_idx), radius=1,color=(0, 0, 250),thickness=-1)
                # cv2.circle(img, center=(ximg_right_gt, h_idx), radius=1,color=(250, 0, 0),thickness=-1)

                for x_this in range(ximg_left_gt, (ximg_right_gt + 1)):
                    if img[h_idx, x_this, 0] >= 200:
                        img[h_idx, x_this, :] = [255, 200, 200]
                    else:
                        img[h_idx, x_this, :] = img[h_idx, x_this, :] + [55, 0, 0]

        return img,y_min


    def create_final_result_on_annotated_image_V0(self,raw_image):
        points_ref = []
        for detected_path_index in range(len(self.detected_data)):
            detected_3d_left = self.detected_data[detected_path_index]["polynomial"]["xy_left"]
            detected_3d_right = self.detected_data[detected_path_index]["polynomial"]["xy_right"]
            for idx_pnt in range(detected_3d_left.shape[0]):
                x3d_detected_left = detected_3d_left[idx_pnt][0]
                y3d_detected_left = detected_3d_left[idx_pnt][1]

                x3d_detected_right = detected_3d_right[idx_pnt][0]
                y3d_detected_right = detected_3d_right[idx_pnt][1]

                ximg_detected_int_left = int(round(x3d_detected_left))
                yimg_detected_int_left = int(round(y3d_detected_left))

                ximg_detected_int_right = int(round(x3d_detected_right))
                yimg_detected_int_right = int(round(y3d_detected_right))

                if (ximg_detected_int_left < 0) or (ximg_detected_int_left >= 960) or (yimg_detected_int_left < 0) or (
                        yimg_detected_int_left >= 540) or (yimg_detected_int_left < 270):
                    continue
                if (ximg_detected_int_right < 0) or (ximg_detected_int_right >= 960) or (yimg_detected_int_right < 0) or (
                        yimg_detected_int_right >= 540) or (yimg_detected_int_right < 270):
                    continue

                for x_this in range(ximg_detected_int_left, (ximg_detected_int_right + 1)):
                    if [yimg_detected_int_left, x_this] not in points_ref:
                        if raw_image[yimg_detected_int_left, x_this, 1] >= 200:
                            raw_image[yimg_detected_int_left, x_this, :] = [200, 255, 200]
                        else:
                            raw_image[yimg_detected_int_left, x_this, :] = raw_image[yimg_detected_int_left, x_this, :] + [0, 55, 0]

                        points_ref.append([yimg_detected_int_left, x_this])
                    else:
                        continue

                cv2.circle(raw_image, center=(ximg_detected_int_left, yimg_detected_int_left), radius=2,
                           color=(255, 255, 0), thickness=-1)
                cv2.circle(raw_image, center=(ximg_detected_int_right, yimg_detected_int_right), radius=2,
                           color=(0, 70, 255), thickness=-1)

        return raw_image

    def create_final_result_on_annotated_image_V1(self,raw_image,match_pairs):
        points_ref = []
        # print(match_pairs)
        # print(len(self.detected_data))
        for i, item in enumerate(match_pairs):
            detected_3d_left = self.detected_data[item[0][1]]["polynomial"]["xy_left"]
            detected_3d_right = self.detected_data[item[0][1]]["polynomial"]["xy_right"]
            for idx_pnt in range(detected_3d_left.shape[0]):
                x3d_detected_left = detected_3d_left[idx_pnt][0]
                y3d_detected_left = detected_3d_left[idx_pnt][1]

                x3d_detected_right = detected_3d_right[idx_pnt][0]
                y3d_detected_right = detected_3d_right[idx_pnt][1]

                ximg_detected_int_left = int(round(x3d_detected_left))
                yimg_detected_int_left = int(round(y3d_detected_left))

                ximg_detected_int_right = int(round(x3d_detected_right))
                yimg_detected_int_right = int(round(y3d_detected_right))

                if (ximg_detected_int_left < 0) or (ximg_detected_int_left >= 960) or (yimg_detected_int_left < 0) or (
                        yimg_detected_int_left >= 540) or (yimg_detected_int_left < 270):
                    continue
                if (ximg_detected_int_right < 0) or (ximg_detected_int_right >= 960) or (yimg_detected_int_right < 0) or (
                        yimg_detected_int_right >= 540) or (yimg_detected_int_right < 270):
                    continue

                for x_this in range(ximg_detected_int_left, (ximg_detected_int_right + 1)):
                    if [yimg_detected_int_left, x_this] not in points_ref:
                        if raw_image[yimg_detected_int_left, x_this, 1] >= 200:
                            raw_image[yimg_detected_int_left, x_this, :] = [200, 255, 200]
                        else:
                            raw_image[yimg_detected_int_left, x_this, :] = raw_image[yimg_detected_int_left, x_this, :] + [0, 55, 0]

                        points_ref.append([yimg_detected_int_left, x_this])
                    else:
                        continue

                cv2.circle(raw_image, center=(ximg_detected_int_left, yimg_detected_int_left), radius=2,
                           color=(255, 255, 0), thickness=-1)
                cv2.circle(raw_image, center=(ximg_detected_int_right, yimg_detected_int_right), radius=2,
                           color=(0, 70, 255), thickness=-1)

        return raw_image



    def create_final_result_on_annotated_image_V2(self,img, match_matrix, match_pairs):
        if len(match_pairs) == 0:
            return img

        paired_detected = []
        all_res_images = []
        print(match_pairs)
        for i,item in enumerate(match_pairs):
            img_cp = copy.deepcopy(img)
            TP_points = match_matrix[item[0][0]][item[0][1]]["matched_detected"]
            FP_points = match_matrix[item[0][0]][item[0][1]]["unmatched_detected"]

            for point in TP_points:
                cv2.circle(img_cp, center=(point[1], point[0]), radius=2, color=(0, 255, 0), thickness=-1)

            for point in FP_points:
                cv2.circle(img_cp, center=(point[1], point[0]), radius=3, color=(0, 0, 255), thickness=-1)

            paired_detected.append(item[0][1])
            all_res_images.append(img_cp)

        # if self.tot_num_gt_paths < self.tot_num_detected_paths:
        #     for num in range(self.tot_num_detected_paths):
        #         if num not in paired_detected:
        #             img_cp = copy.deepcopy(img)
        #             for point in match_matrix[0][num]["matched_detected"]:
        #                 cv2.circle(img_cp, center=(point[1], point[0]), radius=3, color=(250, 250, 0), thickness=-1)
        #             for point in match_matrix[0][num]["unmatched_detected"]:
        #                 cv2.circle(img_cp, center=(point[1], point[0]), radius=3, color=(250, 250, 0), thickness=-1)
        #             all_res_images.append(img_cp)

        if self.tot_num_gt_paths > self.tot_num_detected_paths:
            for num in range(self.tot_num_gt_paths-self.tot_num_detected_paths):
                all_res_images.append(img)


        if len(all_res_images) == 1:
            single_image_final = all_res_images[0]
        else:
            input_to_concat = all_res_images[0]
            for cnt in range(len(all_res_images) - 1):
                single_image_final = cv2.vconcat([input_to_concat, all_res_images[cnt + 1]])
                input_to_concat = single_image_final

        return single_image_final




class seg_validation:
    def __init__(self,dir_GT , index, flag_save, device):
        if flag_save:
            full_dir_GT = dir_GT + "raw/rs" + f"{index+7000:05d}" + ".png"
            GT_image_raw = cv2.imread(full_dir_GT)

            GT_image_modified = copy.deepcopy(GT_image_raw)

            for i, row in enumerate(GT_image_modified):
                for j, pix in enumerate(row):
                    if pix[0] == 12 or pix[0] == 3:
                        GT_image_modified[i, j] = [0, 0, 0]  # used to be 0
                    elif pix[0] == 17 or pix[0] == 18:
                        GT_image_modified[i, j] = [1, 1, 1]  # used to be 1
                    else:
                        GT_image_modified[i, j] = [2, 2, 2]  # used to be 2

            cv2.imwrite(dir_GT + "modified/rs" + f"{index:05d}" + ".png", GT_image_modified)
            GT_image = GT_image_modified
        else:
            GT_image = cv2.imread("test_seg/modified/rs" + f"{index:05d}" + ".png")

        GT_image_modified_1channel = cv2.cvtColor(GT_image, cv2.COLOR_BGR2GRAY)
        GT_image_final_numpy = cv2.resize(GT_image_modified_1channel, (960, 540))
        # for item in GT_image_final_numpy[530]:
        #     print(item)
        GT_image_final_numpy = np.array([GT_image_final_numpy])
        GT_image_final = torch.from_numpy(GT_image_final_numpy).long()
        self.GT_image_final = GT_image_final.to(device)

    def calculate_loss(self, input, min_K, loss_th, weight=None, size_average=True):
        n, c, h, w = input.size()
        nt, ht, wt = self.GT_image_final.size()
        batch_size = input.size()[0]

        if h != ht and w != wt:  # upsample labels
            input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)

        thresh = loss_th

        def _bootstrap_xentropy_single(input, target, K, thresh, weight=None, size_average=True):

            n, c, h, w = input.size()
            input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
            target = target.view(-1)
            loss = F.cross_entropy(
                input, target, weight=weight, reduce=False, size_average=False, ignore_index=250
            )


            sorted_loss, _ = torch.sort(loss, descending=True)

            if sorted_loss[K] > thresh:
                loss = sorted_loss[sorted_loss > thresh]
            else:
                loss = sorted_loss[:K]

            reduced_topk_loss = torch.mean(loss)

            return reduced_topk_loss

        # end

        loss = 0.0
        # Bootstrap from each image not entire batch
        for i in range(batch_size):
            loss += _bootstrap_xentropy_single(
                input=torch.unsqueeze(input[i], 0),
                target=torch.unsqueeze(self.GT_image_final[i], 0),
                K=min_K,
                thresh=thresh,
                weight=weight,
                size_average=size_average,
            )
            # print(loss)
        return loss / float(batch_size)


class cen_validation:
    def __init__(self,dir_GT , index, device, FACTOR_ori_to_labelmap_h =0.5, FACTOR_ori_to_labelmap_w = 0.5):

        full_fname_triplet_json = dir_GT + "raw/rs" + f"{index + 7000:05d}" + ".txt"
        self.list_triplet_json = json.load(open(full_fname_triplet_json, 'r'))
        self.device = device
        num_classes = 1

        h_labelmap = 540
        w_labelmap = 960

        ###
        labelmap_centerline_out = np.zeros((num_classes, 540, 960), dtype=np.float32)
        labelmap_centerline_out0 = labelmap_centerline_out[0]


        ###================================================================================================
        ###
        ###================================================================================================
        mode_vote = 1

        for list_this_set in self.list_triplet_json:
            # list_this_set: {list: N}, consists of N triplets.

            for triplet_this in list_this_set:
                # triplet_this: {list:4} (x_L, x_C, x_R, y)

                ###---------------------------------------------------------------------------------------
                ### get this triplet
                ###---------------------------------------------------------------------------------------
                x_L_fl = triplet_this[0] * FACTOR_ori_to_labelmap_w
                x_C_fl = triplet_this[1] * FACTOR_ori_to_labelmap_w
                x_R_fl = triplet_this[2] * FACTOR_ori_to_labelmap_w
                y = int(round(triplet_this[3] * FACTOR_ori_to_labelmap_h))

                if (y < 0) or (y >= h_labelmap):
                    continue
                # end

                ###
                x_L_int = round(x_L_fl)
                x_R_int = round(x_R_fl)

                ###
                if (x_L_int < 0) or (x_R_int < 0) or (x_L_int >= w_labelmap) or (x_R_int >= w_labelmap):
                    continue
                # end

                ###---------------------------------------------------------------------------------------
                ### vote for centerline
                ###---------------------------------------------------------------------------------------
                if mode_vote == 0:
                    labelmap_centerline_out0[y, x_L_int:(x_R_int + 1)] = 1.0

                elif mode_vote == 1:
                    for x_this in range(x_L_int, x_R_int + 1):
                        if (x_this < 0) or (x_this >= w_labelmap):
                            continue
                        dist_this_to_L = abs(x_this - x_L_fl)
                        dist_this_to_R = abs(x_R_fl - x_this)
                        dist_average = min(dist_this_to_L, dist_this_to_R)

                        val_heat_old = labelmap_centerline_out0[y, x_this]
                        labelmap_centerline_out0[y, x_this] = max(dist_average, val_heat_old)

        output_labelmap_centerline = torch.from_numpy(labelmap_centerline_out).float()
        output_labelmap_centerline = output_labelmap_centerline.to(self.device)

        self.output_labelmap_centerline = output_labelmap_centerline

    def calculate_loss(self,input):

        def L1_loss(x_est, x_gt, b_sigmoid=False):
            # ////////////////////////////////////////////////////////////////////////////////////////////////////////////
            # x_est: (bs, 1, h, w), here 1 is the number of class
            # x_gt: (bs, 1, h, w)
            # ////////////////////////////////////////////////////////////////////////////////////////////////////////////

            # L1 loss
            # https://pytorch.org/docs/master/generated/torch.nn.L1Loss.html

            if b_sigmoid is True:
                x_est = torch.clamp(torch.sigmoid(x_est), min=1e-4, max=1 - 1e-4)
            # end

            loss_a = nn.L1Loss()
            loss_b = loss_a(x_est, x_gt)
            # print(loss_b)
            return loss_b

        print(L1_loss(input, self.output_labelmap_centerline))
        return L1_loss(input, self.output_labelmap_centerline)


    def calculate_loss_regional(self, input, seg_gt):

        loss_sum = 0
        num_pixels = 0
        for image_counter,image in enumerate(input):
            for row_counter,row in enumerate(image[0]):
                for column_counter,pixel in enumerate(row):
                    if seg_gt[image_counter, row_counter, column_counter] < 2:
                        loss_sum += abs(pixel.item() - self.output_labelmap_centerline[image_counter, row_counter, column_counter].item())
                        num_pixels += 1
                    else:
                        continue

        loss = loss_sum/num_pixels

        # print(loss)
        return loss

    def calculate_loss_at_peaks(self, input):
        loss = 0
        num_pixels = 0
        for image_index in range(input.shape[0]):
            vertical_dim = input.shape[2]

            for height in range(vertical_dim):
                dist_min = (235.0/270.0) * height + (-220.0)
                dist_min = max(15, dist_min)
                dist_min = dist_min * (0.1)
                row_centerness_estimated = input[image_index, 0, height]
                x_peaks_estimated, _ = find_peaks(row_centerness_estimated.cpu().data.numpy(), height=dist_min, distance=10)

                row_centerness_gt = self.output_labelmap_centerline[image_index, height]
                x_peaks_gt, _ = find_peaks(row_centerness_gt.cpu().data.numpy(), height=dist_min, distance=10)

                if len(x_peaks_gt) == len(x_peaks_estimated):
                    for index in range(len(x_peaks_gt)):
                        loss += abs(self.output_labelmap_centerline[image_index, height, x_peaks_gt[index]] - input[image_index, 0, height, x_peaks_estimated[index]]) + abs(x_peaks_gt[index] - x_peaks_estimated[index])
                        num_pixels += 1


                else:
                    for x_this_estimated in x_peaks_estimated:
                        loss += (abs(self.output_labelmap_centerline[image_index, height, x_this_estimated] - input[image_index, 0, height, x_this_estimated]))
                        num_pixels += 1

        if num_pixels>0:
            loss_avg = loss / num_pixels
        else:
            loss_avg = torch.tensor(0)


        # print(loss_avg)
        # print(num_pixels)
        return loss_avg


class eval_seg_object:

    def __init__(self, gt_image, estimated_image, image_height = 540, image_width = 960):
        self.gt_image = gt_image
        self.estimated_image = estimated_image
        self.image_height = image_height
        self.image_width = image_width

    def calculate_IoU(self, class_this):
        intersection_counter = 0
        union_counter = 0
        for h in range(self.image_height):
            for w in range(self.image_width):
                if self.gt_image[h,w] == class_this or self.estimated_image[h,w] == class_this:
                    union_counter += 1

                if self.gt_image[h,w] == class_this and self.estimated_image[h,w] == class_this:
                    intersection_counter += 1

        if union_counter > 0:
            IoU = intersection_counter/union_counter

        else:
            IoU = 0

        return IoU



# The below performance measure is not in topology-guided paper
class eval_object_all_pixel_level:
    def __init__(self, gt_data, detected_data, image_height = 540, image_width = 960):
        self.gt_data = gt_data
        self.detected_data = detected_data
        self.image_height = image_height
        self.image_width = image_width
        self.tot_num_gt_paths = len(gt_data)
        self.tot_num_detected_paths = len(detected_data)


    def create_binary_image(self, y_min):
        binary_img = np.zeros((self.image_height, self.image_width))

        for path in self.detected_data:
            detected_3d_left  = path["polynomial"]["xy_left"]
            detected_3d_right = path["polynomial"]["xy_right"]

            for idx_pnt in range(detected_3d_left.shape[0]):
                x3d_detected = detected_3d_left[idx_pnt][0]
                y3d_detected = detected_3d_left[idx_pnt][1]

                ximg_detected_int = int(round(x3d_detected))
                yimg_detected_int = int(round(y3d_detected))

                if (ximg_detected_int < 0) or (ximg_detected_int >= self.image_width) or (yimg_detected_int < 0) or (yimg_detected_int >= self.image_height) or (yimg_detected_int < 270) or (yimg_detected_int < y_min):
                    continue

                binary_img[yimg_detected_int, ximg_detected_int] = 1


            for idx_pnt in range(detected_3d_right.shape[0]):
                x3d_detected = detected_3d_right[idx_pnt][0]
                y3d_detected = detected_3d_right[idx_pnt][1]

                ximg_detected_int = int(round(x3d_detected))
                yimg_detected_int = int(round(y3d_detected))

                if (ximg_detected_int < 0) or (ximg_detected_int >= self.image_width) or (yimg_detected_int < 0) or (yimg_detected_int >= self.image_height) or (yimg_detected_int < 270) or (yimg_detected_int < y_min):
                    continue

                binary_img[yimg_detected_int, ximg_detected_int] = 1


        return binary_img


    def find_matches(self, search_area_offset, y_min):

        if self.tot_num_detected_paths == 0:
            return 0,0
        TP = 0
        FN = 0
        FP = 0

        bin_image = self.create_binary_image(y_min)
        for cnt_gt in range(self.tot_num_gt_paths):
            gt_LR_rail_this = list(self.gt_data.values())[cnt_gt]
            for LR_index in range(2):
                for cnt_point in range(len(gt_LR_rail_this)):
                    ximg_gt = gt_LR_rail_this[cnt_point][LR_index]

                    if (ximg_gt < 0) or (ximg_gt > self.image_width):
                        continue

                    flag_match = 0
                    for i in range(cnt_point - search_area_offset, cnt_point + search_area_offset + 1):
                        for j in range(ximg_gt - search_area_offset, ximg_gt + search_area_offset + 1):
                            if (i < 270) or (i >= self.image_height) or (j < 0) or (j >= self.image_width):
                                continue

                            if bin_image[i, j] != 0:
                                bin_image[i, j] = 0.5
                                flag_match = 1

                    if flag_match == 1:
                        TP = TP + 1
                    else:
                        FN = FN + 1

        FP = FP + np.count_nonzero(bin_image == 1)

        if TP > 0:
            prec   = (TP / (TP + FP))
            recall = (TP / (TP + FN))
        else:
            prec   = 0
            recall = 0

        return prec, recall








