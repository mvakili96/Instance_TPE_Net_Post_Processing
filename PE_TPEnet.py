# 2020/8/11
# Jungwon Kang


import torch
import numpy as np
import time
import cv2
import copy
from sklearn.cluster import MeanShift


from helpers.models import get_model
from helpers.utils  import my_utils_img
from helpers.utils  import my_utils_net
from helpers.utils  import my_utils_3D
from helpers.utils  import my_utils_RPG
from helpers.utils  import my_utils_vis
from scipy.signal import find_peaks

#torch.backends.cudnn.benchmark = False

########################################################################################################################
###
########################################################################################################################
class PathExtraction_TPEnet:

    ###=========================================================================================================
    ### __init__()
    ###=========================================================================================================
    def __init__(self, dim_ins_vectors, dir_weight_file):
        """
        initialize

        :param args
        """

        self.dim_ins_vectors = dim_ins_vectors



        ###---------------------------------------------------------------------------------------------
        ### arrange args
        ###---------------------------------------------------------------------------------------------
        dict_args_net = {"file_weight":dir_weight_file}


        ###---------------------------------------------------------------------------------------------
        ### init obj
        ###---------------------------------------------------------------------------------------------
        self.m_obj_utils_net = my_utils_net.MyUtils_Net(dict_args_net)

        ###---------------------------------------------------------------------------------------------
        ### init model
        ###---------------------------------------------------------------------------------------------
        self.m_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ###
        self.m_model = get_model({"arch": "TPEnet_a"}, dim_ins_vectors)

        ###
        self.m_obj_utils_net.load_weights_to_model(self.m_model)

        self.m_model.eval()
        self.m_model.to(self.m_device)


            # completed to set
            #       m_device
            #       m_model

        return
    #end


    ###=========================================================================================================
    ### process()
    ###=========================================================================================================
    def process(self, img_raw_rsz_uint8,binary_seg_mask_json):
        """

        :param img_raw_rsz_uint8: input image (3ch: rgb)
        :return:
        """



        ###------------------------------------------------------------------------------------------------
        ### 1. input image
        ###------------------------------------------------------------------------------------------------
        img_raw_for_vis = copy.deepcopy((img_raw_rsz_uint8))

        img_raw_rsz_fl_n = self.convert_img_ori_to_img_data(img_raw_rsz_uint8)
            # <input> img_raw_rsz_uint8: ndarray(H,W,C), 0 ~ 255
            # <output> img_raw_rsz_fl_n: ndarray(C,H,W), -X.0 ~ X.0

        img_raw = np.expand_dims(img_raw_rsz_fl_n, 0)
        img_raw = torch.from_numpy(img_raw).float()
        images  = img_raw.to(self.m_device)
        ###------------------------------------------------------------------------------------------------
        ### 2. do feed-forwarding to get centerness/left-right/segmentation output
        ###------------------------------------------------------------------------------------------------

        ### sample time
        time_a = time.time()
        output_instance_segmentation = self.m_model(images)

        ### sample time
        time_b   = time.time()
        dtime_ab = time_b - time_a


        ###------------------------------------------------------------------------------------------------
        ### 3. get instance segmentation final outcome
        ###------------------------------------------------------------------------------------------------
        output_instance_segmentation = output_instance_segmentation[0].transpose(0, 1).transpose(1, 2).contiguous()
        output_instance_segmentation = output_instance_segmentation.detach().cpu().numpy()

        binary_seg_mask_json = np.array(binary_seg_mask_json)
        vectors = output_instance_segmentation[binary_seg_mask_json[:,0],binary_seg_mask_json[:,1]]


        vectors = np.array(vectors)
        clustering = MeanShift(bandwidth=1).fit(vectors)
        labels = clustering.labels_


        image_outcome_instances = self.Create_ins_seg_image(labels,binary_seg_mask_json)

        # DEBUG (show binary image)
        if 0:
            bin_image = np.zeros((540,960))
            for XY in binary_seg_mask_json:
                bin_image[XY[0],XY[1]] = 1

            cv2.imshow("BIN_IMAGE",bin_image)
            cv2.imshow("RAW_IMAGE",img_raw_for_vis)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


        # DEBUG (calculate variance term)
        if 0:
            # SORT GROUND TRUTH DATA
            vectors = [[] for i in range(20)]
            instance_counter = 0
            for point_cnt,XY in enumerate(binary_seg_mask_json):
                vectors[instance_counter].append(output_instance_segmentation[XY[0], XY[1]])
                if point_cnt > 0 and abs(binary_seg_mask_json[point_cnt,0] - binary_seg_mask_json[point_cnt-1,0]) > 5:
                    instance_counter += 1

            for instance_cnt, instance in enumerate(vectors):
                if len(instance) == 0:
                    ind = instance_cnt
                    print(ind)
                    break

            vectors = vectors[0:ind]

            var_loss = []
            for instance_cnt, instance in enumerate(vectors):
                var_margin = 0.5
                vectors_this = torch.tensor(instance)
                instance_mean_vector = torch.mean(vectors_this, dim = 0).reshape(1,self.dim_ins_vectors)
                var_loss_this        = self.Variance_term(vectors_this,instance_mean_vector,var_margin)
                var_loss.append(var_loss_this)
                # print(instance_cnt)
                # print(var_loss_this)
            var_loss = np.array(var_loss)
            print(np.mean(var_loss))

        # DEBUG (calculate distance term)
        if 0:
            vectors1 = torch.tensor(vectors[0:1500])
            vectors2 = torch.tensor(vectors[6000:7522])
            instance_mean_vector1 = torch.mean(vectors1, dim = 0).reshape(1,self.dim_ins_vectors)
            instance_mean_vector2 = torch.mean(vectors2, dim=0).reshape(1, self.dim_ins_vectors)
            dis_loss = torch.cdist(instance_mean_vector1, instance_mean_vector2, p=2)
            print(dis_loss)



        ### sample time
        time_c   = time.time()
        dtime_bc = time_c - time_b
        # print(dtime_bc)


        ###------------------------------------------------------------------------------------------------
        ### 8. output
        ###------------------------------------------------------------------------------------------------
        dict_res_time = {"dtime_ab": dtime_ab,
                         "dtime_bc": dtime_bc
                         }


        return labels


    #end

    def convert_img_ori_to_img_data(self, img_ori_uint8,
                                    rgb_mean=np.array([128.0, 128.0, 128.0]) / 255.0,
                                    rgb_std=np.array([1.0, 1.0, 1.0])):
        """
        convert uint8-type image into floating-type image (with normalization)

        :param img_ori_uint8: ndarray(H,W,C), 0 ~ 255
        :param rgb_mean
        :param rgb_std
        :return: img_data_fl_n_final: ndarray(C,H,W), -X.0 ~ X.0
        """


        #/////////////////////////////////////////////////////////////////////////////////////////////////////////
        # convert img_ori to img_data
        # <input>
        #   img_ori_uint8:      ndarray(H,W,C), 0 ~ 255
        # <output>
        #   img_raw_fl_n_final: ndarray(C,H,W), -X.0 ~ X.0
        #
        # we are doing the following things:
        #   (1) normalize so that 0~255 -> 0.0~1.0
        #   (2) apply rgb_mean
        #   (3) apply rgb_std
        #   (4) convert HWC -> CHW
        #   (5) make sure it is float32 type
        #/////////////////////////////////////////////////////////////////////////////////////////////////////////
        if img_ori_uint8.shape[0] != 540:
            img_ori_uint8 = cv2.resize(img_ori_uint8, (960, 540))

        ###================================================================================================
        ### (1) normalize so that 0~255 -> 0.0~1.0
        ###================================================================================================
        img_ori_fl = img_ori_uint8.astype(np.float32) / 255.0


        ###================================================================================================
        ### (2) apply rgb_mean
        ###================================================================================================
        img_ori_fl_n = img_ori_fl - rgb_mean
            # completed to set
            #       img_ori_fl_n: -X.0 ~ X.0, ndarray(H,W,C)


        ###================================================================================================
        ### (3) apply rgb_std
        ###================================================================================================
        img_ori_fl_n = img_ori_fl_n / rgb_std


        ###================================================================================================
        ### (4) convert HWC -> CHW
        ###================================================================================================
        img_ori_fl_n = img_ori_fl_n.transpose(2, 0, 1)
            # H(0),W(1),C(2) -> C(2),H(0),W(1)
            # completed to set
            #       img_ori_fl_n: -1.0 ~ 1.0, ndarray(C,H,W)


        ###================================================================================================
        ### (5) make sure it is float32 type
        ###================================================================================================
        img_data_fl_n_final = img_ori_fl_n.astype(np.float32)


        return img_data_fl_n_final
            # ndarray(C,H,W), -X.0 ~ X.0
    #end

    def Variance_term(self,instance_vectors,mean_vector,margin):
        all_distances        = torch.cdist(instance_vectors, mean_vector, p=2)
        all_distances_margin = all_distances[all_distances > margin]

        if all_distances_margin.shape[0] != 0:
            tot_num_pixels = instance_vectors.shape[0]
            sum_distances  = torch.sum(torch.pow(all_distances_margin,2))
            mean_distance  = sum_distances/tot_num_pixels
        else:
            mean_distance = 0

        return mean_distance

    def Create_ins_seg_image(self, labels, binary_mask_json):

        rgb_class01 = [232, 35, 244]
        rgb_class02 = [70, 70, 70]
        rgb_class03 = [128, 0, 192]
        rgb_class04 = [153, 153, 190]
        rgb_class05 = [153, 153, 153]
        rgb_class06 = [30, 170, 250]
        rgb_class07 = [0, 220, 220]
        rgb_class08 = [35, 142, 107]
        rgb_class09 = [152, 251, 152]
        rgb_class10 = [180, 130, 70]
        rgb_class11 = [60, 20, 220]
        rgb_class12 = [140, 150, 230]
        rgb_class13 = [142, 0, 0]
        rgb_class14 = [70, 0, 0]
        rgb_class15 = [40, 40, 90]
        rgb_class16 = [100, 80, 0]
        rgb_class17 = [254, 254, 0]
        rgb_class18 = [63, 68, 0]
        rgb_class19 = [252, 68, 0]
        rgb_class20 = [120, 51, 151]

        ###
        rgb_labels = np.array(
            [
                rgb_class01,
                rgb_class02,
                rgb_class03,
                rgb_class04,
                rgb_class05,
                rgb_class06,
                rgb_class07,
                rgb_class08,
                rgb_class09,
                rgb_class10,
                rgb_class11,
                rgb_class12,
                rgb_class13,
                rgb_class14,
                rgb_class15,
                rgb_class16,
                rgb_class17,
                rgb_class18,
                rgb_class19,
                rgb_class20,
            ]
        )

        image_outcome = np.zeros((540,960,3),'uint8')
        for num,item in enumerate(labels):
            image_outcome[binary_mask_json[num][0],binary_mask_json[num][1]] = rgb_labels[int(item)]

        cv2.imshow("Image_Instance_Seg_Outcome",image_outcome)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return image_outcome

    ###=========================================================================================================
    ### convert_one_point_from_world_to_img
    ###=========================================================================================================
    def convert_one_point_from_world_to_img(self, x_in, y_in, z_in = 1.0):
        x_out, y_out = self.m_obj_utils_3D.convert_pnt_world_to_pnt_img_ori(
            np.array([[x_in], [y_in], [z_in]]))
        return [x_out, y_out]

    def show_centerness_on_raw_image(self, raw_image, centerness_image):
        gray_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)

        gray_image_3channel = np.zeros_like(raw_image)  # img_ori_gray3: 3-ch gray img
        gray_image_3channel[:, :, 0] = gray_image
        gray_image_3channel[:, :, 1] = gray_image
        gray_image_3channel[:, :, 2] = gray_image

        # print(gray_image.shape)
        # base_added_red_channel_val = 100
        coeff_red_channel_val = 500
        for i,row in enumerate(centerness_image):
            for j,pixel in enumerate(row):
                added_red_channel_val = int( ((pixel/255)**0.6)*coeff_red_channel_val )
                if gray_image_3channel[i, j, 0] >= 255-added_red_channel_val:
                    gray_image_3channel[i, j, :] = [255, 255-added_red_channel_val, 255-added_red_channel_val]
                else:
                    gray_image_3channel[i, j, :] = gray_image_3channel[i, j, :] + [added_red_channel_val, 0, 0]

        return gray_image_3channel

    def show_final_path_on_ori_noGTdata(self, raw_image, list_paths_final):
        points_ref = []
        for detected_path_index in range(len(list_paths_final)):
            detected_3d_left = list_paths_final[detected_path_index]["polynomial"]["xyz_left_3d"]
            detected_3d_right = list_paths_final[detected_path_index]["polynomial"]["xyz_right_3d"]
            for idx_pnt in range(detected_3d_left.shape[0]):  # Loop for detected points (left) to create binary image
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
            # all_res_images.append(img_cp)



        return raw_image
    ###=========================================================================================================
    ### visualize final path on ori-img
    ###=========================================================================================================
    def show_final_path_on_ori_v0(self, list_paths_final, img_bg, res_for_show=1):
        # ---------------------------------------------------------------------------------------------
        # dict_path_final = {"extracted": dict_path_this,
        #                    "polynomial": dict_path_poly_this}
        #
        #    "extracted": dict_path_this = {"xy_cen_img": [],
        #                                   "xy_left_img": [],
        #                                   "xy_right_img": [],
        #                                   "xyz_cen_3d": [],
        #                                   "xyz_left_3d": [],
        #                                   "xyz_right_3d": [],
        #                                   "xy_switch_img"  : [],  # switch (img)
        #                                   "xyz_switch_3d"  : [],  # switch (3d)
        #                                   "id_node_switch" : []   # switch (id_node)
        #
        #    "polynomial": dict_path_poly_this = {"xyz_cen_3d": arr_xyz_sample_ori,
        #                                         "xyz_left_3d": [],
        #                                         "xyz_right_3d": []}
        # ---------------------------------------------------------------------------------------------
        # img_bg: img_raw_rsz_uint8
        # ---------------------------------------------------------------------------------------------
        #   res_for_show=0: show "extracted"
        #   res_for_show=1: show "polynomial"
        # ---------------------------------------------------------------------------------------------


        ###
        assert (self.m_obj_utils_3D is not None)
        assert (img_bg is not None)


        ###------------------------------------------------------------------------------------------------
        ### create img for visualization
        ###------------------------------------------------------------------------------------------------
        img_vis_ori_rgb = copy.deepcopy(img_bg)
        img_vis_ori_gray1 = cv2.cvtColor(img_vis_ori_rgb, cv2.COLOR_BGR2GRAY)

        img_vis_ori_gray3 = np.zeros_like(img_vis_ori_rgb)  # img_ori_gray3: 3-ch gray img
        img_vis_ori_gray3[:, :, 0] = img_vis_ori_gray1
        img_vis_ori_gray3[:, :, 1] = img_vis_ori_gray1
        img_vis_ori_gray3[:, :, 2] = img_vis_ori_gray1


        ###------------------------------------------------------------------------------------------------
        ###
        ###------------------------------------------------------------------------------------------------
        h_img = img_bg.shape[0]
        w_img = img_bg.shape[1]


        ###------------------------------------------------------------------------------------------------
        ### process each path
        ###------------------------------------------------------------------------------------------------

        for idx_path in range(len(list_paths_final)):

            if idx_path == 10:
                pass
            else:
                continue

            # ---------------------------------------------------------------------------------------
            #
            # ---------------------------------------------------------------------------------------
            dict_extracted = list_paths_final[idx_path]["extracted"]

            ###
            arr_xy_cen_img_extracted = dict_extracted["xy_cen_img"]
            arr_xy_left_img_extracted = dict_extracted["xy_left_img"]
            arr_xy_right_img_extracted = dict_extracted["xy_right_img"]

            #################################################
            arr_xy_switch_img = dict_extracted["xy_switch_img_edge_end"]
            #################################################

            arr_xyz_switch_3d_extracted = dict_extracted["xyz_switch_3d"]


            ###
            arr_xyz_cen_3d_temp = dict_extracted["xyz_cen_3d"]
            arr_y_cen_3d_temp = arr_xyz_cen_3d_temp[:, 1]

            max_y_cen_3d_extracted = max(arr_y_cen_3d_temp)


            # ---------------------------------------------------------------------------------------
            #
            # ---------------------------------------------------------------------------------------
            dict_polynomial = list_paths_final[idx_path]["polynomial"]

            ###
            arr_xyx_cen_3d_polynomial = dict_polynomial["xyz_cen_3d"]
            arr_xyx_left_3d_polynomial = dict_polynomial["xyz_left_3d"]
            arr_xyx_right_3d_polynomial = dict_polynomial["xyz_right_3d"]


            # ---------------------------------------------------------------------------------------
            #
            # ---------------------------------------------------------------------------------------
            totnum_pnts = 0

            if res_for_show == 0:  # show extracted
                totnum_pnts = arr_xy_cen_img_extracted.shape[0]
            else: # show polynomial
                totnum_pnts = arr_xyx_cen_3d_polynomial.shape[0]
            #end

            totnum_pnts = arr_xy_cen_img_extracted.shape[0]
            # ---------------------------------------------------------------------------------------
            #
            # ---------------------------------------------------------------------------------------
            for idx_pnt in range(totnum_pnts):

                ###------------------------------------------------------------------------------
                ###
                ###------------------------------------------------------------------------------
                x_cen_img = None
                y_cen_img = None

                x_left_img = None
                y_left_img = None

                x_right_img = None
                y_right_img = None

                x_cen_3d_this = None
                y_cen_3d_this = None


                ###
                if res_for_show == 0:   # show extracted
                    x_cen_img = arr_xy_cen_img_extracted[idx_pnt, 0]
                    y_cen_img = arr_xy_cen_img_extracted[idx_pnt, 1]

                    x_left_img = arr_xy_left_img_extracted[idx_pnt, 0]
                    y_left_img = arr_xy_left_img_extracted[idx_pnt, 1]

                    x_right_img = arr_xy_right_img_extracted[idx_pnt, 0]
                    y_right_img = arr_xy_right_img_extracted[idx_pnt, 1]
                else:
                    ###
                    x_cen_3d_this = arr_xy_cen_img_extracted[idx_pnt, 0]
                    y_cen_3d_this = arr_xy_cen_img_extracted[idx_pnt, 1]
                    #z_cen_3d_this = arr_xyx_cen_3d_polynomial[idx_pnt, 2]

                    x_left_3d_this = arr_xy_left_img_extracted[idx_pnt, 0]
                    y_left_3d_this = arr_xy_left_img_extracted[idx_pnt, 1]
                    #z_left_3d_this = arr_xyx_left_3d_polynomial[idx_pnt, 2]

                    x_right_3d_this = arr_xy_right_img_extracted[idx_pnt, 0]
                    y_right_3d_this = arr_xy_right_img_extracted[idx_pnt, 1]
                    #z_right_3d_this = arr_xyx_right_3d_polynomial[idx_pnt, 2]


                    ###
                    x_cen_img = x_cen_3d_this
                    y_cen_img = y_cen_3d_this

                    x_left_img = x_left_3d_this
                    y_left_img = y_left_3d_this

                    x_right_img = x_right_3d_this
                    y_right_img = y_right_3d_this

                    # x_cen_img, y_cen_img = self.m_obj_utils_3D.convert_pnt_world_to_pnt_img_ori(np.array([[x_cen_3d_this], [y_cen_3d_this], [1.0]]))
                    # x_left_img, y_left_img = self.m_obj_utils_3D.convert_pnt_world_to_pnt_img_ori(np.array([[x_left_3d_this], [y_left_3d_this], [1.0]]))
                    # x_right_img, y_right_img = self.m_obj_utils_3D.convert_pnt_world_to_pnt_img_ori(np.array([[x_right_3d_this], [y_right_3d_this], [1.0]]))

                    # x_cen_img = arr_xy_cen_img_extracted[idx_pnt, 0]
                    # y_cen_img = arr_xy_cen_img_extracted[idx_pnt, 1]
                    #
                    # x_left_img = arr_xy_left_img_extracted[idx_pnt, 0]
                    # y_left_img = arr_xy_left_img_extracted[idx_pnt, 1]
                    #
                    # x_right_img = arr_xy_right_img_extracted[idx_pnt, 0]
                    # y_right_img = arr_xy_right_img_extracted[idx_pnt, 1]
                #end


                ###
                # if y_cen_3d_this >= max_y_cen_3d_extracted:
                #     continue
                # # end


                ###
                x_cen_img_int = int(round(x_cen_img))
                y_cen_img_int = int(round(y_cen_img))

                x_left_img_int = int(round(x_left_img))
                y_left_img_int = int(round(y_left_img))

                x_right_img_int = int(round(x_right_img))
                y_right_img_int = int(round(y_right_img))



                ###------------------------------------------------------------------------------
                ### DRAW
                ###------------------------------------------------------------------------------
                val_b_cen = 128
                val_g_cen = 0
                val_r_cen = 0

                val_b_left = 20
                val_g_left = 100
                val_r_left = 250

                val_b_right = 250
                val_g_right = 250
                val_r_right = 0


                # for idx_sw in range(arr_xyz_switch_3d_extracted.shape[0]):
                #     y_switch_3d_this = arr_xyz_switch_3d_extracted[idx_sw, 1]
                #
                #     dy_abs = abs(y_cen_3d_this - y_switch_3d_this)
                #
                #     if dy_abs < 12.0:
                #         val_b_cen = 0
                #         val_g_cen = 0
                #         val_r_cen = 255
                #     #end
                # #end


                ### draw pnt (cen)
                if (0 <= x_cen_img_int) and (x_cen_img_int < w_img) and (0 <= y_cen_img_int) and (y_cen_img_int < h_img):
                    if (idx_path < 10):# or (idx_path > 0 and y_cen_3d_this > y_switch_3d_this - 25 ):
                        cv2.circle(img_vis_ori_gray3, center=(x_cen_img_int, y_cen_img_int), radius=2,
                               color=(val_b_cen, val_g_cen, val_r_cen), thickness=-1)
                # end


                ## draw pnt (left)
                if (0 <= x_left_img_int) and (x_left_img_int < w_img) and (0 <= y_left_img_int) and (y_left_img_int < h_img):
                    if (idx_path < 10):# or (idx_path > 0 and y_cen_3d_this > y_switch_3d_this - 25):
                        cv2.circle(img_vis_ori_gray3, center=(x_left_img_int, y_left_img_int), radius=1,
                               color=(val_b_left, val_g_left, val_r_left), thickness=-1)
                # end


                ### draw pnt (right)
                if (0 <= x_right_img_int) and (x_right_img_int < w_img) and (0 <= y_right_img_int) and (y_right_img_int < h_img):
                    if (idx_path < 10):# or (idx_path > 0 and y_cen_3d_this > y_switch_3d_this - 25):
                        cv2.circle(img_vis_ori_gray3, center=(x_right_img_int, y_right_img_int), radius=1,
                               color=(val_b_right, val_g_right, val_r_right), thickness=-1)
                # end
            # end
        # end



        ###------------------------------------------------------------------------------------------------
        ### show
        # ###------------------------------------------------------------------------------------------------
        # cv2.imshow('final_path_on_ori', img_vis_ori_gray3)
        # cv2.waitKey(0)

        ### save (temp)
        if 0:
            fname_output_temp = "./res_imgs" + "final_path_ori_" + str(self.m_temp_idx_res_img_on_ori) + '.jpg'
            cv2.imwrite(fname_output_temp, img_vis_ori_gray3)
            self.m_temp_idx_res_img_on_ori += 1
        #end

        return img_vis_ori_gray3

    #end


    ###=========================================================================================================
    ### visualize final path on ori-img
    ###=========================================================================================================
    def show_final_path_on_ori_v1(self, list_paths_final, img_bg_in, res_for_show=1):
        # ---------------------------------------------------------------------------------------------------
        # list_paths_out:
        #   list_paths_out[i]: ith path, is {dict:3}
        #       'extracted': having the following
        #            -> set in def _convert_to_paths_as_vertices_v2(..):
        #            dict_path_this = {"id_edge": [],
        #                              "xy_cen_img": [],
        #                              "xy_left_img": [],
        #                              "xy_right_img": [],
        #                              "xyz_cen_3d": [],
        #                              "xyz_left_3d": [],
        #                              "xyz_right_3d": [],
        #                              ###
        #                              "id_node_switch": [],  # switch (id_node)
        #                              "xy_switch_img": [],  # switch (img)
        #                              "xyz_switch_3d": [],  # switch (3d)
        #                              ###
        #                              "xy_switch_img_edge_start": [],
        #                              "xyz_switch_3d_edge_start": [],
        #                              "xy_switch_img_edge_end": [],  # equal to "xy_switch_img"
        #                              "xyz_switch_3d_edge_end": []  # equal to "xyz_switch_3d"
        #                              }
        #
        #       'polynomial': having the following dict
        #           -> set in def _get_paths_by_polynomial_fitting(..):
        #            dict_path_poly_this = {"xyz_cen_3d": sample_arr_xyz_cen_ori,
        #                                   "xyz_left_3d": sample_arr_xyz_left_ori,
        #                                   "xyz_right_3d": sample_arr_xyz_right_ori,
        #                                   "coeff_poly_cen_3d_new": coeff_poly_cen,
        #                                   "coeff_poly_left_3d_new": coeff_poly_left,
        #                                   "coeff_poly_right_3d_new": coeff_poly_right}
        #
        #       'type_path': list_type_paths[idx_path] -> EGO or NON-EGO
        #
        # ---------------------------------------------------------------------------------------------
        # img_bg: img_raw_rsz_uint8
        # ---------------------------------------------------------------------------------------------
        #   res_for_show=0: show "extracted"
        #   res_for_show=1: show "polynomial"
        # ---------------------------------------------------------------------------------------------


        ###
        assert (self.m_obj_utils_3D is not None)
        assert (img_bg_in is not None)


        ###------------------------------------------------------------------------------------------------
        ### create img for visualization
        ###------------------------------------------------------------------------------------------------
        img_vis_res = copy.deepcopy(img_bg_in)

        h_img = img_vis_res.shape[0]
        w_img = img_vis_res.shape[1]


        ###------------------------------------------------------------------------------------------------
        ### set path sequence for drawing (for first drawing non-ego paths, then drawing ego path)
        ###------------------------------------------------------------------------------------------------
        list_idx_path_for_drawing = []

        ### for non-ego path
        for idx_path in range(len(list_paths_final)):
            type_path_this = list_paths_final[idx_path]["type_path"]

            if type_path_this is my_utils_RPG.TYPE_path.NON_EGO:
                list_idx_path_for_drawing.append(idx_path)
            #end
        #end

        ### for ego path
        for idx_path in range(len(list_paths_final)):
            type_path_this = list_paths_final[idx_path]["type_path"]

            if type_path_this is my_utils_RPG.TYPE_path.EGO:
                list_idx_path_for_drawing.append(idx_path)
            #end
        #end
            # completed to set
            #   list_idx_path_for_drawing




        ###------------------------------------------------------------------------------------------------
        ### step 0: draw (filled) path-region
        ###------------------------------------------------------------------------------------------------
        for idx_draw in range(len(list_paths_final)):
            idx_path = list_idx_path_for_drawing[idx_draw]
            type_path_this = list_paths_final[idx_path]["type_path"]


            # ---------------------------------------------------------------------------------------
            # get info ("extracted") for this path
            # ---------------------------------------------------------------------------------------
            dict_extracted = list_paths_final[idx_path]["extracted"]

            ###
            arr_xy_cen_img_extracted      = dict_extracted["xy_cen_img"]
            arr_xy_left_img_extracted     = dict_extracted["xy_left_img"]
            arr_xy_right_img_extracted    = dict_extracted["xy_right_img"]

            arr_xyz_switch_3d_extracted   = dict_extracted["xyz_switch_3d"]
            info_xyz_switch_3d_edge_start = dict_extracted["xyz_switch_3d_edge_start"]
            info_xyz_switch_3d_edge_end   = dict_extracted["xyz_switch_3d_edge_end"]


            ### just for setting max y for visualization
            arr_xyz_cen_3d_temp           = dict_extracted["xyz_cen_3d"]
            arr_y_cen_3d_temp             = arr_xyz_cen_3d_temp[:, 1]
            max_y_cen_3d_extracted        = max(arr_y_cen_3d_temp)
                # max y for visualization


            # ---------------------------------------------------------------------------------------
            # get info ("polynomial-fitted") for this path
            # ---------------------------------------------------------------------------------------
            dict_polynomial = list_paths_final[idx_path]["polynomial"]

            ###
            arr_xyx_cen_3d_polynomial = dict_polynomial["xyz_cen_3d"]
            arr_xyx_left_3d_polynomial = dict_polynomial["xyz_left_3d"]
            arr_xyx_right_3d_polynomial = dict_polynomial["xyz_right_3d"]


            # ---------------------------------------------------------------------------------------
            #
            # ---------------------------------------------------------------------------------------
            totnum_pnts = 0

            if res_for_show == 0:  # show extracted
                totnum_pnts = arr_xy_cen_img_extracted.shape[0]
            else: # show polynomial
                totnum_pnts = arr_xyx_cen_3d_polynomial.shape[0]
            #end


            # ---------------------------------------------------------------------------------------
            # draw (filled) path-region
            # ---------------------------------------------------------------------------------------
            for idx_pnt in range(totnum_pnts):

                ###------------------------------------------------------------------------------
                ### get info
                ###------------------------------------------------------------------------------
                x_cen_img = None
                y_cen_img = None

                x_left_img = None
                y_left_img = None

                x_right_img = None
                y_right_img = None

                x_cen_3d_this = None
                y_cen_3d_this = None


                ###
                if res_for_show == 0:   # show extracted
                    x_cen_img = arr_xy_cen_img_extracted[idx_pnt, 0]
                    y_cen_img = arr_xy_cen_img_extracted[idx_pnt, 1]

                    x_left_img = arr_xy_left_img_extracted[idx_pnt, 0]
                    y_left_img = arr_xy_left_img_extracted[idx_pnt, 1]

                    x_right_img = arr_xy_right_img_extracted[idx_pnt, 0]
                    y_right_img = arr_xy_right_img_extracted[idx_pnt, 1]
                else:
                    ### 3d coordinate
                    x_cen_3d_this = arr_xyx_cen_3d_polynomial[idx_pnt, 0]
                    y_cen_3d_this = arr_xyx_cen_3d_polynomial[idx_pnt, 1]
                    #z_cen_3d_this = arr_xyx_cen_3d_polynomial[idx_pnt, 2]

                    x_left_3d_this = arr_xyx_left_3d_polynomial[idx_pnt, 0]
                    y_left_3d_this = arr_xyx_left_3d_polynomial[idx_pnt, 1]
                    #z_left_3d_this = arr_xyx_left_3d_polynomial[idx_pnt, 2]

                    x_right_3d_this = arr_xyx_right_3d_polynomial[idx_pnt, 0]
                    y_right_3d_this = arr_xyx_right_3d_polynomial[idx_pnt, 1]
                    #z_right_3d_this = arr_xyx_right_3d_polynomial[idx_pnt, 2]


                    ### img coordinate
                    x_cen_img, y_cen_img = self.m_obj_utils_3D.convert_pnt_world_to_pnt_img_ori(np.array([[x_cen_3d_this], [y_cen_3d_this], [1.0]]))
                    x_left_img, y_left_img = self.m_obj_utils_3D.convert_pnt_world_to_pnt_img_ori(np.array([[x_left_3d_this], [y_left_3d_this], [1.0]]))
                    x_right_img, y_right_img = self.m_obj_utils_3D.convert_pnt_world_to_pnt_img_ori(np.array([[x_right_3d_this], [y_right_3d_this], [1.0]]))
                #end


                ### check if it is over max-y (for visualization)
                if y_cen_3d_this >= max_y_cen_3d_extracted:
                    continue
                # end


                ###
                x_cen_img_int = int(round(x_cen_img))
                y_cen_img_int = int(round(y_cen_img))

                x_left_img_int = int(round(x_left_img))
                y_left_img_int = int(round(y_left_img))

                x_right_img_int = int(round(x_right_img))
                y_right_img_int = int(round(y_right_img))


                ###------------------------------------------------------------------------------------
                ### check if it is in a valid image region
                ###------------------------------------------------------------------------------------
                b_draw = False

                if (0 <= x_cen_img_int) and (x_cen_img_int < w_img) and (0 <= y_cen_img_int) and (y_cen_img_int < h_img) and \
                   (0 <= x_left_img_int) and (x_left_img_int < w_img) and (0 <= y_left_img_int) and (y_left_img_int < h_img) and \
                   (0 <= x_right_img_int) and (x_right_img_int < w_img) and (0 <= y_right_img_int) and (y_right_img_int < h_img):

                    b_draw = True
                #end

                ###
                if b_draw is False:
                    continue
                #end


                ###------------------------------------------------------------------------------------
                ### check if it is a switch edge
                ###------------------------------------------------------------------------------------
                b_switch_edge = False
                b_switch_edge_of_non_ego_path = False

                for idx_sw in range(len(info_xyz_switch_3d_edge_start)):
                    y_switch_edge_3d_this_start = info_xyz_switch_3d_edge_start[idx_sw][1]
                    y_switch_edge_3d_this_end   = info_xyz_switch_3d_edge_end[idx_sw][1]

                    if (y_switch_edge_3d_this_start <= y_cen_3d_this) and (y_cen_3d_this <= y_switch_edge_3d_this_end):
                        b_switch_edge = True
                    #end
                #end

                ###
                if (b_switch_edge is True) and (type_path_this is my_utils_RPG.TYPE_path.NON_EGO):
                    b_switch_edge_of_non_ego_path = True
                #end


                ###------------------------------------------------------------------------------------
                ### check if it is in switch region
                ###------------------------------------------------------------------------------------
                b_switchregion = False

                for idx_sw in range(arr_xyz_switch_3d_extracted.shape[0]):
                    y_switch_3d_this = arr_xyz_switch_3d_extracted[idx_sw, 1]

                    dy = y_cen_3d_this - y_switch_3d_this

                    if (-12.0 <= dy) and (dy <= 5.0):
                        b_switchregion = True
                    #end
                #end


                ###
                if (b_switch_edge_of_non_ego_path is True) and (b_switchregion is False):
                    continue
                #end


                ###------------------------------------------------------------------------------------
                ### draw path region
                ###------------------------------------------------------------------------------------
                type_region = 0     # 0(normal region), 1(switch region)

                if b_switchregion is True:
                    type_region = 1
                #end

                ###
                for x_img in range(x_left_img_int, x_right_img_int + 1):
                    b_old, g_old, r_old = img_bg_in[y_cen_img_int, x_img, :]
                    b_new, g_new, r_new = my_utils_vis.adjust_rgb_for_region(b_old, g_old, r_old, type_region)
                    img_vis_res[y_cen_img_int, x_img, :] = [b_new, g_new, r_new]
                #end
            # end
        # end



        ###------------------------------------------------------------------------------------------------
        ### step 1: draw paths
        ###------------------------------------------------------------------------------------------------
        for idx_draw in range(len(list_paths_final)):
            idx_path = list_idx_path_for_drawing[idx_draw]
            type_path_this = list_paths_final[idx_path]["type_path"]

            # ---------------------------------------------------------------------------------------
            # get info ("extracted") for this path
            # ---------------------------------------------------------------------------------------
            dict_extracted = list_paths_final[idx_path]["extracted"]

            ###
            arr_xy_cen_img_extracted      = dict_extracted["xy_cen_img"]
            arr_xy_left_img_extracted     = dict_extracted["xy_left_img"]
            arr_xy_right_img_extracted    = dict_extracted["xy_right_img"]

            arr_xyz_switch_3d_extracted   = dict_extracted["xyz_switch_3d"]

            info_xyz_switch_3d_edge_start = dict_extracted["xyz_switch_3d_edge_start"]
            info_xyz_switch_3d_edge_end   = dict_extracted["xyz_switch_3d_edge_end"]


            ### just for setting max y for visualization
            arr_xyz_cen_3d_temp     = dict_extracted["xyz_cen_3d"]
            arr_y_cen_3d_temp       = arr_xyz_cen_3d_temp[:, 1]
            max_y_cen_3d_extracted  = max(arr_y_cen_3d_temp)
                # max y for visualization


            # ---------------------------------------------------------------------------------------
            # get info ("polynomial-fitted") for this path
            # ---------------------------------------------------------------------------------------
            dict_polynomial = list_paths_final[idx_path]["polynomial"]

            ###
            arr_xyx_cen_3d_polynomial = dict_polynomial["xyz_cen_3d"]
            arr_xyx_left_3d_polynomial = dict_polynomial["xyz_left_3d"]
            arr_xyx_right_3d_polynomial = dict_polynomial["xyz_right_3d"]


            # ---------------------------------------------------------------------------------------
            #
            # ---------------------------------------------------------------------------------------
            totnum_pnts = 0

            if res_for_show == 0:  # show extracted
                totnum_pnts = arr_xy_cen_img_extracted.shape[0]
            else: # show polynomial
                totnum_pnts = arr_xyx_cen_3d_polynomial.shape[0]
            #end


            # ---------------------------------------------------------------------------------------
            # draw paths
            # ---------------------------------------------------------------------------------------
            for idx_pnt in range(totnum_pnts):

                ###------------------------------------------------------------------------------
                ### get info
                ###------------------------------------------------------------------------------
                x_cen_img = None
                y_cen_img = None

                x_left_img = None
                y_left_img = None

                x_right_img = None
                y_right_img = None

                x_cen_3d_this = None
                y_cen_3d_this = None


                ###
                if res_for_show == 0:   # show extracted
                    x_cen_img = arr_xy_cen_img_extracted[idx_pnt, 0]
                    y_cen_img = arr_xy_cen_img_extracted[idx_pnt, 1]

                    x_left_img = arr_xy_left_img_extracted[idx_pnt, 0]
                    y_left_img = arr_xy_left_img_extracted[idx_pnt, 1]

                    x_right_img = arr_xy_right_img_extracted[idx_pnt, 0]
                    y_right_img = arr_xy_right_img_extracted[idx_pnt, 1]
                else:
                    ###
                    x_cen_3d_this = arr_xyx_cen_3d_polynomial[idx_pnt, 0]
                    y_cen_3d_this = arr_xyx_cen_3d_polynomial[idx_pnt, 1]
                    #z_cen_3d_this = arr_xyx_cen_3d_polynomial[idx_pnt, 2]

                    x_left_3d_this = arr_xyx_left_3d_polynomial[idx_pnt, 0]
                    y_left_3d_this = arr_xyx_left_3d_polynomial[idx_pnt, 1]
                    #z_left_3d_this = arr_xyx_left_3d_polynomial[idx_pnt, 2]

                    x_right_3d_this = arr_xyx_right_3d_polynomial[idx_pnt, 0]
                    y_right_3d_this = arr_xyx_right_3d_polynomial[idx_pnt, 1]
                    #z_right_3d_this = arr_xyx_right_3d_polynomial[idx_pnt, 2]


                    ###
                    x_cen_img, y_cen_img = self.m_obj_utils_3D.convert_pnt_world_to_pnt_img_ori(np.array([[x_cen_3d_this], [y_cen_3d_this], [1.0]]))
                    x_left_img, y_left_img = self.m_obj_utils_3D.convert_pnt_world_to_pnt_img_ori(np.array([[x_left_3d_this], [y_left_3d_this], [1.0]]))
                    x_right_img, y_right_img = self.m_obj_utils_3D.convert_pnt_world_to_pnt_img_ori(np.array([[x_right_3d_this], [y_right_3d_this], [1.0]]))
                #end


                ### check if it is over max-y (for visualization)
                if y_cen_3d_this >= max_y_cen_3d_extracted:
                    continue
                # end


                ###
                x_cen_img_int = int(round(x_cen_img))
                y_cen_img_int = int(round(y_cen_img))

                x_left_img_int = int(round(x_left_img))
                y_left_img_int = int(round(y_left_img))

                x_right_img_int = int(round(x_right_img))
                y_right_img_int = int(round(y_right_img))



                ###------------------------------------------------------------------------------
                ### DRAW
                ###------------------------------------------------------------------------------
                val_b_cen = 0
                val_g_cen = 128
                val_r_cen = 0

                val_b_left = 20
                val_g_left = 100
                val_r_left = 250

                val_b_right = 250
                val_g_right = 200
                val_r_right = 0


                ###
                b_draw = False

                if (0 <= x_cen_img_int) and (x_cen_img_int < w_img) and (0 <= y_cen_img_int) and (y_cen_img_int < h_img) and \
                   (0 <= x_left_img_int) and (x_left_img_int < w_img) and (0 <= y_left_img_int) and (y_left_img_int < h_img) and \
                   (0 <= x_right_img_int) and (x_right_img_int < w_img) and (0 <= y_right_img_int) and (y_right_img_int < h_img):

                    b_draw = True
                #end

                if b_draw is False:
                    continue
                #end


                ###------------------------------------------------------------------------------------
                ### check if it is a switch edge
                ###------------------------------------------------------------------------------------
                b_switch_edge = False
                b_switch_edge_of_non_ego_path = False

                for idx_sw in range(len(info_xyz_switch_3d_edge_start)):
                    y_switch_edge_3d_this_start = info_xyz_switch_3d_edge_start[idx_sw][1]
                    y_switch_edge_3d_this_end   = info_xyz_switch_3d_edge_end[idx_sw][1]

                    if (y_switch_edge_3d_this_start <= y_cen_3d_this) and (y_cen_3d_this <= y_switch_edge_3d_this_end):
                        b_switch_edge = True
                    #end
                #end

                if (b_switch_edge is True) and (type_path_this is my_utils_RPG.TYPE_path.NON_EGO):
                    b_switch_edge_of_non_ego_path = True
                #end


                ###------------------------------------------------------------------------------------
                ### check if it is in switch region
                ###------------------------------------------------------------------------------------
                b_switchregion = False

                for idx_sw in range(arr_xyz_switch_3d_extracted.shape[0]):
                    y_switch_3d_this = arr_xyz_switch_3d_extracted[idx_sw, 1]

                    dy = y_cen_3d_this - y_switch_3d_this

                    if (-12.0 <= dy) and (dy <= 5.0):
                        b_switchregion = True
                    #end
                #end


                ###
                if (b_switch_edge_of_non_ego_path is True) and (b_switchregion is False):
                    continue
                #end


                ###
                val_radius = 1

                if type_path_this is my_utils_RPG.TYPE_path.EGO:
                    val_radius = 3
                #end

                #cv2.circle(img_vis_res, center=(x_cen_img_int, y_cen_img_int), radius=2, color=(val_b_cen, val_g_cen, val_r_cen), thickness=-1)
                cv2.circle(img_vis_res, center=(x_left_img_int, y_left_img_int), radius=val_radius, color=(val_b_left, val_g_left, val_r_left), thickness=-1)
                cv2.circle(img_vis_res, center=(x_right_img_int, y_right_img_int), radius=val_radius, color=(val_b_right, val_g_right, val_r_right), thickness=-1)
            # end
        # end



        ###------------------------------------------------------------------------------------------------
        ### show
        ###------------------------------------------------------------------------------------------------
        cv2.imshow('final_path_on_ori', img_vis_res)
        cv2.waitKey(1)

        ### save (temp)
        if 0:
            #fname_output_temp = "/home/yu1/Desktop/dir_temp/temp_res0/final_path_on_ori/" + "final_path_ori_" + str(self.m_temp_idx_res_img_on_ori) + '.jpg'
            fname_output_temp = "/home/yu1/Desktop/dir_res_path_full/nyc_temp/" + "final_path_ori_" + str(self.m_temp_idx_res_img_on_ori) + '.png'
            cv2.imwrite(fname_output_temp, img_vis_res)
            self.m_temp_idx_res_img_on_ori += 1
        #end

    #end



    ###=========================================================================================================
    ### visualize final path on ipm
    ###=========================================================================================================
    def show_final_path_on_ipm(self, list_paths_final, img_bg, res_for_show=1):
        #---------------------------------------------------------------------------------------------
        # dict_path_final = {"extracted": dict_path_this,
        #                    "polynomial": dict_path_poly_this}
        #
        #    "extracted": dict_path_this = {"xy_cen_img": [],
        #                                   "xy_left_img": [],
        #                                   "xy_right_img": [],
        #                                   "xyz_cen_3d": [],
        #                                   "xyz_left_3d": [],
        #                                   "xyz_right_3d": []}
        #
        #    "polynomial": dict_path_poly_this = {"xyz_cen_3d": arr_xyz_sample_ori,
        #                                         "xyz_left_3d": [],
        #                                         "xyz_right_3d": []}
        #---------------------------------------------------------------------------------------------
        # img_bg: img_raw_rsz_uint8
        #---------------------------------------------------------------------------------------------
        #   res_for_show=0: show "extracted"
        #   res_for_show=1: show "polynomial"
        #---------------------------------------------------------------------------------------------


        ###
        assert (self.m_obj_utils_3D is not None)

        #img_raw_rsz_uint8 = self.m_obj_utils_rpg.get_img_raw_rsz_uint8()
        #assert (img_raw_rsz_uint8 is not None)


        ###------------------------------------------------------------------------------------------------
        ### create img for visualization
        ###------------------------------------------------------------------------------------------------
        #img_vis_ipm_rgb   = self.m_obj_utils_3D.create_img_IPM(img_raw_rsz_uint8)
        img_vis_ipm_rgb = self.m_obj_utils_3D.create_img_IPM(img_bg)
        img_vis_ipm_gray1 = cv2.cvtColor(img_vis_ipm_rgb, cv2.COLOR_BGR2GRAY)

        img_vis_ipm_gray3 = np.zeros_like(img_vis_ipm_rgb)  # img_ipm_gray3: 3-ch gray img
        img_vis_ipm_gray3[:, :, 0] = img_vis_ipm_gray1
        img_vis_ipm_gray3[:, :, 1] = img_vis_ipm_gray1
        img_vis_ipm_gray3[:, :, 2] = img_vis_ipm_gray1

        ###
        h_img_bev, w_img_bev = self.m_obj_utils_3D.get_size_img_bev()


        ###------------------------------------------------------------------------------------------------
        ###
        ###------------------------------------------------------------------------------------------------
        # h_img_temp = img_raw_rsz_uint8.shape[0]
        # w_img_temp = img_raw_rsz_uint8.shape[1]
        h_img_temp = img_bg.shape[0]
        w_img_temp = img_bg.shape[1]

        x_pnt_bev, y_pnt_bev = self.m_obj_utils_3D.convert_pnt_img_ori_to_pnt_bev(np.array([[w_img_temp/2], [h_img_temp - 1], [1.0]]))

        x_bev_visualize_int = int(round(x_pnt_bev))
        y_bev_visualize_int = int(round(y_pnt_bev))


        ###------------------------------------------------------------------------------------------------
        ### process each section
        ###------------------------------------------------------------------------------------------------
        for idx_path in range(len(list_paths_final)):
            # val_r = int(self.m_rgb_cluster[idx_path, 0])
            # val_g = int(self.m_rgb_cluster[idx_path, 1])
            # val_b = int(self.m_rgb_cluster[idx_path, 2])

            #------------------------------------------------------------------------------
            #
            #------------------------------------------------------------------------------
            dict_extracted = list_paths_final[idx_path]["extracted"]

            arr_xyz_cen_temp = dict_extracted["xyz_cen_3d"]
            arr_y_cen_temp   = arr_xyz_cen_temp[:, 1]

            max_y_cen_extracted = max(arr_y_cen_temp)


            #------------------------------------------------------------------------------
            #
            #------------------------------------------------------------------------------
            dict_polynomial = list_paths_final[idx_path]["polynomial"]

            arr_xyz_cen = None
            arr_xyz_left = None
            arr_xyz_right = None

            if res_for_show == 0:
                arr_xyz_cen = dict_extracted["xyz_cen_3d"]
                arr_xyz_left = dict_extracted["xyz_left_3d"]
                arr_xyz_right = dict_extracted["xyz_right_3d"]
            else:
                arr_xyz_cen = dict_polynomial["xyz_cen_3d"]
                arr_xyz_left = dict_polynomial["xyz_left_3d"]
                arr_xyz_right = dict_polynomial["xyz_right_3d"]
            #end


            for idx_pnt in range(arr_xyz_cen.shape[0]):
                ###
                x_cen_3d = arr_xyz_cen[idx_pnt, 0]
                y_cen_3d = arr_xyz_cen[idx_pnt, 1]

                x_left_3d = arr_xyz_left[idx_pnt, 0]
                y_left_3d = arr_xyz_left[idx_pnt, 1]

                x_right_3d = arr_xyz_right[idx_pnt, 0]
                y_right_3d = arr_xyz_right[idx_pnt, 1]

                if y_cen_3d >= max_y_cen_extracted:
                    continue
                #end


                ###
                x_cen_bev, y_cen_bev = self.m_obj_utils_3D.convert_pnt_world_to_pnt_bev(np.array([[x_cen_3d], [y_cen_3d], [1.0]]))
                x_left_bev, y_left_bev = self.m_obj_utils_3D.convert_pnt_world_to_pnt_bev(np.array([[x_left_3d], [y_left_3d], [1.0]]))
                x_right_bev, y_right_bev = self.m_obj_utils_3D.convert_pnt_world_to_pnt_bev(np.array([[x_right_3d], [y_right_3d], [1.0]]))


                ###
                x_cen_bev_int = int(round(x_cen_bev))
                y_cen_bev_int = int(round(y_cen_bev))

                x_left_bev_int = int(round(x_left_bev))
                y_left_bev_int = int(round(y_left_bev))

                x_right_bev_int = int(round(x_right_bev))
                y_right_bev_int = int(round(y_right_bev))


                ###
                if y_cen_bev_int >= y_bev_visualize_int:
                    continue
                #end


                ### draw pnt (cen)
                val_b = 0
                val_g = 128
                val_r = 0

                if (0 <= x_cen_bev_int) and (x_cen_bev_int < w_img_bev) and (0 <= y_cen_bev_int) and (y_cen_bev_int < h_img_bev):
                    cv2.circle(img_vis_ipm_gray3, center=(x_cen_bev_int, y_cen_bev_int), radius=2, color=(val_b, val_g, val_r), thickness=-1)
                #end

                ### draw pnt (left)
                val_b = 20
                val_g = 100
                val_r = 250

                if (0 <= x_left_bev_int) and (x_left_bev_int < w_img_bev) and (0 <= y_left_bev_int) and (y_left_bev_int < h_img_bev):
                    cv2.circle(img_vis_ipm_gray3, center=(x_left_bev_int, y_left_bev_int), radius=2, color=(val_b, val_g, val_r), thickness=-1)
                #end

                ### draw pnt (right)
                val_b = 250
                val_g = 250
                val_r = 0

                if (0 <= x_right_bev_int) and (x_right_bev_int < w_img_bev) and (0 <= y_right_bev_int) and (y_right_bev_int < h_img_bev):
                    cv2.circle(img_vis_ipm_gray3, center=(x_right_bev_int, y_right_bev_int), radius=2, color=(val_b, val_g, val_r), thickness=-1)
                #end

            #end
        #end


        ###------------------------------------------------------------------------------------------------
        ### show
        ###------------------------------------------------------------------------------------------------
        cv2.imshow('final_path_on_ipm', img_vis_ipm_gray3)
        cv2.waitKey(1)


        ### save (temp)
        if 0:
            #fname_output_temp = "/home/yu1/Desktop/dir_temp/temp_res0/final_path_on_ipm/" + "final_path_ipm_" + str(self.m_temp_idx_res_img_on_ipm) + '.jpg'
            fname_output_temp = "/home/yu1/Desktop/dir_res_path/NYC/final_path_on_ipm/" + "final_path_ipm_" + str(self.m_temp_idx_res_img_on_ori) + '.png'
            cv2.imwrite(fname_output_temp, img_vis_ipm_gray3)
            self.m_temp_idx_res_img_on_ipm += 1
        #end
    #end


    ###=========================================================================================================
    ###
    ###=========================================================================================================
    def show_imgs_res_interim(self, dir_output, fname_img_in, img_raw_in, img_res_seg, img_res_centerness_combined, img_res_triplet_localmax, b_save=False):
        """
        show result images and save them as files (if wanted)

        :param dir_output:
        :param fname_img_in:
        :param img_raw_in:
        :param img_res_seg:
        :param img_res_centerness_combined:
        :param img_res_triplet_localmax:
        :param b_save:
        """


        ###------------------------------------------------------------------------------------------------
        ### set fname (for saving)
        ###------------------------------------------------------------------------------------------------

        ###
        # fname_out_img_raw_in                  = dir_output + 'in_' + fname_img_in
        # fname_out_img_res_seg                 = dir_output + 'seg_' + fname_img_in
        # fname_out_img_res_centerness_combined = dir_output + 'centerness_' + fname_img_in
        # fname_out_img_res_vis_temp2           = dir_output + 'triplet_' + fname_img_in
        fname_out_img_raw_in                  = dir_output + '/in/'         + 'in_'         + fname_img_in
        fname_out_img_res_seg                 = dir_output + '/seg/'        + 'seg_'        + fname_img_in
        fname_out_img_res_centerness_combined = dir_output + '/centerness/' + 'centerness_' + fname_img_in
        fname_out_img_res_vis_temp2           = dir_output + '/triplet/'    + 'triplet_'    + fname_img_in


        ###------------------------------------------------------------------------------------------------
        ### show and save
        ###------------------------------------------------------------------------------------------------

        ### show
        cv2.imshow('img_raw_in', img_raw_in)
        cv2.imshow('img_res_seg', img_res_seg)
        cv2.imshow('img_res_centerness', img_res_centerness_combined)  # combination of direct and LR
        cv2.imshow('img_res_triplet_localmax', img_res_triplet_localmax)  # center, and corresponding left, right


        ### save
        if b_save is True:
            cv2.imwrite(fname_out_img_raw_in, img_raw_in)
            cv2.imwrite(fname_out_img_res_seg, img_res_seg)
            cv2.imwrite(fname_out_img_res_centerness_combined, img_res_centerness_combined)
            cv2.imwrite(fname_out_img_res_vis_temp2, img_res_triplet_localmax)
        #end

        cv2.waitKey(0)

    #END



#end


########################################################################################################################



