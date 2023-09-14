# 2020/7/10
# Jungwon Kang


import os
import torch
import numpy as np
import collections
import cv2
import copy
from scipy.signal import find_peaks



########################################################################################################################
###
########################################################################################################################
class MyUtils_Image:

    ###
    m_param_triplet_nms_alpha = None
    m_param_triplet_nms_beta = None
    m_param_triplet_nms_min = None
    m_param_triplet_nms_scale = None


    m_temp_idx_res_img = 0  # temp


    ###############################################################################################################
    ###
    ###############################################################################################################
    def __init__(self, dict_args=None):

        if dict_args is not None:
            self.m_param_triplet_nms_alpha = dict_args["param_triplet_nms_alpha"]
            self.m_param_triplet_nms_beta  = dict_args["param_triplet_nms_beta"]
            self.m_param_triplet_nms_min   = dict_args["param_triplet_nms_min"]
            self.m_param_triplet_nms_scale = dict_args["param_triplet_nms_scale"]
        #end
    #end


    ###############################################################################################################
    ###
    ###############################################################################################################
    def read_img_raw_jpg_from_file(self, full_fname_img_raw_jpg, size_img_rsz,
                                    rgb_mean = np.array([128.0, 128.0, 128.0]) / 255.0,
                                    rgb_std = np.array([1.0, 1.0, 1.0])):
        """
        read a raw image from a file and convert uint8-type image into floating-type image (with normalization)

        :param full_fname_img_raw_jpg
        :param size_img_rsz
        :param rgb_mean
        :param rgb_std
        :return: img_raw_rsz_uint8: ndarray(H,W,C), 0~255
                 img_raw_rsz_fl_n_final: ndarray(C,H,W), -X.0 ~ X.0, BGR
        """


        ###================================================================================================
        ### read img_raw_jpg
        ###================================================================================================
        img_raw = cv2.imread(full_fname_img_raw_jpg)
            # completed to set
            #       img_raw: ndarray(H,W,C), 0 ~ 255

            # Note that opencv uses BGR, that is:
            #   img_raw[:,:,0] -> B
            #   img_raw[:,:,1] -> G
            #   img_raw[:,:,2] -> R


        ###================================================================================================
        ### resize img
        ###================================================================================================
        img_raw_rsz_uint8 = cv2.resize(img_raw, (size_img_rsz['w'], size_img_rsz['h']))
            # completed to set
            #       img_raw_rsz_uint8

        ### <<debugging>>
        if 0:
            cv2.imshow('img_raw_rsz_uint8', img_raw_rsz_uint8)
            cv2.waitKey(1)
        # end


        ###================================================================================================
        ### convert img_raw to img_data
        ###================================================================================================
        img_raw_rsz_fl_n_final = self.convert_img_ori_to_img_data(img_raw_rsz_uint8)
            # completed to set
            #       img_raw_rsz_fl_n_final: ndarray(C,H,W), -X.0 ~ X.0


        ### <<debugging>>
        if 0:
            img_raw_temp0 = convert_img_data_to_img_ori(img_raw_rsz_fl_n_final)
            cv2.imshow('img_raw_temp0', img_raw_temp0)
            cv2.waitKey(1)
        # end


        return img_raw_rsz_uint8, img_raw_rsz_fl_n_final
    #end


    ###############################################################################################################
    ###
    ###############################################################################################################
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


    ###############################################################################################################
    ###
    ###############################################################################################################
    def convert_img_data_to_img_ori(self, img_data_fl_n,
                                    rgb_mean=np.array([128.0, 128.0, 128.0]) / 255.0,
                                    rgb_std=np.array([1.0, 1.0, 1.0])):
        """
        convert floating-type image (with normalization) into uint8-type image

        :param img_data_fl_n: ndarray(C,H,W), -X.0 ~ X.0, BGR
        :param rgb_mean
        :param rgb_std
        :return: img_ori_uint8: ndarray(H,W,C), 0 ~ 255, BGR
        """


        #/////////////////////////////////////////////////////////////////////////////////////////////////////////
        # convert img_data to img_raw
        # <output>
        #   img_data_fl_n: ndarray(C,H,W), -X.0 ~ X.0, BGR
        # <input>
        #   img_ori_uint8: ndarray(H,W,C), 0 ~ 255, BGR
        #
        #   we are doing the following things:
        #   (1) convert CHW -> HWC
        #   (2) apply rgb_std
        #   (3) apply rgb_mean
        #   (4) de-normalize so that 0.0~1.0 -> 0~255
        #   (5) make it uint8
        #/////////////////////////////////////////////////////////////////////////////////////////////////////////

        img_out = copy.deepcopy(img_data_fl_n)

        ###================================================================================================
        ### (1) convert CHW -> HWC
        ###================================================================================================
        img_data_fl_n = img_data_fl_n.transpose(1, 2, 0)
            # C(0),H(1),W(2) -> H(1),W(2),C(0)
            # completed to set
            #       img_data_fl_n: ndarray(H,W,C)


        ###================================================================================================
        ### (2) apply rgb_std
        ###================================================================================================
        img_data_fl_n = img_data_fl_n*rgb_std


        ###================================================================================================
        ### (3) apply rgb_mean
        ###================================================================================================
        img_data_fl_n = img_data_fl_n + rgb_mean


        ###================================================================================================
        ### (4) de-normalize so that 0.0~1.0 -> 0~255
        ###================================================================================================
        img_data_fl_n = img_data_fl_n*255.0


        ###================================================================================================
        ### (5) make it uint8
        ###================================================================================================
        img_ori_uint8 = img_data_fl_n.astype(np.uint8)


        return img_ori_uint8
    #end


    ###############################################################################################################
    ###
    ###############################################################################################################
    def decode_segmap(self, labelmap):
        """
        decode and visualize labelmap into visible result image

        :param labelmap: label_map, ndarray (h, w)
        :return: img_label_rgb
        """

        # labelmap: label_map, ndarray (h, w)


        ###------------------------------------------------------------------------------------------
        ### setting
        ###------------------------------------------------------------------------------------------
        n_classes = 19

        ###
        rgb_class00 = [128,  64, 128]   # 00: road
        rgb_class01 = [244,  35, 232]   # 01: sidewalk
        rgb_class02 = [ 70,  70,  70]   # 02: construction
        rgb_class03 = [192,   0, 128]   # 03: tram-region
        rgb_class04 = [190, 153, 153]   # 04: fence
        rgb_class05 = [153, 153, 153]   # 05: pole
        rgb_class06 = [250, 170,  30]   # 06: traffic-light
        rgb_class07 = [220, 220,   0]   # 07: traffic-sign
        rgb_class08 = [107, 142,  35]   # 08: vegetation
        rgb_class09 = [152, 251, 152]   # 09: terrain
        rgb_class10 = [ 70, 130, 180]   # 10: sky
        rgb_class11 = [220,  20,  60]   # 11: human
        rgb_class12 = [230, 150, 140]   # 12: rail-region
        rgb_class13 = [  0,   0, 142]   # 13: car
        rgb_class14 = [  0,   0,  70]   # 14: truck
        rgb_class15 = [ 90,  40,  40]   # 15: trackbed
        rgb_class16 = [  0,  80, 100]   # 16: on-rails
        rgb_class17 = [  0, 254, 254]   # 17: rail-line_train
        rgb_class18 = [  0,  68,  63]   # 18: rail-line_tram


        ###
        rgb_labels = np.array(
            [
                rgb_class00,
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
            ]
        )


        ###------------------------------------------------------------------------------------------
        ### convert label_map into img_label_rgb
        ###------------------------------------------------------------------------------------------

        ### create default img
        r = np.ones_like(labelmap )*250          # 250: indicating invalid label
        g = np.ones_like(labelmap )*250
        b = np.ones_like(labelmap )*250

        for l in range(0, n_classes):
            ### find
            idx_set = (labelmap == l)           # idx_set: ndarray, (h, w), bool

            ### assign
            r[idx_set] = rgb_labels[l, 0]       # r: 0 ~ 255
            g[idx_set] = rgb_labels[l, 1]       # g: 0 ~ 255
            b[idx_set] = rgb_labels[l, 2]       # b: 0 ~ 255
        # end

        img_label_rgb = np.zeros((labelmap.shape[0], labelmap.shape[1], 3))
        img_label_rgb[:, :, 0] = r
        img_label_rgb[:, :, 1] = g
        img_label_rgb[:, :, 2] = b



        img_label_rgb = img_label_rgb[:, :, ::-1]       # rgb -> bgr (for following opencv convention)
        img_label_rgb = img_label_rgb.astype(np.uint8)


        return img_label_rgb
    #end


    ###############################################################################################################
    ###
    ###############################################################################################################
    def decode_output_centerness(self, res_in, num_channel_reg):
        """
        decode and visualize centerness output

        :param res_in
        :return: res_out, img_res_out
        """
        if num_channel_reg == 1:
            res_relu = torch.relu(res_in)
            res_a = torch.clamp(res_relu[0], min=0.0, max=100.0)
            res_b = res_a[0].detach().cpu().numpy()

            res_out = res_b
            img_res_out = res_b.astype(np.uint8)


        elif num_channel_reg == 3:
            res_sigmoid = torch.clamp(torch.sigmoid(res_in), min=1e-4, max=1 - 1e-4)

            ###
            res_a = res_sigmoid[0].permute(1, 2, 0)     # res_a : tensor(512, 1024, 1)
            res_b = res_a[:, :, 0]                      # res_b : tensor(512, 1024)
            res_c = res_b * 255.0
            res_d = torch.clamp(res_c, min=0.0, max=255.0)
            res_e = res_d.detach().cpu().numpy()


            ###
            res_out = res_b.detach().cpu().numpy()
                # completed to set
                #       res_out: ndarray (h,w), 0.0 ~ 1.0

            ###
            img_res_out = res_e.astype(np.uint8)
                # completed to set
                #       img_res_out: ndarray (h,w), uint8


        return res_out, img_res_out
    #end


    ###############################################################################################################
    ###
    ###############################################################################################################
    def decode_output_leftright(self, res_in):
        """
        decode and visualize left & right rail estimation output

        :param res_in
        :return: res_left, res_right, img_res_left, img_res_right
        """

        ###
        res_relu = torch.relu(res_in)


        ###
        res0     = res_relu[0]                                          # res0: tensor (2, h, w)

        ###
        res_left_a      = res0[0, :, :]                                 # res_left  : tensor(h, w)
        res_left_b      = torch.clamp(res_left_a, min=0.0, max=255.0)
        res_left_c      = res_left_b.detach().cpu().numpy()
        img_res_left    = res_left_c.astype(np.uint8)
            # completed to set
            #       img_res_left: ndarray (h,w), uint8


        ###
        res_right_a     = res0[1, :, :]                                 # res_right : tensor(h, w)
        res_right_b     = torch.clamp(res_right_a, min=0.0, max=255.0)
        res_right_c     = res_right_b.detach().cpu().numpy()
        img_res_right   = res_right_c.astype(np.uint8)
            # completed to set
            #       img_res_right: ndarray (h,w), uint8


        ###
        res_left = res_left_c
        res_right = res_right_c
            # completed to set
            #       res_left: ndarray (h,w) float32     0.0 ~ X.X
            #       res_right: ndarray (h,w) float32    0.0 ~ X.X


        return res_left, res_right, img_res_left, img_res_right
    #end


    ###############################################################################################################
    ###
    ###############################################################################################################
    def adjust_rgb(self, type, b_old_uint8, g_old_uint8, r_old_uint8):
        """
        adjust rgb for a pixel (for visualization)

        :param type:
        :param b_old_uint8:
        :param g_old_uint8:
        :param r_old_uint8:
        :return: b_new_int, g_new_int, r_new_int
        """

        ###
        dr_int = 0
        dg_int = 0
        db_int = 0

        if type == 0:       # track region
            dr_int = 0
            dg_int = 100
            db_int = 0
        elif type == 1:     # left
            dr_int = 100
            dg_int = 0
            db_int = 0
        elif type == 2:     # right
            dr_int = 0
            dg_int = 0
            db_int = 100
        elif type == 3:     # center
            dr_int = 0
            dg_int = 200
            db_int = 0
        #end


        ###
        r_new_int = int(r_old_uint8) + dr_int
        g_new_int = int(g_old_uint8) + dg_int
        b_new_int = int(b_old_uint8) + db_int


        ###
        r_new_int = min(r_new_int, 255)
        r_new_int = max(r_new_int, 0)

        g_new_int = min(g_new_int, 255)
        g_new_int = max(g_new_int, 0)

        b_new_int = min(b_new_int, 255)
        b_new_int = max(b_new_int, 0)


        ###
        return b_new_int, g_new_int, r_new_int
    #end


    ###############################################################################################################
    ###
    ###############################################################################################################
    def visualize_res_triplet_localmax(self, img_raw_rsz_uint8, res_centerness, res_left, res_right):
        """
        visualize triplet point extraction result showing only local maxima of triplet points

        :param img_raw_rsz_uint8:
        :param res_centerness:
        :param res_left:
        :param res_right:
        :return: img_res_rgb
        """


        ###=========================================================================================================
        ### show centerline and corresponding left, right rails
        ###
        ### res_centerness: 0.0 ~ 1.0, float32
        ### res_left:       0.0 ~ X.X, float32
        ### res_right:      0.0 ~ X.X, float32
        ###=========================================================================================================

        # <scipy.signal.find_peaks>
        #   https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html
        #   https://stackoverflow.com/questions/31070563/find-all-local-maxima-and-minima-when-x-and-y-values-are-given-as-numpy-arrays/31073798


        h_img = res_centerness.shape[0]
        w_img = res_centerness.shape[1]


        ###================================================================================================
        ### find local maxima
        ###================================================================================================
        mat_local_max = np.zeros_like(res_centerness)

        for y in range(0, h_img):
            ###
            centerness_thisrow = res_centerness[y, :]

            ###
            #dist_min = (235.0/270.0)*y - 220.0
            #dist_min = max(15.0, dist_min)
            #dist_min = dist_min*0.6

            dist_min = (self.m_param_triplet_nms_alpha)*y + (self.m_param_triplet_nms_beta)
            dist_min = max((self.m_param_triplet_nms_min), dist_min)
            dist_min = dist_min*(self.m_param_triplet_nms_scale)

            ###
            set_x_peaks, _ = find_peaks(centerness_thisrow, height=0.5, distance=dist_min)


            ###
            num_local_max = set_x_peaks.size

            for i in range(0, num_local_max):
                x_this = set_x_peaks[i]
                c_this = centerness_thisrow[x_this]
                mat_local_max[y, x_this] = c_this
            #end
        #end
            # completed to set
            #       mat_local_max: ndarray (h,w), 0.0 ~ 1.0


        ### <debugging>>
        # img_mat_local_max = mat_local_max*255.0
        # img_mat_local_max = img_mat_local_max.astype(np.uint8)
        # img_mat_local_max_rgb = cv2.cvtColor(img_mat_local_max, cv2.COLOR_GRAY2BGR)
        # cv2.imshow('img_mat_local_max_rgb', img_mat_local_max_rgb)
        # cv2.waitKey(1)


        ###================================================================================================
        ###
        ###================================================================================================

        ###
        yx_center = np.where(mat_local_max >= 0.5)

        set_y_center = yx_center[0]
        set_x_center = yx_center[1]

        totnum_yx = set_y_center.shape[0]
            # completed to set
            #       set_y_center
            #       set_x_center
            #       totnum_yx


        ###================================================================================================
        ###
        ###================================================================================================
        img_res_rgb = copy.deepcopy(img_raw_rsz_uint8)

        ### fill region
        for idx_yx in range(0, totnum_yx):
            y_cen = set_y_center[idx_yx]
            x_cen = set_x_center[idx_yx]

            dx_left  = res_left [y_cen, x_cen]
            dx_right = res_right[y_cen, x_cen]

            y_this  = y_cen
            x_left  = max(0, int(round(x_cen - dx_left)))
            x_right = min(int(round(x_cen + dx_right)), w_img - 1)

            ### fill region
            for x_this in range(x_left, x_right + 1):
                b_old, g_old, r_old = img_res_rgb[y_this, x_this, :]
                b_new, g_new, r_new = self.adjust_rgb(0, b_old, g_old, r_old)
                img_res_rgb[y_this, x_this, :] = (b_new, g_new, r_new)
            #end
        #end
            # completed to set
            #       img_res_rgb


        ### draw pnts
        for idx_yx in range(0, totnum_yx):
            y_cen = set_y_center[idx_yx]
            x_cen = set_x_center[idx_yx]

            dx_left  = res_left [y_cen, x_cen]
            dx_right = res_right[y_cen, x_cen]

            y_this  = y_cen
            x_left  = max(0, int(round(x_cen - dx_left)))
            x_right = min(int(round(x_cen + dx_right)), w_img - 1)

            ### draw pnts
            cv2.circle(img_res_rgb, center=(x_left, y_this),  radius=3, color=(20,  100, 250), thickness=-1)
            cv2.circle(img_res_rgb, center=(x_right, y_this), radius=3, color=(250, 250,   0), thickness=-1)
            cv2.circle(img_res_rgb, center=(x_cen, y_this),   radius=3, color=(0,   128,   0), thickness=-1)
        #end
            # completed to set
            #       img_res_rgb


        ###================================================================================================
        ###
        ###================================================================================================
        # alpha = 0.3
        # beta  = 1.0 - alpha
        # img_res_final = cv2.addWeighted(src1=img_res_region_rgb, alpha=alpha, src2=img_res_pnt_rgb, beta=beta, gamma=0)


        ###================================================================================================
        ###
        ###================================================================================================
        # cv2.imshow('visualize_res_triplet_localmax', img_res_rgb)
        # cv2.waitKey(1)


        return img_res_rgb
    #end


    ###############################################################################################################
    ###
    ###############################################################################################################
    def extract_triplet_pnts_localmax(self, res_centerness, res_left, res_right, obj_utils_3D):
        """
        extract triplet points (only local maxima of triplet points)

        :param img_raw_rsz_uint8:
        :param res_centerness:
        :param res_left:
        :param res_right:
        :return: img_res_rgb
        """


        ###=========================================================================================================
        ### show centerline and corresponding left, right rails
        ###
        ### res_centerness: 0.0 ~ 1.0, float32
        ### res_left:       0.0 ~ X.X, float32
        ### res_right:      0.0 ~ X.X, float32
        ###=========================================================================================================

        # <scipy.signal.find_peaks>
        #   https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html
        #   https://stackoverflow.com/questions/31070563/find-all-local-maxima-and-minima-when-x-and-y-values-are-given-as-numpy-arrays/31073798


        h_img = res_centerness.shape[0]
        w_img = res_centerness.shape[1]


        ###================================================================================================
        ### find local maxima
        ###================================================================================================
        list_dict_local_max = [[] for _ in range(h_img)]


        for y in range(270, h_img):
            centerness_thisrow = res_centerness[y, :]
            ###
            dist_min = (self.m_param_triplet_nms_alpha)*y + (self.m_param_triplet_nms_beta)
            dist_min = max((self.m_param_triplet_nms_min), dist_min)
            dist_min = dist_min*(self.m_param_triplet_nms_scale)
            ###
            if res_left is None and res_right is None:
                set_x_peaks, _ = find_peaks(centerness_thisrow, height=dist_min, distance=10)
            else:
                set_x_peaks, _ = find_peaks(centerness_thisrow, height=0.5, distance=10)


            ###
            num_local_max = set_x_peaks.size

            for i in range(0, num_local_max):
                ### get data of local max
                x_this = set_x_peaks[i]
                c_this = centerness_thisrow[x_this]                    # in the new method containing only cen as output, c_this is distance to the left/right

                ###
                x_cen = int(x_this)
                y_cen = y

                ###
                if res_left is None and res_right is None:
                    x_left = max(0, int(x_cen-c_this))
                    x_right = min(int(x_cen+c_this), w_img - 1)
                else:
                    dx_left = res_left[y_cen, x_cen]
                    dx_right = res_right[y_cen, x_cen]
                    x_left = max(0, int(round(x_cen - dx_left)))
                    x_right = min(int(round(x_cen + dx_right)), w_img - 1)



                ###
                xy_cen_img = [x_cen, y_cen]
                xy_left_img = [x_left, y_cen]
                xy_right_img = [x_right, y_cen]

                ###
                x_cen_3d, y_cen_3d, z_cen_3d       = obj_utils_3D.convert_pnt_img_ori_to_pnt_world( np.array([[x_cen],   [y_cen], [1.0]]) )
                x_left_3d, y_left_3d, z_left_3d    = obj_utils_3D.convert_pnt_img_ori_to_pnt_world( np.array([[x_left],  [y_cen], [1.0]]) )
                x_right_3d, y_right_3d, z_right_3d = obj_utils_3D.convert_pnt_img_ori_to_pnt_world( np.array([[x_right], [y_cen], [1.0]]) )

                ### append data
                dict_pnt_this = {"centerness": c_this,
                                 "xy_cen_img": [x_cen, y_cen],
                                 "xy_left_img": [x_left, y_cen],
                                 "xy_right_img": [x_right, y_cen],
                                 "xyz_cen_3d": [x_cen_3d, y_cen_3d, z_cen_3d],
                                 "xyz_left_3d": [x_left_3d, y_left_3d, z_left_3d],
                                 "xyz_right_3d": [x_right_3d, y_right_3d, z_right_3d]}

                ###
                list_dict_local_max[y].append(dict_pnt_this)
            #end
        #end


        ### <debugging>>
        # img_mat_local_max = mat_local_max*255.0
        # img_mat_local_max = img_mat_local_max.astype(np.uint8)
        # img_mat_local_max_rgb = cv2.cvtColor(img_mat_local_max, cv2.COLOR_GRAY2BGR)
        # cv2.imshow('img_mat_local_max_rgb', img_mat_local_max_rgb)
        # cv2.waitKey(1)

        ###================================================================================================
        ### debugging (visualize)
        ###================================================================================================
        if 0:
            ### create bg img
            img_ipm_rgb = obj_utils_3D.create_img_IPM(img_raw_rsz_uint8)
            img_ipm_gray1 = cv2.cvtColor(img_ipm_rgb, cv2.COLOR_BGR2GRAY)

            img_ipm_gray3 = np.zeros_like(img_ipm_rgb)      # img_ipm_gray3: 3-ch gray img
            img_ipm_gray3[:, :, 0] = img_ipm_gray1
            img_ipm_gray3[:, :, 1] = img_ipm_gray1
            img_ipm_gray3[:, :, 2] = img_ipm_gray1


            ###
            h_img_bev, w_img_bev = obj_utils_3D.get_size_img_bev()


            ###
            for y in range(0, h_img):
                list_thisrow = list_dict_local_max[y]

                if len(list_thisrow) == 0:
                    continue
                #end

                for dict_this in list_thisrow:
                    xy_cen_img   = dict_this["xy_cen_img"]
                    xy_left_img  = dict_this["xy_left_img"]
                    xy_right_img = dict_this["xy_right_img"]

                    x_cen_bev, y_cen_bev     = obj_utils_3D.convert_pnt_img_ori_to_pnt_bev( np.array([[xy_cen_img[0]], [xy_cen_img[1]], [1.0]]) )
                    x_left_bev, y_left_bev   = obj_utils_3D.convert_pnt_img_ori_to_pnt_bev( np.array([[xy_left_img[0]], [xy_left_img[1]], [1.0]]) )
                    x_right_bev, y_right_bev = obj_utils_3D.convert_pnt_img_ori_to_pnt_bev( np.array([[xy_right_img[0]], [xy_right_img[1]], [1.0]]) )


                    ###
                    x_cen_bev_int = int(round(x_cen_bev))
                    y_cen_bev_int = int(round(y_cen_bev))
                    x_left_bev_int = int(round(x_left_bev))
                    y_left_bev_int = int(round(y_left_bev))
                    x_right_bev_int = int(round(x_right_bev))
                    y_right_bev_int = int(round(y_right_bev))


                    ### draw pnts
                    if (0 <= x_cen_bev_int) and (x_cen_bev_int < w_img_bev) and (0 <= y_cen_bev_int) and (y_cen_bev_int < h_img_bev):
                        cv2.circle(img_ipm_gray3, center=(x_cen_bev_int, y_cen_bev_int),     radius=2, color=(0, 128, 0),    thickness=-1)
                    #end

                    if (0 <= x_left_bev_int) and (x_left_bev_int < w_img_bev) and (0 <= y_left_bev_int) and (y_left_bev_int < h_img_bev):
                        cv2.circle(img_ipm_gray3, center=(x_left_bev_int, y_left_bev_int),   radius=2, color=(20, 100, 250), thickness=-1)
                    #end

                    if (0 <= x_right_bev_int) and (x_right_bev_int < w_img_bev) and (0 <= y_right_bev_int) and (y_right_bev_int < h_img_bev):
                        cv2.circle(img_ipm_gray3, center=(x_right_bev_int, y_right_bev_int), radius=2, color=(250, 250, 0),  thickness=-1)
                    #end

                #end

            #end

            # cv2.imshow('extracted_triplet_ipm', img_ipm_gray3)
            # cv2.waitKey(1)


            ### save (temp)
            fname_output_temp = "/home/yu1/Desktop/dir_temp/temp_res0/triplet_ipm/" + "triplet_ipm_" + str(self.m_temp_idx_res_img) + '.jpg'
            cv2.imwrite(fname_output_temp, img_ipm_gray3)
            self.m_temp_idx_res_img += 1

        #end


        return list_dict_local_max
    #end



    ###############################################################################################################
    ###
    ###############################################################################################################
    def compute_centerness_from_leftright(self, res_left, res_right):
        """
        compute centerness from left/right rail estimation

        :param res_left:
        :param res_right:
        :return: res_weight, img_res_weight
        """

        ###=========================================================================================================
        ### show left rails - right rails
        ###
        ### res_left:       0.0 ~ X.X, float32
        ### res_right:      0.0 ~ X.X, float32
        ###=========================================================================================================


        ###================================================================================================
        ###
        ###================================================================================================
        res_delta = abs(res_left - res_right)
        res_sum   = abs(res_left) + abs(res_right)


        res_ratio = res_delta/res_sum           # close to 0: high centerness

        res_ratio[np.isnan(res_ratio)] = 1.0
        res_ratio[res_sum <= 1.0] = 1.0

        res_weight = 1.0 - res_ratio


        res_weight_b   = res_weight*255.0
        img_res_weight = res_weight_b.astype(np.uint8)


        return res_weight, img_res_weight
    #end

#END

########################################################################################################################
########################################################################################################################

