# 2021/7/20
# 2021/8/24
# Jungwon Kang


import cv2
import numpy as np
import copy


########################################################################################################################
###
########################################################################################################################
class MyHelper_GT:

    ###============================================================================================================
    ###
    ###============================================================================================================
    def __init__(self, title_testrun_this, w_img, param_dx_valid_a, param_dx_valid_b):

        self.m_title_testrun_this = title_testrun_this

        self.m_x_valid_lower = (0.5*w_img) + param_dx_valid_a
        self.m_x_valid_upper = (0.5*w_img) + param_dx_valid_b


        ### set bgr
        #self.m_val_bgr_rail_L = (0, 0, 255)     # bgr
        #self.m_val_bgr_rail_R = (0, 220, 0)

        self.m_val_bgr_rail_L = (20, 100, 250)     # bgr
        self.m_val_bgr_rail_R = (250, 200, 0)


        return


    ###============================================================================================================
    ###
    ###============================================================================================================
    def get_gt_final(self, gt_raw_dict_xs_img_rail_LR, gt_raw_dict_XYZ_pnt_in_cam_rail_L, gt_raw_dict_XYZ_pnt_in_cam_rail_R):
        # gt_raw_dict_rail_pnt_x_img: {dict: n}, where a dict is ndarr (540,2)
        # gt_raw_dict_xyz_pnt_rail_left_in_cam: {dict: n}, where a dict is {list: 540}
        # gt_raw_dict_xyz_pnt_rail_right_in_cam: {dict: n}, where a dict is {list: 540}
        # title_testrun_this: 'TEST4_RUN2_GOPRO' or 'TEST7_RUN2_NRS_GOPRO'


        ###
        gt_final_dict_xs_img_rail_LR = {}
        gt_final_dict_XYZ_pnt_in_cam_rail_L = {}
        gt_final_dict_XYZ_pnt_in_cam_rail_R = {}


        ### filter out unnecessary rails, get only valid ones (candidate ego-paths)
        for key_this in gt_raw_dict_xs_img_rail_LR:      # key_this: 'ADErail1', 'ADErail2', 'ADErail3'

            ### check 1
            if self.m_title_testrun_this == 'TEST7_RUN2_NRS_GOPRO':
                if key_this == 'ADErail3':
                    continue

            ### get point pair for one rail
            arr_xs_img_rail_LR = gt_raw_dict_xs_img_rail_LR[key_this]  # (h_img, 2), point pair (i.e, L,R)
            h_img = arr_xs_img_rail_LR.shape[0]
                # completed to set
                #       gt_rail_pnt_x_img: (h_img, 2), point pair


            ### get the point pair at the bottom
            pnt_pair_bottom = arr_xs_img_rail_LR[(h_img - 1), :]    # (2,)   -> (L,R)

            x_L = pnt_pair_bottom[0]
            x_R = pnt_pair_bottom[1]


            ### check 2: if bottom is not a paired one, then just pass.
            if (x_L < 0) or (x_R < 0):
                continue


            ### check 3: if bottom is off from the center, then just pass
            x_cen = (x_L + x_R) / 2.0

            if (x_cen < self.m_x_valid_lower) or (x_cen > self.m_x_valid_upper):
                continue


            ### note that only valid key_this researches here.
            gt_final_dict_xs_img_rail_LR[key_this]        = gt_raw_dict_xs_img_rail_LR[key_this]
            gt_final_dict_XYZ_pnt_in_cam_rail_L[key_this] = gt_raw_dict_XYZ_pnt_in_cam_rail_L[key_this]
            gt_final_dict_XYZ_pnt_in_cam_rail_R[key_this] = gt_raw_dict_XYZ_pnt_in_cam_rail_R[key_this]
        #end

        return gt_final_dict_xs_img_rail_LR, gt_final_dict_XYZ_pnt_in_cam_rail_L, gt_final_dict_XYZ_pnt_in_cam_rail_R
            # gt_final_dict_xs_img_rail_LR:        {dict: m}, where a dict is ndarr (540,2)
            # gt_final_dict_XYZ_pnt_in_cam_rail_L: {dict: m}, where a dict is {list: 540}
            # gt_final_dict_XYZ_pnt_in_cam_rail_R: {dict: m}, where a dict is {list: 540}


    ###============================================================================================================
    ###
    ###============================================================================================================
    def visualize_track_debug(self, img_in, idx_step, gt_final_dict_xs_img_rail_LR):

        totnum_tracks = len(gt_final_dict_xs_img_rail_LR)
        img_show = copy.deepcopy(img_in)


        for key_this in gt_final_dict_xs_img_rail_LR:       # key_this: 'ADErail1', 'ADErail2', 'ADErail3'

            ### get point pair for one rail
            arr_xs_img_rail_LR = gt_final_dict_xs_img_rail_LR[key_this]  # (h_img, 2), point pair
            h_img = arr_xs_img_rail_LR.shape[0]
                # completed to set
                #       arr_xs_img_rail_LR: (h_img, 2), point pair

            ###
            for y_this in range(h_img):
                xs_pair = arr_xs_img_rail_LR[y_this, :]

                x_L = xs_pair[0]
                x_R = xs_pair[1]

                if (x_L < 0) or (x_R < 0):
                    continue

                img_show[y_this, x_L, :] = self.m_val_bgr_rail_L
                img_show[y_this, x_R, :] = self.m_val_bgr_rail_R
            #end
        #end


        ### put text
        font = cv2.FONT_HERSHEY_SIMPLEX
        val_bgr_info = (0, 0, 0)
        str_info_a = 'Frame: %d' % idx_step
        cv2.putText(img_show, str_info_a, (20, 30), font, 0.7, val_bgr_info, 1, cv2.LINE_AA)

        str_info_b = 'Number of tracks: %d' % totnum_tracks
        cv2.putText(img_show, str_info_b, (20, 55), font, 0.7, val_bgr_info, 1, cv2.LINE_AA)


        ### show
        cv2.imshow('img_track_debug', img_show)
        # cv2.imwrite(full_fname_img_in, img_in)
        cv2.waitKey(1)

        return
    #END


    ###============================================================================================================
    ###
    ###============================================================================================================
    def visualize_track_release(self, img_in, idx_step, gt_final_dict_xs_img_rail_LR):

        totnum_tracks = len(gt_final_dict_xs_img_rail_LR)
        img_show = copy.deepcopy(img_in)


        for key_this in gt_final_dict_xs_img_rail_LR:       # key_this: 'ADErail1', 'ADErail2', 'ADErail3'

            ### get point pair for one rail
            arr_xs_img_rail_LR = gt_final_dict_xs_img_rail_LR[key_this]  # (h_img, 2), point pair
            h_img = arr_xs_img_rail_LR.shape[0]
            # completed to set
            #       arr_xs_img_rail_LR: (h_img, 2), point pair

            ###
            for y_this in range(h_img):
                xs_pair = arr_xs_img_rail_LR[y_this, :]

                x_L = xs_pair[0]
                x_R = xs_pair[1]

                if (x_L < 0) or (x_R < 0):
                    continue


                ### track region
                for x_this in range(x_L, (x_R + 1)):
                    b_old_uint8, g_old_uint8, r_old_uint8 = img_show[y_this, x_this, :]
                    b_new_uint8, g_new_uint8, r_new_uint8 = self.adjust_rgb_track_region(b_old_uint8, g_old_uint8, r_old_uint8)
                    img_show[y_this, x_this, :] = [b_new_uint8, g_new_uint8, r_new_uint8]
                #end

                ### track
                cv2.circle(img_show, center=(x_L, y_this), radius=2, color=self.m_val_bgr_rail_L, thickness=-1)
                cv2.circle(img_show, center=(x_R, y_this), radius=2, color=self.m_val_bgr_rail_R, thickness=-1)
            #end
        #end


        ### put text
        font = cv2.FONT_HERSHEY_SIMPLEX
        val_bgr_info = (0, 0, 0)
        str_info_a = 'Frame: %d' % idx_step
        cv2.putText(img_show, str_info_a, (20, 30), font, 0.7, val_bgr_info, 1, cv2.LINE_AA)

        str_info_b = 'Number of tracks: %d' % totnum_tracks
        cv2.putText(img_show, str_info_b, (20, 55), font, 0.7, val_bgr_info, 1, cv2.LINE_AA)


        ### show
        cv2.imshow('img_track_release', img_show)
        # fname_output_temp = "/home/yu1/Desktop/dir_temp/temp_res0/final_path_on_ori/" + "final_path_ori_" + str(self.m_temp_idx_res_img_on_ori) + '.jpg'
        # cv2.imwrite(fname_output_temp, img_vis_ori_gray3)

        cv2.waitKey(1)



        return
    #END



    ###============================================================================================================
    ###
    ###============================================================================================================
    def visualize_dist_debug(self, img_in, idx_step, gt_final_dict_xs_img_rail_LR, gt_final_dict_XYZ_pnt_in_cam_rail_L, gt_final_dict_XYZ_pnt_in_cam_rail_R):
        # gt_final_dict_XYZ_pnt_in_cam_rail_L: {dict: n}, where a dict is {list: h_img}
        # gt_final_dict_XYZ_pnt_in_cam_rail_R: {dict: n}, where a dict is {list: h_img}

        totnum_tracks = len(gt_final_dict_xs_img_rail_LR)
        img_show = copy.deepcopy(img_in)


        for key_this in gt_final_dict_xs_img_rail_LR:       # key_this: 'ADErail1', 'ADErail2', 'ADErail3'

            ### get point pair for one rail
            arr_xs_img_rail_LR = gt_final_dict_xs_img_rail_LR[key_this]             # (h_img, 2), point pair
            dict_XYZ_rail_L    = gt_final_dict_XYZ_pnt_in_cam_rail_L[key_this]      # {list: h_img}
            dict_XYZ_rail_R    = gt_final_dict_XYZ_pnt_in_cam_rail_R[key_this]      # {list: h_img}

            h_img = arr_xs_img_rail_LR.shape[0]
                # completed to set
                #       arr_xs_img_rail_LR: (h_img, 2), point pair

            ###
            for y_this in range(h_img):
                xs_pair    = arr_xs_img_rail_LR[y_this, :]
                XYZ_rail_L = dict_XYZ_rail_L[y_this]
                XYZ_rail_R = dict_XYZ_rail_R[y_this]

                ###
                x_L = xs_pair[0]
                x_R = xs_pair[1]

                if (x_L < 0) or (x_R < 0):
                    continue

                if len(XYZ_rail_L) == 0 or len(XYZ_rail_R) == 0:
                    continue

                Z_L = XYZ_rail_L[2]
                Z_R = XYZ_rail_R[2]


                ###
                mag_Z_L = Z_L/50.0
                mag_Z_L = min(mag_Z_L, 1.0)
                mag_Z_L = max(mag_Z_L, 0.0)
                #mag_Z_L = np.uint8(mag_Z_L*255.0)

                mag_Z_R = Z_R/50.0
                mag_Z_R = min(mag_Z_R, 1.0)
                mag_Z_R = max(mag_Z_R, 0.0)
                #mag_Z_R = np.uint8(mag_Z_R*255.0)


                #img_show[y_this, x_L, :] = (mag_Z_L, mag_Z_L, mag_Z_L)
                #img_show[y_this, x_R, :] = (mag_Z_R, mag_Z_R, mag_Z_R)

                mag_Z_L = int(mag_Z_L*255.0)
                mag_Z_R = int(mag_Z_R*255.0)

                cv2.circle(img_show, center=(x_L, y_this), radius=2, color=(mag_Z_L, mag_Z_L, mag_Z_L), thickness=-1)
                cv2.circle(img_show, center=(x_R, y_this), radius=2, color=(mag_Z_R, mag_Z_R, mag_Z_R), thickness=-1)
            #end
        #end


        ### put text
        font = cv2.FONT_HERSHEY_SIMPLEX
        val_bgr_info = (0, 0, 0)
        str_info_a = 'Frame: %d' % idx_step
        cv2.putText(img_show, str_info_a, (20, 30), font, 0.7, val_bgr_info, 1, cv2.LINE_AA)

        str_info_b = 'Number of tracks: %d' % totnum_tracks
        cv2.putText(img_show, str_info_b, (20, 55), font, 0.7, val_bgr_info, 1, cv2.LINE_AA)


        ### show
        cv2.imshow('img_dist_debug', img_show)
        # cv2.imwrite(full_fname_img_in, img_in)
        cv2.waitKey(1)

        return
    #END


    ###============================================================================================================
    ###
    ###============================================================================================================
    def visualize_dist_release(self, img_in, idx_step, gt_final_dict_xs_img_rail_LR, gt_final_dict_XYZ_pnt_in_cam_rail_L, gt_final_dict_XYZ_pnt_in_cam_rail_R, format_fname_img_res):
        # gt_final_dict_XYZ_pnt_in_cam_rail_L: {dict: n}, where a dict is {list: h_img}
        # gt_final_dict_XYZ_pnt_in_cam_rail_R: {dict: n}, where a dict is {list: h_img}

        totnum_tracks = len(gt_final_dict_xs_img_rail_LR)
        img_show = copy.deepcopy(img_in)


        ### draw track region
        for key_this in gt_final_dict_xs_img_rail_LR:       # key_this: 'ADErail1', 'ADErail2', 'ADErail3'

            ### get point pair for one rail
            arr_xs_img_rail_LR = gt_final_dict_xs_img_rail_LR[key_this]             # (h_img, 2), point pair
            dict_XYZ_rail_L    = gt_final_dict_XYZ_pnt_in_cam_rail_L[key_this]      # {list: h_img}
            dict_XYZ_rail_R    = gt_final_dict_XYZ_pnt_in_cam_rail_R[key_this]      # {list: h_img}

            h_img = arr_xs_img_rail_LR.shape[0]
                # completed to set
                #       arr_xs_img_rail_LR: (h_img, 2), point pair

            ###
            for y_this in range(h_img):
                xs_pair    = arr_xs_img_rail_LR[y_this, :]
                XYZ_rail_L = dict_XYZ_rail_L[y_this]
                XYZ_rail_R = dict_XYZ_rail_R[y_this]

                ###
                x_L = xs_pair[0]
                x_R = xs_pair[1]

                if (x_L < 0) or (x_R < 0):
                    continue

                if len(XYZ_rail_L) == 0 or len(XYZ_rail_R) == 0:
                    continue


                Z_L = XYZ_rail_L[2]
                Z_R = XYZ_rail_R[2]
                Z_avg = (Z_L + Z_R)/2.0

                ###
                mag_Z_avg = Z_avg/30.0
                mag_Z_avg = min(mag_Z_avg, 1.0)
                mag_Z_avg = max(mag_Z_avg, 0.0)
                mag_Z_avg = int(mag_Z_avg*255.0)

                ### track region
                for x_this in range(x_L, (x_R + 1)):
                    img_show[y_this, x_this, :] = [mag_Z_avg, mag_Z_avg, mag_Z_avg]
                #end
            #end
        #end


        ### draw tracks
        for key_this in gt_final_dict_xs_img_rail_LR:       # key_this: 'ADErail1', 'ADErail2', 'ADErail3'

            ### get point pair for one rail
            arr_xs_img_rail_LR = gt_final_dict_xs_img_rail_LR[key_this]             # (h_img, 2), point pair
            dict_XYZ_rail_L    = gt_final_dict_XYZ_pnt_in_cam_rail_L[key_this]      # {list: h_img}
            dict_XYZ_rail_R    = gt_final_dict_XYZ_pnt_in_cam_rail_R[key_this]      # {list: h_img}

            h_img = arr_xs_img_rail_LR.shape[0]
                # completed to set
                #       arr_xs_img_rail_LR: (h_img, 2), point pair

            ###
            for y_this in range(h_img):
                xs_pair    = arr_xs_img_rail_LR[y_this, :]
                XYZ_rail_L = dict_XYZ_rail_L[y_this]
                XYZ_rail_R = dict_XYZ_rail_R[y_this]

                ###
                x_L = xs_pair[0]
                x_R = xs_pair[1]

                if (x_L < 0) or (x_R < 0):
                    continue

                if len(XYZ_rail_L) == 0 or len(XYZ_rail_R) == 0:
                    continue

                ### track
                cv2.circle(img_show, center=(x_L, y_this), radius=2, color=self.m_val_bgr_rail_L, thickness=-1)
                cv2.circle(img_show, center=(x_R, y_this), radius=2, color=self.m_val_bgr_rail_R, thickness=-1)
            #end
        #end


        ### put text
        font = cv2.FONT_HERSHEY_SIMPLEX
        val_bgr_info = (0, 0, 0)
        str_info_a = 'Frame: %d' % idx_step
        cv2.putText(img_show, str_info_a, (20, 30), font, 0.7, val_bgr_info, 1, cv2.LINE_AA)

        str_info_b = 'Number of tracks: %d' % totnum_tracks
        cv2.putText(img_show, str_info_b, (20, 55), font, 0.7, val_bgr_info, 1, cv2.LINE_AA)


        ### show
        fullfname_temp = format_fname_img_res % idx_step
        #fullfname_temp = '/home/yu1/Desktop/dir_TPEnet_eval/res_img_gt_visualized/img_res_%d.png' % idx_step

        cv2.imshow('img_dist_debug', img_show)
        cv2.imwrite(fullfname_temp, img_show)
        cv2.waitKey(1)

        return
    #END


    ###============================================================================================================
    ###
    ###============================================================================================================
    def adjust_rgb_track_region(self, b_old_uint8, g_old_uint8, r_old_uint8):

        db_int = 0
        dg_int = 50
        dr_int = 0

        ###
        b_new_int = int(b_old_uint8) + db_int
        g_new_int = int(g_old_uint8) + dg_int
        r_new_int = int(r_old_uint8) + dr_int

        ###
        b_new_int = min(b_new_int, 255)
        b_new_int = max(b_new_int, 0)

        g_new_int = min(g_new_int, 255)
        g_new_int = max(g_new_int, 0)

        b_new_int = min(b_new_int, 255)
        b_new_int = max(b_new_int, 0)

        ###
        b_new_uint8 = np.uint8(b_new_int)
        g_new_uint8 = np.uint8(g_new_int)
        r_new_uint8 = np.uint8(r_new_int)

        return b_new_uint8, g_new_uint8, r_new_uint8



########################################################################################################################
########################################################################################################################


# ### fill img
# b_old_uint8, g_old_uint8, r_old_uint8 = img_in[y_img, x_img, :]
#
# # IF this is switch-region (supported by two rails and more)
# if arr_y_global_id_switchregion_this[y_img] > 0 and arr_cnt_valid_global_id_switchregion_all_rails[y_img] > 1:
#     b_new_uint8, g_new_uint8, r_new_uint8 = adjust_rgb_for_switchregion(b_old_uint8, g_old_uint8, r_old_uint8)
# else:
#     b_new_uint8, g_new_uint8, r_new_uint8 = adjust_rgb_for_region(b_old_uint8, g_old_uint8, r_old_uint8,
#                                                                   class_ADErail_this)
# # end
#
#
# ###
# img_out_rail_region[y_img, x_img, :] = [b_new_uint8, g_new_uint8, r_new_uint8]





