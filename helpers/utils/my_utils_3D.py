# 2020/09/02
# Jungwon Kang


import pickle
import numpy as np
import cv2

from .inverse_perspective_mapping    import my_IPM


########################################################################################################################
###
########################################################################################################################
class MyUtils_3D:

    m_obj_my_ipm = None
    m_mat_tf_from_img_ori_to_bev = None
    m_mat_tf_from_img_ori_to_world = None
    m_mat_tf_from_world_to_img_bev = None
    m_mat_tf_from_world_to_img_ori = None
    m_param_h_img_bev = None
    m_param_w_img_bev = None


    ###============================================================================================================
    ### MyUtils_3D::__init__()
    ###============================================================================================================
    def __init__(self, dict_args):
        """
        :param mode:
        """

        self.m_obj_my_ipm, \
        self.m_mat_tf_from_img_ori_to_bev, \
        self.m_mat_tf_from_img_ori_to_world, \
        self.m_mat_tf_from_world_to_img_bev, \
        self.m_mat_tf_from_world_to_img_ori, \
        self.m_param_h_img_bev, \
        self.m_param_w_img_bev = self._init_obj_IPM(dict_args)

    #end


    ###============================================================================================================
    ### MyUtils_3D::_init_obj_IPM_YDHR()
    ###============================================================================================================
    def _init_obj_IPM(self, dict_args):

        ###---------------------------------------------------------------------------------------------
        ### set params
        ###---------------------------------------------------------------------------------------------

        ### set params
        param_mat_K                 = dict_args["param_3D_ipm_camera_intrinsic_matrix"]
        theta_rotx_deg              = dict_args["param_3D_ipm_camera_pitch_angle"]
        vec_trans_cam_in_world      = dict_args["param_3D_ipm_camera_pos_wrt_ground_plane"]

        param_scale_pixel_per_meter = dict_args["param_3D_ipm_img_pixel_per_meter"]
        param_h_img_bev             = dict_args["param_3D_ipm_img_height"]
        param_w_img_bev             = dict_args["param_3D_ipm_img_width"]

        param_offset_y              = param_h_img_bev - 1
        param_offset_x              = param_w_img_bev / 2.0


        ###---------------------------------------------------------------------------------------------
        ### create
        ###---------------------------------------------------------------------------------------------
        obj_my_ipm = my_IPM.MyIPM(mat_K=param_mat_K,
                                   param_scale_pixel_per_meter=param_scale_pixel_per_meter,
                                   param_h_img_bev=param_h_img_bev,
                                   param_w_img_bev=param_w_img_bev,
                                   param_offset_y=param_offset_y,
                                   param_offset_x=param_offset_x,
                                   theta_rotx_deg=theta_rotx_deg,
                                   vec_trans_cam_in_world=vec_trans_cam_in_world)


        ###---------------------------------------------------------------------------------------------
        ### get
        ###---------------------------------------------------------------------------------------------
        mat_tf_from_img_ori_to_bev = obj_my_ipm.get_transform_from_img_ori_to_bev()


        ###---------------------------------------------------------------------------------------------
        ### get
        ###---------------------------------------------------------------------------------------------
        mat_tf_from_world_to_img_ori = obj_my_ipm.get_transform_from_world_to_img_ori()
        mat_tf_from_img_ori_to_world = np.linalg.inv(mat_tf_from_world_to_img_ori)
            # completed to set
            #       mat_tf_from_img_ori_to_world: (3,3) matrix


        ###---------------------------------------------------------------------------------------------
        ### get
        ###---------------------------------------------------------------------------------------------
        mat_tf_from_world_to_img_bev = obj_my_ipm.get_transform_from_world_to_img_bev()


        return obj_my_ipm, \
               mat_tf_from_img_ori_to_bev, \
               mat_tf_from_img_ori_to_world, \
               mat_tf_from_world_to_img_bev, \
               mat_tf_from_world_to_img_ori, \
               param_h_img_bev, \
               param_w_img_bev
    #end


    ###============================================================================================================
    ### MyUtils_3D::get_size_img_bev()
    ###============================================================================================================
    def get_size_img_bev(self):
        return self.m_param_h_img_bev, self.m_param_w_img_bev


    ###============================================================================================================
    ### MyUtils_3D::create_img_IPM()
    ###============================================================================================================
    def create_img_IPM(self, img_in):
        ###
        img_ipm_out = self.m_obj_my_ipm.process(img_in)

        ###
        if 0:
            cv2.imshow("img_ipm_out", img_ipm_out)
            cv2.waitKey(0)
        #end

        return img_ipm_out
    #end


    ###============================================================================================================
    ### MyUtils_3D::convert_pnt_img_ori_to_pnt_world()
    ###============================================================================================================
    def convert_pnt_img_ori_to_pnt_world(self, pnt_img_ori):
        # pnt_img_ori: (3,1)

        pnt_world_ = self.m_mat_tf_from_img_ori_to_world @ pnt_img_ori
        pnt_world = pnt_world_ / pnt_world_[2, 0]

        x_world = float(pnt_world[0, 0])
        y_world = float(pnt_world[1, 0])
        z_world = 0.0

        return x_world, y_world, z_world
    #end


    ###============================================================================================================
    ### MyUtils_3D::convert_pnt_img_ori_to_pnt_bev()
    ###============================================================================================================
    def convert_pnt_img_ori_to_pnt_bev(self, pnt_img_ori):
        # pnt_img_ori: (3,1)

        pnt_bev_ = self.m_mat_tf_from_img_ori_to_bev @ pnt_img_ori
        pnt_bev = pnt_bev_ / pnt_bev_[2, 0]

        x_bev = pnt_bev[0, 0]
        y_bev = pnt_bev[1, 0]

        return x_bev, y_bev     # pnt on img_bev
    #end


    ###============================================================================================================
    ### MyUtils_3D::convert_pnt_world_to_pnt_bev()
    ###============================================================================================================
    def convert_pnt_world_to_pnt_bev(self, pnt_world):
        # pnt_world: (3,1)

        pnt_bev = self.m_mat_tf_from_world_to_img_bev @ pnt_world

        x_bev = pnt_bev[0, 0]
        y_bev = pnt_bev[1, 0]

        return x_bev, y_bev     # pnt on img_bev
    #end


    ###============================================================================================================
    ### MyUtils_3D::convert_pnt_world_to_pnt_img_ori()
    ###============================================================================================================
    def convert_pnt_world_to_pnt_img_ori(self, pnt_world):
        # pnt_world: (3,1) -> [X_world, Y_world, 1]

        pnt_img_ori = self.m_mat_tf_from_world_to_img_ori @ pnt_world

        x_img_ori = pnt_img_ori[0, 0] / pnt_img_ori[2, 0]
        y_img_ori = pnt_img_ori[1, 0] / pnt_img_ori[2, 0]

        return x_img_ori, y_img_ori     # pnt on img_ori
    #end

#END



########################################################################################################################
### __main__()
########################################################################################################################
# if __name__ == "__main__":
#
#     ###
#     obj_my_utils_3D = MyUtils_3D()
#
#     a = 1



    ###============================================================================================================
    ### MyUtils_3D::_init_obj_IPM_YDHR()
    ###============================================================================================================
    # def _init_obj_IPM_YDHR(self):
    #
    #     ###---------------------------------------------------------------------------------------------
    #     ### set params
    #     ###---------------------------------------------------------------------------------------------
    #
    #     ### set params (k_mat)
    #     file_cam_calib_params_input = '/media/yu1/hdd_my/Dataset_YDHR_OCT009_OCT10/img_gopro/img_for_calib/img_calib_res_params_960_540/params_cam_calib_960_540.pickle'
    #
    #     with open(file_cam_calib_params_input, 'rb') as fh:
    #         dict_temp = pickle.load(fh)
    #     # end
    #
    #     param_mat_K = dict_temp['k_mat']
    #     #cam_coeff_distort = dict_temp['coeff_distort']
    #
    #
    #     ### set params
    #     param_scale_pixel_per_meter = 20.0
    #     param_h_img_bev = 1000
    #     param_w_img_bev = 400
    #     param_offset_y = param_h_img_bev - 1
    #     param_offset_x = param_w_img_bev / 2.0
    #
    #     theta_rotx_deg = -90.0 - 3.5
    #     vec_trans_cam_in_world = np.array([0.0, 0.0, 1.5])
    #
    #
    #     ###---------------------------------------------------------------------------------------------
    #     ### create
    #     ###---------------------------------------------------------------------------------------------
    #     obj_my_ipm = my_IPM.MyIPM(mat_K=param_mat_K,
    #                                param_scale_pixel_per_meter=param_scale_pixel_per_meter,
    #                                param_h_img_bev=param_h_img_bev,
    #                                param_w_img_bev=param_w_img_bev,
    #                                param_offset_y=param_offset_y,
    #                                param_offset_x=param_offset_x,
    #                                theta_rotx_deg=theta_rotx_deg,
    #                                vec_trans_cam_in_world=vec_trans_cam_in_world)
    #
    #
    #     ###---------------------------------------------------------------------------------------------
    #     ### get
    #     ###---------------------------------------------------------------------------------------------
    #     mat_tf_from_img_ori_to_bev = obj_my_ipm.get_transform_from_img_ori_to_bev()
    #
    #
    #     ###---------------------------------------------------------------------------------------------
    #     ### get
    #     ###---------------------------------------------------------------------------------------------
    #     mat_tf_from_world_to_img_ori = obj_my_ipm.get_transform_from_world_to_img_ori()
    #     mat_tf_from_img_ori_to_world = np.linalg.inv(mat_tf_from_world_to_img_ori)
    #         # completed to set
    #         #       mat_tf_from_img_ori_to_world: (3,3) matrix
    #
    #
    #     ###---------------------------------------------------------------------------------------------
    #     ### get
    #     ###---------------------------------------------------------------------------------------------
    #     mat_tf_from_world_to_img_bev = obj_my_ipm.get_transform_from_world_to_img_bev()
    #
    #
    #     return obj_my_ipm, \
    #            mat_tf_from_img_ori_to_bev, \
    #            mat_tf_from_img_ori_to_world, \
    #            mat_tf_from_world_to_img_bev, \
    #            param_h_img_bev, \
    #            param_w_img_bev
    # #end


