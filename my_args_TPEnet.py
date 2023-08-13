# 2020/8/11
# Jungwon Kang


import pickle
import numpy as np
import argparse



########################################################################################################################
###
########################################################################################################################
def str2bool(argument):
    if argument.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif argument.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Wrong argument in argparse, should be a boolean')
    #end
#END


########################################################################################################################
### define parameters for operation (outside PE_TPEnet)
########################################################################################################################
def define_args_operation(DATASET_for_use):
    parser = argparse.ArgumentParser(description="Params_operation")

    if DATASET_for_use == 0:
        ###===================================================================================================
        ### For YDHR
        ###===================================================================================================
        parser.add_argument('--size_img_process', type=int, default={'h': 540, 'w': 960}, help='image size used in a process')
        parser.add_argument('--dir_input', type=str, default="./sample_input_imgs/ydhr", help='directory for input images')
        parser.add_argument('--dir_output', type=str, default="./res_imgs", help='directory for output images')
        parser.add_argument('--b_save_res_imgs_as_file', type=str2bool, nargs='?', const=True, default=False, help='Save res imgs as file?')
    elif DATASET_for_use == 1:
        ###===================================================================================================
        ### For NYC
        ###===================================================================================================
        parser.add_argument('--size_img_process', type=int, default={'h': 540, 'w': 960}, help='image size used in a process')
        parser.add_argument('--dir_input', type=str, default="./sample_input_imgs/nyc", help='directory for input images')
        parser.add_argument('--dir_output', type=str, default="./res_imgs", help='directory for output images')
        parser.add_argument('--b_save_res_imgs_as_file', type=str2bool, nargs='?', const=True, default=False, help='Save res imgs as file?')
    else:
        ###===================================================================================================
        ### For OTHER
        ###===================================================================================================
        parser.add_argument('--size_img_process', type=int, default={'h': 540, 'w': 960}, help='image size used in a process')
        parser.add_argument('--dir_input', type=str, default="./sample_input_imgs/Metrolinx_2", help='directory for input images')
        parser.add_argument('--dir_output', type=str, default="./res_imgs", help='directory for output images')
        parser.add_argument('--b_save_res_imgs_as_file', type=str2bool, nargs='?', const=True, default=True, help='Save res imgs as file?')
    #end

    return parser
#END

#parser.add_argument('--dir_input', type=str, default="/media/yu1/hdd_my/Dataset_YDHR_OCT009_OCT10/img_gopro/selected/test7_run2_normal_reverse_switch_in", help='directory for input images')
#parser.add_argument('--dir_input', type=str, default="/home/yu1/proj_avin/dataset/img_nyc_1280_720", help='directory for input images')


########################################################################################################################
### define parameters for algorithm (inside PE_TPEnet)
########################################################################################################################
def define_args_algorithm(DATASET_for_use):
    parser = argparse.ArgumentParser(description="Params_algorithm")


    if DATASET_for_use == 0:
        ###===================================================================================================
        ### For YDHR
        ###===================================================================================================

        ### basic
        parser.add_argument('--b_create_imgs_res_interim', type=str2bool, nargs='?', const=True, default=False, help='create interim res images')

        ### network weight
        # TPEnet_a_weights_20800
        # Mybest_7000_30000_only_cen
        # Mybest_7000_39000_2_only_cen_noseg          Mybest_7000_83700_2_only_cen
        parser.add_argument('--file_weight', type=str, default="./net_weight/Mybest_219gooz01.pkl", help='location of network weight file')

        ### params for detection of local maxima triplets
        parser.add_argument('--param_triplet_nms_alpha', type=float, default=235.0/270.0, help='param (alpha) for detection of local maxima triplets')
        parser.add_argument('--param_triplet_nms_beta', type=float, default=-220.0, help='param (beta) for detection of local maxima triplets')
        parser.add_argument('--param_triplet_nms_min', type=float, default=15.0, help='param (min) for detection of local maxima triplets')
        parser.add_argument('--param_triplet_nms_scale', type=float, default=0.1, help='param (scale) for detection of local maxima triplets')

        ### params for inverse perspective mapping (for 3d data) -> would be re-set by set_value_for_args_algorithm()
        parser.add_argument('--param_3D_ipm_camera_intrinsic_matrix', type=float, default=np.zeros(shape=(3,3)), help='param for 3D data - camera intrinsic matrix')
        parser.add_argument('--param_3D_ipm_camera_pitch_angle', type=float, default=-90.0, help='param for 3D data - camera pitch angle (deg)')
        parser.add_argument('--param_3D_ipm_camera_pos_wrt_ground_plane', type=float, default=np.array([0.0, 0.0, 1.5]), help='param for 3D data - camera position wrt ground plane (meter)')
        parser.add_argument('--param_3D_ipm_img_pixel_per_meter', type=float, default=20.0, help='pixel per meter in ipm-img')
        parser.add_argument('--param_3D_ipm_img_height', type=int, default=1000, help='height of ipm-img')
        parser.add_argument('--param_3D_ipm_img_width', type=int, default=400, help='width of ipm-img')

        ### params for rail path graph
        parser.add_argument('--param_rpg_subedge_thres_dx_3d', type=float, default=10, help='(meter)')
        parser.add_argument('--param_rpg_subedge_thres_dy_img', type=int, default=10, help='(pixel)')
        parser.add_argument('--param_rpg_subedge_height_section', type=int, default=5, help='(pixel)')

        parser.add_argument('--param_rpg_nodeedge_thres_dist_img_for_seed', type=int, default=296, help='(pixel)')
        parser.add_argument('--param_rpg_nodeedge_thres_dx_3d', type=float, default=25, help='(meter)')
        parser.add_argument('--param_rpg_nodeedge_thres_dy_img', type=int, default=20, help='(pixel)')

        parser.add_argument('--param_rpg_path_vertices_valid_y_min', type=float, default=100.0, help='(meter)')

        parser.add_argument('--param_rpg_poly_fitting_y_max', type=float, default=150.0, help='(meter)')
        parser.add_argument('--param_rpg_poly_fitting_degree', type=int, default=2, help='degree of polynomial')

    else:
        ###===================================================================================================
        ### For NYC
        ###===================================================================================================

        ### basic
        parser.add_argument('--b_create_imgs_res_interim', type=str2bool, nargs='?', const=True, default=False,
                            help='create interim res images')

        ### network weight
        parser.add_argument('--file_weight', type=str,
                            default="./net_weight/TPEnet_a_weights_20800.pkl",
                            help='location of network weight file')

        ### params for detection of local maxima triplets
        parser.add_argument('--param_triplet_nms_alpha', type=float, default=145.0 / 270.0,
                            help='param (alpha) for detection of local maxima triplets')
        parser.add_argument('--param_triplet_nms_beta', type=float, default=-140.0,
                            help='param (beta) for detection of local maxima triplets')
        parser.add_argument('--param_triplet_nms_min', type=float, default=5.0,
                            help='param (min) for detection of local maxima triplets')
        parser.add_argument('--param_triplet_nms_scale', type=float, default=0.5,
                            help='param (scale) for detection of local maxima triplets')

        ### params for inverse perspective mapping (for 3d data) -> would be re-set by set_value_for_args_algorithm()
        parser.add_argument('--param_3D_ipm_camera_intrinsic_matrix', type=float, default=np.zeros(shape=(3, 3)),
                            help='param for 3D data - camera intrinsic matrix')
        parser.add_argument('--param_3D_ipm_camera_pitch_angle', type=float, default=-90.0,
                            help='param for 3D data - camera pitch angle (deg)')
        parser.add_argument('--param_3D_ipm_camera_pos_wrt_ground_plane', type=float, default=np.array([0.0, 0.0, 2.0]),
                            help='param for 3D data - camera position wrt ground plane (meter)')
        parser.add_argument('--param_3D_ipm_img_pixel_per_meter', type=float, default=20.0,
                            help='pixel per meter in ipm-img')
        parser.add_argument('--param_3D_ipm_img_height', type=int, default=1000, help='height of ipm-img')
        parser.add_argument('--param_3D_ipm_img_width', type=int, default=400, help='width of ipm-img')

        ### params for rail path graph
        parser.add_argument('--param_rpg_subedge_thres_dx_3d', type=float, default=0.5, help='(meter)')
        parser.add_argument('--param_rpg_subedge_thres_dy_img', type=int, default=10, help='(pixel)')
        parser.add_argument('--param_rpg_subedge_height_section', type=int, default=5, help='(pixel)')

        parser.add_argument('--param_rpg_nodeedge_thres_dist_img_for_seed', type=int, default=96, help='(pixel)')
        parser.add_argument('--param_rpg_nodeedge_thres_dx_3d', type=float, default=1.0, help='(meter)')
        parser.add_argument('--param_rpg_nodeedge_thres_dy_img', type=int, default=20, help='(pixel)')

        parser.add_argument('--param_rpg_path_vertices_valid_y_min', type=float, default=10.0, help='(meter)')

        parser.add_argument('--param_rpg_poly_fitting_y_max', type=float, default=150.0, help='(meter)')
        parser.add_argument('--param_rpg_poly_fitting_degree', type=int, default=2, help='degree of polynomial')

    #end


    return parser
#END


########################################################################################################################
### define parameters for algorithm (inside PE_TPEnet)
########################################################################################################################
def set_value_for_args_algorithm(DATASET_for_use, args):
    """
    In this function, we set values for the following args:
        args.param_3D_ipm_camera_intrinsic_matrix
        args.param_3D_ipm_camera_pitch_angle
        args.param_3D_ipm_camera_pos_wrt_ground_plane
        args.param_3D_ipm_img_pixel_per_meter
        args.param_3D_ipm_img_height
        args.param_3D_ipm_img_width

    :param DATASET_for_use:
    :param args:
    :return:
    """

    if DATASET_for_use == 0:
        ###===================================================================================================
        ### For YDHR
        ###===================================================================================================

        ###---------------------------------------------------------------------------------------------------
        ### read camera intrinsic matrix from file
        ###---------------------------------------------------------------------------------------------------
        file_camera_calib = "./camera_calib/params_cam_calib_960_540.pickle"
            # The file './camera_calib/params_cam_calib_960_540.pickle' is
            #   obtained using gopro images used in YDHR Oct 9&10 data collection.

        dict_temp = None

        with open(file_camera_calib, 'rb') as fh:
            dict_temp = pickle.load(fh)
        # end

        param_mat_K = dict_temp['k_mat']  # (3x3) ndarr

        ###---------------------------------------------------------------------------------------------------
        ### set value for args
        ###---------------------------------------------------------------------------------------------------
        args.param_3D_ipm_camera_intrinsic_matrix = param_mat_K
        args.param_3D_ipm_camera_pitch_angle = -90.0 - 3.5
        args.param_3D_ipm_camera_pos_wrt_ground_plane = np.array([0.0, 0.0, 1.5])
        args.param_3D_ipm_img_pixel_per_meter = 20.0
        args.param_3D_ipm_img_height = 1000
        args.param_3D_ipm_img_width = 400

    else:
        ###===================================================================================================
        ### For NYC
        ###===================================================================================================

        ###---------------------------------------------------------------------------------------------------
        ### read camera intrinsic matrix from file (temp)
        ###---------------------------------------------------------------------------------------------------
        file_camera_calib = "./camera_calib/params_cam_calib_960_540.pickle"
            # As we don't have camera intrinsic matrix for NYC images,
            #   here we just temporarily used the file './camera_calib/params_cam_calib_960_540.pickle'
            #   as a temporary camera intrinsic matrix.

        dict_temp = None

        with open(file_camera_calib, 'rb') as fh:
            dict_temp = pickle.load(fh)
        # end

        param_mat_K = dict_temp['k_mat']  # (3x3) ndarr

        ### temp
        param_mat_K[0, 0] = 500.0
        param_mat_K[1, 1] = 500.0


        ###---------------------------------------------------------------------------------------------------
        ### set value for args
        ###---------------------------------------------------------------------------------------------------
        args.param_3D_ipm_camera_intrinsic_matrix = param_mat_K
        args.param_3D_ipm_camera_pitch_angle = -90.0
        args.param_3D_ipm_camera_pos_wrt_ground_plane = np.array([0.0, 0.0, 1.75])
        args.param_3D_ipm_img_pixel_per_meter = 20.0
        args.param_3D_ipm_img_height = 1000
        args.param_3D_ipm_img_width = 400
    #end

    return args
#END
