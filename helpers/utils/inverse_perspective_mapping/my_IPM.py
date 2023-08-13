# Aug 12 2020
# Jungwon Kang

import cv2
import math
import numpy as np


####################################################################################################################################
### class MyUtil
####################################################################################################################################
class MyUtil:
    ###=============================================================================================
    ### __init_()
    ###=============================================================================================
    def __init__(self):
        return
    #end


    ###=============================================================================================
    ### rotz()
    ###=============================================================================================
    def rotz(self, theta_in):
        # theta: rad
        # output: (3x3) matrix

        ct = math.cos(theta_in)
        st = math.sin(theta_in)

        mat = np.array([[ ct, -st, 0.0],
                              [ st,  ct, 0.0],
                              [0.0, 0.0, 1.0]])

        return mat
    #end


    ###=============================================================================================
    ### roty()
    ###=============================================================================================
    def roty(self, theta_in):
        # theta: rad
        # output: (3x3) matrix

        ct = math.cos(theta_in)
        st = math.sin(theta_in)

        mat = np.array([[ ct, 0.0,  st],
                              [0.0, 1.0, 0.0],
                              [-st, 0.0,  ct]])

        return mat
    #end


    ###=============================================================================================
    ### rotx()
    ###=============================================================================================
    def rotx(self, theta_in):
        # theta: rad
        # output: (3x3) matrix

        ct = math.cos(theta_in)
        st = math.sin(theta_in)

        mat = np.array([[1.0, 0.0, 0.0],
                              [0.0,  ct, -st],
                              [0.0,  st,  ct]])

        return mat
    #end


    ###=============================================================================================
    ### convert_rot_trans_to_hmat()
    ###=============================================================================================
    def convert_rot_trans_to_hmat(self, mat_rot, vec_trans):
        # convert rotation & translation into homogeneous matrix (4,4)
        #   mat_rot: (3,3)
        #   vec_trans: (3,)

        hmat = np.zeros(shape=(4,4))

        hmat[0,0] = mat_rot[0,0]
        hmat[0,1] = mat_rot[0,1]
        hmat[0,2] = mat_rot[0,2]
        hmat[0,3] = vec_trans[0]

        hmat[1,0] = mat_rot[1,0]
        hmat[1,1] = mat_rot[1,1]
        hmat[1,2] = mat_rot[1,2]
        hmat[1,3] = vec_trans[1]

        hmat[2,0] = mat_rot[2,0]
        hmat[2,1] = mat_rot[2,1]
        hmat[2,2] = mat_rot[2,2]
        hmat[2,3] = vec_trans[2]

        hmat[3,3] = 1.0

        return hmat



#end


####################################################################################################################################
### class MyIPM
####################################################################################################################################
class MyIPM:
    """
    Note that we are going through the following transformations:
        @ bev-img -> world
        @ world -> camera
        @ camera -> ori-img
    """

    obj_my_util = MyUtil()

    mat_tf_from_img_ori_to_img_bev = None


    ###=============================================================================================
    ### __init_()
    ###=============================================================================================
    def __init__(self, mat_K, param_scale_pixel_per_meter, param_h_img_bev, param_w_img_bev, param_offset_y, param_offset_x, theta_rotx_deg, vec_trans_cam_in_world):

        self.mat_K = mat_K              # 3x3 matrix
        self.param_scale_pixel_per_meter = param_scale_pixel_per_meter
        self.param_h_img_bev = param_h_img_bev
        self.param_w_img_bev = param_w_img_bev
        self.param_offset_y = param_offset_y
        self.param_offset_x = param_offset_x

        self.theta_rotx_deg = theta_rotx_deg
        self.vec_trans_cam_in_world = vec_trans_cam_in_world


        ###------------------------------------------------------------------------------------------------------------------------------
        ### build a transformation (bev-img -> ori-img)
        ###------------------------------------------------------------------------------------------------------------------------------
        mat_tf_from_img_bev_to_img_ori = self.get_transform_from_bev_to_img_ori()

        mat_tf_from_img_ori_to_img_bev_ = np.linalg.inv(mat_tf_from_img_bev_to_img_ori)
        self.mat_tf_from_img_ori_to_img_bev = mat_tf_from_img_ori_to_img_bev_ / mat_tf_from_img_ori_to_img_bev_[2, 2]
            # completed to set
            #       self.mat_tf_from_img_ori_to_img_bev

    #end


    ###=============================================================================================
    ### process()
    ###=============================================================================================
    def process(self, img_ori):

        if self.mat_tf_from_img_ori_to_img_bev is None:
            print("No transformation was set.")
            return
        #end


        img_bev = cv2.warpPerspective(img_ori, self.mat_tf_from_img_ori_to_img_bev, (self.param_w_img_bev, self.param_h_img_bev))

        return img_bev
    #end


    ###=============================================================================================
    ### transformation: world -> camera
    ###=============================================================================================
    def get_pose_cam_in_world(self):

        ### make hmat_cam_in_world
        theta_rotx_deg = self.theta_rotx_deg
        theta_rotx_rad = theta_rotx_deg*(math.pi / 180.0)

        mat_rot_cam_in_world = self.obj_my_util.rotx(theta_rotx_rad)    # -90 rot_x
        vec_trans_cam_in_world = self.vec_trans_cam_in_world

        hmat_cam_in_world = self.obj_my_util.convert_rot_trans_to_hmat(mat_rot_cam_in_world, vec_trans_cam_in_world)
            # hmat_cam_in_world: 4x4 matrix

        return hmat_cam_in_world
    #end


    ###=============================================================================================
    ### transformation: bev-img -> world
    ###=============================================================================================
    def get_transform_from_world_to_img_bev(self):
        s    = self.param_scale_pixel_per_meter
        offx = self.param_offset_x
        offy = self.param_offset_y

        mat_transform = np.array([[s, 0.0, offx], [0.0, -s, offy], [0.0, 0.0, 1.0]])

        return mat_transform
    #end


    ###=============================================================================================
    ### [final] transformation: bev-img <- world <- camera <- ori-img
    ###=============================================================================================
    def get_transform_from_img_ori_to_bev(self):
        return self.mat_tf_from_img_ori_to_img_bev
    #end


    ###=============================================================================================
    ### [final] transformation: bev-img -> world -> camera -> ori-img
    ###=============================================================================================
    def get_transform_from_bev_to_img_ori(self):
        """
        @ bev-img -> world: by get_transform_from_world_to_img_bev()
        @ world -> camera:  by get_pose_cam_in_world()
        @ camera -> ori-img: by k
        """

        ### get transform from cam to img_ori
        mat_tf_from_cam_to_img_ori = self.mat_K
            # completed to set
            #       mat_tf_from_cam_to_img_ori: 3x3 matrix


        ### get transform from world to cam
        hmat_cam_in_world = self.get_pose_cam_in_world()
        hmat_world_in_cam = np.linalg.inv(hmat_cam_in_world)
        mat_tf_from_world_to_cam = hmat_world_in_cam[0:3, [0, 1, 3]]
            # completed to set
            #       mat_tf_from_world_to_cam: 3x3 matrix


        ### get transform from img_bev to world
        mat_tf_from_world_to_img_bev = self.get_transform_from_world_to_img_bev()
        mat_tf_from_img_bev_to_world = np.linalg.inv(mat_tf_from_world_to_img_bev)
            # completed to set
            #       mat_tf_from_img_bev_to_cam: 3x3 matrix


        ### combine
        mat_tf_from_img_bev_to_img_ori = mat_tf_from_cam_to_img_ori @ mat_tf_from_world_to_cam @ mat_tf_from_img_bev_to_world
        mat_tf_from_img_bev_to_img_ori_n = mat_tf_from_img_bev_to_img_ori / mat_tf_from_img_bev_to_img_ori[2, 2]
            # completed to set
            #       mat_tf_from_img_bev_to_img_ori_n: 3x3 matrix


        return mat_tf_from_img_bev_to_img_ori_n
    #end


    ###=============================================================================================
    ### [final] transformation: world -> camera -> ori-img
    ###=============================================================================================
    def get_transform_from_world_to_img_ori(self):
        """
        @ world -> camera:  by get_pose_cam_in_world()
        @ camera -> ori-img: by k
        """

        ### get transform from cam to img_ori
        mat_tf_from_cam_to_img_ori = self.mat_K
            # completed to set
            #       mat_tf_from_cam_to_img_ori: 3x3 matrix


        ### get transform from world to cam
        hmat_cam_in_world = self.get_pose_cam_in_world()
        hmat_world_in_cam = np.linalg.inv(hmat_cam_in_world)
        mat_tf_from_world_to_cam = hmat_world_in_cam[0:3, [0, 1, 3]]
            # completed to set
            #       mat_tf_from_world_to_cam: 3x3 matrix


        ### combine
        mat_tf_from_world_to_img_ori = mat_tf_from_cam_to_img_ori @ mat_tf_from_world_to_cam
            # completed to set
            #       mat_tf_from_img_bev_to_img_ori: 3x3 matrix


        return mat_tf_from_world_to_img_ori
    #end


#end



####################################################################################################################################
### __main__()
####################################################################################################################################
if __name__ == '__main__':
    ###=======================================================================================
    ### IPM parameters (by user)
    ###=======================================================================================
    param_mat_K= np.array([[500.0, 0.0, 640.0], [0.0, 500.0, 360.0], [0.0, 0.0, 1.0]])

    param_scale_pixel_per_meter = 20.0
    param_h_img_bev = 1000
    param_w_img_bev = 400
    param_offset_y = param_h_img_bev - 1
    param_offset_x = param_w_img_bev / 2.0

    theta_rotx_deg = -90.0 - 12.0
    vec_trans_cam_in_world = np.array([0.0, 0.0, 1.0])



    ###=======================================================================================
    ### loop parameters (by user)
    ###=======================================================================================
    format_fname_img_ori_in = 'C:\\Users\\JUNGWON KANG\\Desktop\\proj_MyIPM\\sample_imgs\\test_set\\clips\\0530\\1492626224112349377_0\\%d.jpg'
    format_fname_img_bev_out = 'C:\\Users\\JUNGWON KANG\\Desktop\\proj_MyIPM\\sample_imgs\\output\\bev_%d.jpg'

    idx_img_s = 0
    idx_img_e = 20

    b_show_img_ori_in = 0
    b_show_img_bev_out = 1


    ###=======================================================================================
    ### create object of MyIPM
    ###=======================================================================================
    obj_my_ipm = MyIPM(mat_K=param_mat_K,
                       param_scale_pixel_per_meter=param_scale_pixel_per_meter,
                       param_h_img_bev = param_h_img_bev,
                       param_w_img_bev = param_w_img_bev,
                       param_offset_y = param_offset_y,
                       param_offset_x = param_offset_x,
                       theta_rotx_deg = theta_rotx_deg,
                       vec_trans_cam_in_world = vec_trans_cam_in_world)


    ###=======================================================================================
    ### loop
    ###=======================================================================================
    for idx_img in range(idx_img_s, (idx_img_e + 1)):
        print("Processing img [%d]" % idx_img)

        ###========================================================================
        ### set fname
        ###========================================================================
        fname_img_ori_in = format_fname_img_ori_in % idx_img
        fname_img_bev_out = format_fname_img_bev_out % idx_img


        ###========================================================================
        ### read img_ori
        ###========================================================================
        img_ori_in = cv2.imread(fname_img_ori_in)

        if img_ori_in is None:
            print("  Can not read the img...")
            continue
        #end

        if b_show_img_ori_in:
            cv2.imshow("img_ori_in", img_ori_in)
            cv2.waitKey(1)
        #end


        ###========================================================================
        ### transform (img_ori_in to img_bev_out)
        ###========================================================================
        img_bev_out = obj_my_ipm.process(img_ori_in)

        cv2.imwrite(fname_img_bev_out, img_bev_out)
        cv2.waitKey(1)


        if b_show_img_bev_out:
            cv2.imshow("img_bev_out", img_bev_out)
            cv2.waitKey(1)
        #end
    #end

    print("The END")


#end
