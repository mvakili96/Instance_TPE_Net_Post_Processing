# Aug 23 2020
# Jungwon Kang


import cv2
import numpy as np
import math
import enum
import copy
import scipy


from helpers.utils.rail_path_graph.my_RPG_core  import TYPE_node
from helpers.utils.rail_path_graph.my_RPG_core  import DLList_RPG
from helpers.utils.my_common_type               import *

# <to-be-added>
#   condition: id_node_a (near) - id_node_b (far)
#   self.prev has a single id?


########################################################################################################################
###
########################################################################################################################
# class TYPE_path(enum.Enum):
#     EGO = 0
#     NON_EGO = 1     # NON-EGO: EGO path, but not approachable (by switch state)
# #End


########################################################################################################################
###
########################################################################################################################
class MyUtils_RailPathGraph:

    m_rgb_cluster = None

    m_obj_utils_3D      = None
    m_img_raw_rsz_uint8 = None

    m_img_supp_ori_for_vis = None       # supplementary img for visualization
    m_img_supp_ipm_for_vis = None


    m_temp_idx_img_vis_ori = 805        # temp
    m_temp_idx_img_vis_ipm = 805
    m_temp_idx_img_debug1_ori = 805
    m_temp_idx_img_debug1_ipm = 805
    m_temp_idx_img_final_path_ipm = 805


    ### params
    m_param_rpg_subedge_thres_dx_3d    = None
    m_param_rpg_subedge_thres_dy_img   = None
    m_param_rpg_subedge_height_section = None

    m_param_rpg_nodeedge_thres_dist_img_for_seed = None
    m_param_rpg_nodeedge_thres_dx_3d  = None
    m_param_rpg_nodeedge_thres_dy_img = None

    m_param_rpg_path_vertices_valid_y_min = None

    m_param_rpg_poly_fitting_y_max  = None
    m_param_rpg_poly_fitting_degree = None


    ###=========================================================================================================
    ### MyUtils_RailPathGraph::__init__()
    ###=========================================================================================================
    def __init__(self, dict_args):

        self.init_params(dict_args)
        self.init_rgb_table()
    #end


    ###=========================================================================================================
    ### MyUtils_RailPathGraph::__init__()
    ###=========================================================================================================
    def init_params(self, dict_args):

        self.m_param_rpg_subedge_thres_dx_3d    = dict_args["param_rpg_subedge_thres_dx_3d"]
        self.m_param_rpg_subedge_thres_dy_img   = dict_args["param_rpg_subedge_thres_dy_img"]
        self.m_param_rpg_subedge_height_section = dict_args["param_rpg_subedge_height_section"]

        self.m_param_rpg_nodeedge_thres_dist_img_for_seed = dict_args["param_rpg_nodeedge_thres_dist_img_for_seed"]
        self.m_param_rpg_nodeedge_thres_dx_3d             = dict_args["param_rpg_nodeedge_thres_dx_3d"]
        self.m_param_rpg_nodeedge_thres_dy_img            = dict_args["param_rpg_nodeedge_thres_dy_img"]

        self.m_param_rpg_path_vertices_valid_y_min  = dict_args["param_rpg_path_vertices_valid_y_min"]

        self.m_param_rpg_poly_fitting_y_max     = dict_args["param_rpg_poly_fitting_y_max"]
        self.m_param_rpg_poly_fitting_degree    = dict_args["param_rpg_poly_fitting_degree"]
    #end


    ###=========================================================================================================
    ### MyUtils_RailPathGraph::
    ###=========================================================================================================
    def init_rgb_table(self):
        ###
        rgb_cluster00 = [128,  64, 128]   #
        rgb_cluster01 = [244,  35, 232]   #
        rgb_cluster02 = [ 70,  70,  70]   #
        rgb_cluster03 = [192,   0, 128]   #
        rgb_cluster04 = [190, 153, 153]   #
        rgb_cluster05 = [153, 153, 153]   #
        rgb_cluster06 = [250, 170,  30]   #
        rgb_cluster07 = [220, 220,   0]   #
        rgb_cluster08 = [107, 142,  35]   #
        rgb_cluster09 = [152, 251, 152]   #
        rgb_cluster10 = [ 70, 130, 180]   #
        rgb_cluster11 = [220,  20,  60]   #
        rgb_cluster12 = [230, 150, 140]   #
        rgb_cluster13 = [  0,   0, 142]   #
        rgb_cluster14 = [  0,   0,  70]   #
        rgb_cluster15 = [ 90,  40,  40]   #
        rgb_cluster16 = [  0,  80, 100]   #
        rgb_cluster17 = [  0, 254, 254]   #
        rgb_cluster18 = [  0,  68,  63]   #


        ###
        rgb_labels = np.array(
            [
                rgb_cluster00,
                rgb_cluster01,
                rgb_cluster02,
                rgb_cluster03,
                rgb_cluster04,
                rgb_cluster05,
                rgb_cluster06,
                rgb_cluster07,
                rgb_cluster08,
                rgb_cluster09,
                rgb_cluster10,
                rgb_cluster11,
                rgb_cluster12,
                rgb_cluster13,
                rgb_cluster14,
                rgb_cluster15,
                rgb_cluster16,
                rgb_cluster17,
                rgb_cluster18,
            ]
        )


        ###
        self.m_rgb_cluster = rgb_labels
    #end


    ###=========================================================================================================
    ### MyUtils_RailPathGraph::process_b()
    ###=========================================================================================================
    def get_img_raw_rsz_uint8(self):
        return self.m_img_raw_rsz_uint8
    #end


    ###=========================================================================================================
    ### MyUtils_RailPathGraph::process_b()
    ###=========================================================================================================
    def process(self, list_triplet_points_local_max, obj_utils_3D, img_raw_rsz_uint8, seg_image, num_seg_classes, flag_seg_in_PP):
        """
        - create cluster-graph
        - decode paths from cluster-graph
        :return:
        """

        ###
        self.m_obj_utils_3D = obj_utils_3D
        self.m_img_raw_rsz_uint8 = img_raw_rsz_uint8

        ###
        h_img = img_raw_rsz_uint8.shape[0]
        w_img = img_raw_rsz_uint8.shape[1]


        ###------------------------------------------------------------------------------------------------
        ### 1. create subedges in all sections in an image
        ###------------------------------------------------------------------------------------------------
        list_dict_all_sections = self._create_subedges_in_all_sections(list_triplet_points_local_max, h_img, w_img)


        ###------------------------------------------------------------------------------------------------
        ### 2. create nodes & edges for rail path graph
        ###------------------------------------------------------------------------------------------------
        list_dict_edge, \
        max_id_edge_sofar, \
        list_type_node, \
        list_id_node_a_for_edge, \
        list_id_node_b_for_edge = self._create_nodes_edges_for_railpathgraph(list_dict_all_sections, h_img, w_img)

        # print(list_type_node)
        # print(list_id_node_a_for_edge)
        # print(list_id_node_b_for_edge)
        # print(list_type_node[0:5])
        # print(list_id_node_a_for_edge[0:5])
        # print(list_id_node_b_for_edge[0:5])

        ###------------------------------------------------------------------------------------------------
        ### 3. build rail path graph and get feasible paths (in the form of nodes) from the rail path graph
        ###------------------------------------------------------------------------------------------------
        list_paths_as_node_set = self._get_feasible_paths_from_railpathgraph(max_id_edge_sofar, list_type_node, list_id_node_a_for_edge, list_id_node_b_for_edge)
        print(list_paths_as_node_set)


        ###------------------------------------------------------------------------------------------------
        ### 4. convert paths (in the form of vertices)
        ###------------------------------------------------------------------------------------------------
        list_paths_as_vertices = self._convert_to_paths_as_vertices_v2(list_dict_edge, list_paths_as_node_set, list_type_node)


        ###------------------------------------------------------------------------------------------------
        ### 5. [TEMP at the moment] set path type (based on switch state)
        ###------------------------------------------------------------------------------------------------
        list_type_paths = self._set_type_paths(list_paths_as_vertices)

        # print(list_paths_as_vertices[0].keys())


        ###------------------------------------------------------------------------------------------------
        ### 6. get paths by polynomial fitting
        ###------------------------------------------------------------------------------------------------
        list_paths_final = self._get_paths_by_polynomial_fitting(list_paths_as_vertices, list_type_paths, seg_image, num_seg_classes, flag_seg_in_PP)

        # dick = list_paths_final[0]
        # # print(dick.keys())
        # dick1 = dick["polynomial"]
        # print(dick1.keys())


        ### debugging
        # if len(list_paths_final) >= 2:
        #     print("aaa")
        # #end

        return list_paths_final
    #end


    ###=========================================================================================================
    ### MyUtils_RailPathGraph::
    ###=========================================================================================================
    def _adjust_rgb(self, type, b_old_uint8, g_old_uint8, r_old_uint8):
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

        if type == 0:
            dr_int = -50
            dg_int = -50
            db_int = -50
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


    ###=========================================================================================================
    ###
    ###=========================================================================================================
    def _set_type_paths(self, list_paths_as_vertices):

        # This routine is for setting path type based on switch state.
        #   At the moment, it is under development, and it should be completed later.

        totnum_paths = len(list_paths_as_vertices)

        ###
        list_type_paths = [TYPE_path.NON_EGO]*totnum_paths


        if totnum_paths == 1:
            list_type_paths[0] = TYPE_path.EGO

        elif totnum_paths >= 2:
            ### At the moment, just set the right-most path as ego-path
            ###
            ego_idx_path = None
            ego_x_cen_3d = None

            for idx_path in range(totnum_paths):
                ###---------------------------------------------------------------------------------------
                ### get data for this path
                ###---------------------------------------------------------------------------------------
                dict_path_this = list_paths_as_vertices[idx_path]
                arr_xyz_cen_3d = dict_path_this["xyz_cen_3d"]
                x_cen_3d = arr_xyz_cen_3d[-1, 0]


                if ego_idx_path is None:
                    ego_idx_path = idx_path
                    ego_x_cen_3d = x_cen_3d
                else:
                    if x_cen_3d >= ego_x_cen_3d:
                        ego_idx_path = idx_path
                        ego_x_cen_3d = x_cen_3d
                    #end
                #end
            #end

            ###
            list_type_paths[ego_idx_path] = TYPE_path.EGO
        #end


        ### temp
        # if totnum_paths >= 2:
        #      a = 1
        # #end

        return list_type_paths


    ###=========================================================================================================
    ### MyUtils_RailPathGraph::create_subedges_in_each_section
    ###=========================================================================================================
    def _create_subedges_in_one_section(self, list_triplet_points_local_max, y_min, y_max, w_img, param_thres_dx_3d, param_thres_dy_img):

        ###---------------------------------------------------------------------------------------------
        ### init for this section
        ###---------------------------------------------------------------------------------------------
        max_id_cluster_sofar  = -1
        list_dict_pnt_cluster = [[] for _ in range(w_img)]  # can use any large number, instead of w_img
        list_dict_pnt_ref     = [None] * w_img  # can use any large number, instead of w_img
        # list_dict_pnt_ref: having dicts of
        #       dict_pnt_ref = {"id_cluster": id_cluster_for_this_pnt_ref,
        #                       "info_pnt": dict_pnt_this}


        ###---------------------------------------------------------------------------------------------
        ### sweep from near to far (in this section)
        ###---------------------------------------------------------------------------------------------
        for y in range(y_max, y_min - 1, -1):

            ### get pnt at this row
            list_thisrow = list_triplet_points_local_max[y]

            if len(list_thisrow) == 0:
                continue
            # end

            ### process for each pnt at this row
            for idx_dict_pnt_this, dict_pnt_this in enumerate(list_thisrow):
                ###---------------------------------------------------------------------------------
                ### get info of this pnt
                ###---------------------------------------------------------------------------------
                xy_cen_img_pnt_this = dict_pnt_this["xy_cen_img"]
                xyz_cen_3d_pnt_this = dict_pnt_this["xyz_cen_3d"]

                ###---------------------------------------------------------------------------------
                ### check if this pnt belongs to an existing cluster or newly appeared pnt
                ###---------------------------------------------------------------------------------
                b_new_pnt = True
                id_cluster_for_pnt_this = None

                ###
                if max_id_cluster_sofar == -1:
                    pass
                else:  # max_id_cluster_sofar >= 0

                    ### find existing cluster for min
                    dx_img_min = w_img * 1000
                    dy_img_for_min = None
                    dx_3d_for_min = None

                    for id_cluster_existing in range(0, max_id_cluster_sofar + 1):
                        ###
                        dict_pnt_ref = list_dict_pnt_ref[id_cluster_existing]["info_pnt"]
                        xy_cen_img_pnt_ref = dict_pnt_ref["xy_cen_img"]
                        xyz_cen_3d_pnt_ref = dict_pnt_ref["xyz_cen_3d"]

                        ###
                        dx_img = abs(xy_cen_img_pnt_this[0] - xy_cen_img_pnt_ref[0])
                        dy_img = abs(xy_cen_img_pnt_this[1] - xy_cen_img_pnt_ref[1])
                        dx_3d = abs(xyz_cen_3d_pnt_this[0] - xyz_cen_3d_pnt_ref[0])

                        ###
                        if dx_img <= dx_img_min:
                            id_cluster_for_pnt_this = id_cluster_existing

                            dx_img_min = dx_img
                            dy_img_for_min = dy_img
                            dx_3d_for_min = dx_3d
                        # end
                    # end
                    # completed to set
                    #   id_cluster_for_pnt_this
                    #   dx_img_min
                    #   dy_img_for_min
                    #   dx_3d_for_min

                    ### determine if this pnt is new/existing
                    # param_thres_dx_3d: e.g. 0.5
                    # param_thres_dy_img: 10

                    if dx_img_min < param_thres_dx_3d and dy_img_for_min < param_thres_dy_img:
                        b_new_pnt = False
                    else:
                        b_new_pnt = True
                    # end
                # end

                ###---------------------------------------------------------------------------------
                ### update
                ###---------------------------------------------------------------------------------
                # <if is a newly appeared pnt>
                #   set this pnt as a reference pnt
                #   update reference pnt info
                # ---------------------------------------------------------------------
                # <if it is an existing cluster>
                #   assign existing cluster-id
                #   update reference pnt info
                # ---------------------------------------------------------------------

                id_cluster_for_this_pnt_ref = -1
                ### set id_cluster_for_this_pnt_ref
                if b_new_pnt is True:
                    id_cluster_for_this_pnt_ref = max_id_cluster_sofar + 1
                    max_id_cluster_sofar = id_cluster_for_this_pnt_ref
                else:
                    id_cluster_for_this_pnt_ref = id_cluster_for_pnt_this
                # end


                ### set updated info for this_pnt_ref & this_pnt_cluster
                dict_pnt = {"id_subedge": id_cluster_for_this_pnt_ref,
                            "info_pnt": dict_pnt_this}


                ### update (this_pnt_ref)
                list_dict_pnt_ref[id_cluster_for_this_pnt_ref] = dict_pnt

                ### update (this cluster)
                list_dict_pnt_cluster[id_cluster_for_this_pnt_ref].append(dict_pnt)
            # end
        # end
        # completed to set
        #   list_dict_pnt_cluster
        #   max_id_cluster_sofar

        return list_dict_pnt_cluster, max_id_cluster_sofar
    #end


    ###=========================================================================================================
    ### MyUtils_RailPathGraph::create_subedges_in_each_section
    ###=========================================================================================================
    def _create_subedges_in_all_sections(self, list_triplet_points_local_max, h_img, w_img):
        """
        create subedges in each section in an image

        :param list_triplet_points_local_max:
        :param h_img:
        :param w_img:
        :return:
        """

        #------------------------------------------------------------------------------------------------
        # Please, see extract_triplet_localmax() in <my_utils_img.py>
        #   to see the definition of list_triplet_points_local_max.
        #------------------------------------------------------------------------------------------------

        ###
        # param_thres_dx_3d = 0.5
        # param_thres_dy_img = 10
        # param_height_section = 5
        param_thres_dx_3d = self.m_param_rpg_subedge_thres_dx_3d
        param_thres_dy_img = self.m_param_rpg_subedge_thres_dy_img
        param_height_section = self.m_param_rpg_subedge_height_section




        ###---------------------------------------------------------------------------------------------
        ### init
        ###---------------------------------------------------------------------------------------------
        list_dict_all_sections = []
        id_section = -1

        param_height_1st_section = 50
        id_section += 1
        dict_1st_section = {"id_section": id_section,
                            "y_min": h_img - param_height_1st_section,
                            "y_max": h_img - 1,
                            "subedges": None}
        list_dict_all_sections.append(dict_1st_section)

        for y in range(h_img - 1 - param_height_1st_section, -1, -1*param_height_section):
            ###
            id_section += 1

            ###
            y_max = y
            y_min = y - param_height_section + 1
            #print("y_max: [%d], y_min: [%d]" % (y_max, y_min))

            dict_one_section = {"id_section": id_section,
                                "y_min": y_min,
                                "y_max": y_max,
                                "subedges": None}

            list_dict_all_sections.append(dict_one_section)
        #end
            # completed to init
            #       list_id_section_for_y[]
            #       list_dict_all_sections[]

        totnum_sections = len(list_dict_all_sections)


        ###---------------------------------------------------------------------------------------------
        ### process each section
        ###---------------------------------------------------------------------------------------------
        for id_section in range(totnum_sections):
            ###
            y_min = list_dict_all_sections[id_section]["y_min"]
            y_max = list_dict_all_sections[id_section]["y_max"]

            ###
            list_dict_pnt_cluster, \
            max_id_cluster_sofar = self._create_subedges_in_one_section(list_triplet_points_local_max,
                                                                        y_min, y_max, w_img, param_thres_dx_3d, param_thres_dy_img)

            if id_section == 0:
                remove = []
                # print(len(list_dict_pnt_cluster[0]))
                for cnt in range(len(list_dict_pnt_cluster)):
                    if len(list_dict_pnt_cluster[cnt])>0 and len(list_dict_pnt_cluster[cnt])<30:
                        remove.append(cnt)
                # print(remove)
                # for item in remove:
                #     print(list_dict_pnt_cluster[item][0]["info_pnt"])
                list_dict_pnt_cluster = [list_dict_pnt_cluster[i] for i in range(len(list_dict_pnt_cluster)) if i not in remove]
                max_id_cluster_sofar = max_id_cluster_sofar - len(remove)

            ###
            if max_id_cluster_sofar == -1:
                list_dict_all_sections[id_section]["subedges"] = None
            else:
                list_dict_all_sections[id_section]["subedges"] = list_dict_pnt_cluster[0:(max_id_cluster_sofar+1)]
            #end
        #end
            # completed to set
            #       list_dict_all_sections[]


        ###---------------------------------------------------------------------------------------------
        ### visualize
        ###---------------------------------------------------------------------------------------------
        if 0:
            self._visualize_debug_subedges_in_all_sections_ori(list_dict_all_sections)
            self._visualize_debug_subedges_in_all_sections_ipm(list_dict_all_sections)
        #end


        return list_dict_all_sections
        # ---------------------------------------------------------------------------------------
        # list_dict_all_sections[], having each list of
        #       dict_one_section = {"id_section": id_section,
        #                           "y_min": y_min,
        #                           "y_max": y_max,
        #                           "subedges": None}
        #       where,
        #           "subedges" has
        #               dict_pnt = {"id_cluster": id_cluster_for_this_pnt_ref,
        #                           "info_pnt": dict_pnt_this}
        #       where,
        #           dict_pnt_this = {"centerness": c_this,
        #                            "xy_cen_img": [x_cen, y_cen],
        #                            "xy_left_img": [x_left, y_cen],
        #                            "xy_right_img": [x_right, y_cen],
        #                            "xyz_cen_3d": [x_cen_3d, y_cen_3d, z_cen_3d],
        #                            "xyz_left_3d": [x_left_3d, y_left_3d, z_left_3d],
        #                            "xyz_right_3d": [x_right_3d, y_right_3d, z_right_3d]}
        # ---------------------------------------------------------------------------------------
    #end


    ###=========================================================================================================
    ### MyUtils_RailPathGraph::
    ###=========================================================================================================
    def _create_nodes_edges_for_railpathgraph(self, list_dict_all_sections, h_img, w_img):
        #------------------------------------------------------------------------------------------------
        # list_dict_all_sections[], having each list of
        #       dict_one_section = {"id_section": id_section,
        #                           "y_min": y_min,
        #                           "y_max": y_max,
        #                           "subedges": None}
        #       where,
        #           "subedges" has
        #               dict_pnt = {"id_subedge": id_cluster_for_this_pnt_ref,
        #                           "info_pnt": dict_pnt_this}
        #
        #       where,
        #               dict_pnt_this = {"centerness": c_this,
        #                                "xy_cen_img": [x_cen, y_cen],
        #                                "xy_left_img": [x_left, y_cen],
        #                                "xy_right_img": [x_right, y_cen],
        #                                "xyz_cen_3d": [x_cen_3d, y_cen_3d, z_cen_3d],
        #                                "xyz_left_3d": [x_left_3d, y_left_3d, z_left_3d],
        #                                "xyz_right_3d": [x_right_3d, y_right_3d, z_right_3d]}
        #------------------------------------------------------------------------------------------------


        ###
        # param_thres_dist_img_for_seed = int(w_img*0.1)
        # param_thres_dx_3d       = 1.0
        # param_thres_dy_img      = 20
        param_thres_dist_img_for_seed = self.m_param_rpg_nodeedge_thres_dist_img_for_seed
        param_thres_dx_3d             = self.m_param_rpg_nodeedge_thres_dx_3d
        param_thres_dy_img            = self.m_param_rpg_nodeedge_thres_dy_img



        ###---------------------------------------------------------------------------------------------------------
        ### init
        ###---------------------------------------------------------------------------------------------------------
        max_id_edge_sofar  = -1
        list_dict_edge     = [[] for _ in range(w_img)]     # saving subedge, can use any large number, instead of w_img
        list_dict_ref      = [None]*w_img                   # saving subedge, can use any large number, instead of w_img
            # list_dict_ref: having dicts of
            #       dict_pnt = {"id_subedge": id_cluster_for_this_pnt_ref,
            #                   "info_pnt": dict_pnt_this}

        list_type_node          = [None]*w_img      # storing node type, list_type_node[id_node] = TYPE_node
        list_id_node_a_for_edge = [None]*w_img      # storing id_node for an edge, list_id_node_a_for_edge[id_edge] = id_node
        list_id_node_b_for_edge = [None]*w_img


        ###---------------------------------------------------------------------------------------------------------
        ### sweep from near to far
        ###---------------------------------------------------------------------------------------------------------
        b_found_seed_subedge = False

        for id_section in range(len(list_dict_all_sections)):   # len(list_dict_all_sections): totnum_sections
            ###
            subedges = list_dict_all_sections[id_section]["subedges"]


            ###
            if subedges is None:
                continue
            #end
            # print(len(subedges))

            ###------------------------------------------------------------------------------------------------
            ### find seed-subedge
            ###------------------------------------------------------------------------------------------------
            if b_found_seed_subedge is False:
                ## IF seed subedge is not found yet.

                #------------------------------------------------------------------------------------------
                # find a seed-candidate subedge
                #------------------------------------------------------------------------------------------
                id_subedge_min        = -1
                dist_img_for_seed_min = 10000000

                for id_subedge in range(len(subedges)):   # len(subedges): totnum_subedges_in_this_section
                    ### get pnts in this subedge
                    list_dict_this_subedge = subedges[id_subedge]
                    #totnum_pnts_this_subedge = len(list_dict_this_subedge)

                    ### get pnt of this subedge (nearest to the camera)
                    xy_cen_img_pnt_this_subedge_nearest = list_dict_this_subedge[0]["info_pnt"]["xy_cen_img"]
                    #xyz_cen_3d_pnt_this_subedge_nearest = list_dict_this_subedge[0]["info_pnt"]["xyz_cen_3d"]

                    ###
                    dx = xy_cen_img_pnt_this_subedge_nearest[0] - (w_img/2)
                    dy = xy_cen_img_pnt_this_subedge_nearest[1] - (h_img - 1)
                    dist_this = math.sqrt( (dx*dx) + (dy*dy) )

                    ###
                    # print(dist_this)

                    if dist_this < param_thres_dist_img_for_seed:
                        if dist_this < dist_img_for_seed_min:
                            id_subedge_min = id_subedge
                            dist_img_for_seed_min = dist_this
                        #end
                    #end
                #end
                    # completed to set
                    #       id_subedge_min: id_subedge for seed


                #------------------------------------------------------------------------------------------
                # update on seed-subedge
                #------------------------------------------------------------------------------------------
                if id_subedge_min == -1:    # if a seed-subedge is not found
                    pass
                else:       # if a seed-subedge is found -> assign valid id_edge
                    assert max_id_edge_sofar == -1      # max_id_edge_sofar should be -1.

                    ###
                    id_edge_for_this_subedge = 0
                    max_id_edge_sofar        = 0
                    list_dict_this_subedge   = subedges[id_subedge_min]

                    dict_subedge_for_edge = {"id_edge": id_edge_for_this_subedge,
                                             "list_dict_this_subedge": list_dict_this_subedge
                                             }

                    dict_subedge_for_ref  = {"id_edge": id_edge_for_this_subedge,
                                             "list_dict_this_subedge": list_dict_this_subedge,
                                             "b_active": True
                                             }

                    ### update (edge)
                    list_dict_edge[id_edge_for_this_subedge].append(dict_subedge_for_edge)

                    ### update (ref)
                    list_dict_ref[id_edge_for_this_subedge] = dict_subedge_for_ref

                    ### update list_type_node[] (for node)
                    new_id_node                 = id_edge_for_this_subedge

                    list_type_node[new_id_node] = TYPE_node.END
                    list_id_node_a_for_edge[id_edge_for_this_subedge] = -1      # -1: seed node
                    list_id_node_b_for_edge[id_edge_for_this_subedge] =  0

                    ### set flag
                    b_found_seed_subedge = True
                #end
                    # completed to set
                    #       list_dict_edge[0]
                    #       list_dict_ref[0]


                #------------------------------------------------------------------------------------------
                # continue
                #------------------------------------------------------------------------------------------
                # print("seed done")
                continue
            #end


            # At this point, seed-subedge is found.
            # Note that
            #   list_dict_ref[id_edge]  : having ref points for id_edge assignment
            #   list_dict_edge[id_edge] : having info for id_edge

            ###------------------------------------------------------------------------------------------------
            ### check if a subedge belongs to an existing edge
            ###------------------------------------------------------------------------------------------------
            list_id_edge_existing_for_subedges = [-1]*len(subedges)
            list_id_subedge_for_id_edge_existing = [[] for _ in range(max_id_edge_sofar + 1)]
                # list_id_edge_existing_for_subedges[id_subedge] = id_edge_existing
                # list_id_subedge_for_id_edge_existing[id_edge].append(id_subedge)


            for id_subedge in range(len(subedges)):   # len(subedges): totnum_subedges_in_this_section
                #------------------------------------------------------------------------------------------
                # get pnts in this subedge
                #------------------------------------------------------------------------------------------
                list_dict_this_subedge = subedges[id_subedge]

                ### get pnt of this subedge (nearest to the camera)
                xy_cen_img_pnt_this_subedge_nearest = list_dict_this_subedge[0]["info_pnt"]["xy_cen_img"]
                xyz_cen_3d_pnt_this_subedge_nearest = list_dict_this_subedge[0]["info_pnt"]["xyz_cen_3d"]


                #------------------------------------------------------------------------------------------
                # check if this subedge belongs to an existing valid-edge or not
                #------------------------------------------------------------------------------------------

                ### find existing valid-edge for min
                dx_img_min     = w_img * 1000
                id_edge_existing_for_this_subedge = -1


                for id_edge_existing in range(0, max_id_edge_sofar + 1):
                    ###
                    b_active_this_ref = list_dict_ref[id_edge_existing]["b_active"]

                    if b_active_this_ref is False:
                        continue
                    #end

                    ###
                    list_dict_this_subedge_ref   = list_dict_ref[id_edge_existing]["list_dict_this_subedge"]
                    totnum_pnts_this_subedge_ref = len(list_dict_this_subedge_ref)

                    xy_cen_img_pnt_ref = list_dict_this_subedge_ref[totnum_pnts_this_subedge_ref - 1]["info_pnt"]["xy_cen_img"]     # farthest pnt in subedge
                    xyz_cen_3d_pnt_ref = list_dict_this_subedge_ref[totnum_pnts_this_subedge_ref - 1]["info_pnt"]["xyz_cen_3d"]     # farthest pnt in subedge

                    ###
                    dx_img = abs(xy_cen_img_pnt_this_subedge_nearest[0] - xy_cen_img_pnt_ref[0])
                    dy_img = abs(xy_cen_img_pnt_this_subedge_nearest[1] - xy_cen_img_pnt_ref[1])
                    dx_3d  = abs(xyz_cen_3d_pnt_this_subedge_nearest[0] - xyz_cen_3d_pnt_ref[0])

                    ###
                    if dx_img <= dx_img_min:
                        if dx_img <= param_thres_dx_3d and dy_img <= param_thres_dy_img:
                            # param_thres_dx_3d: e.g. 0.5
                            # param_thres_dy_img: 10

                            id_edge_existing_for_this_subedge = id_edge_existing
                            dx_img_min                        = dx_img
                        #end
                    #end
                #end
                    # completed to set
                    #   id_edge_existing_for_this_subedge


                #------------------------------------------------------------------------------------------
                # store
                #------------------------------------------------------------------------------------------
                list_id_edge_existing_for_subedges[id_subedge] = id_edge_existing_for_this_subedge

                if id_edge_existing_for_this_subedge != -1:
                    list_id_subedge_for_id_edge_existing[id_edge_existing_for_this_subedge].append(id_subedge)
                #end
            #end
                # completed to set
                #       list_id_edge_existing_for_subedges[id_subedge]
                #       list_id_subedge_for_id_edge_existing[id_edge]



            # At this point,
            #   Based on
            #       list_id_edge_existing_for_subedges[id_subedge]
            #       list_id_subedge_for_id_edge_existing[id_edge]
            #       we need to update the following things:
            #       list_dict_ref[id_edge]  : having ref points for id_edge assignment
            #       list_dict_edge[id_edge] : having info for id_edge

            ###------------------------------------------------------------------------------------------------
            ### assign id_edge to subedge
            ###------------------------------------------------------------------------------------------------
            for id_subedge in range(len(subedges)):   # len(subedges): totnum_subedges_in_this_section
                #------------------------------------------------------------------------------------------
                # get pnts in this subedge
                #------------------------------------------------------------------------------------------
                id_edge_existing_for_this_subedge = list_id_edge_existing_for_subedges[id_subedge]

                if id_edge_existing_for_this_subedge == -1:
                    continue
                #end


                #------------------------------------------------------------------------------------------
                # get totnum subedges for id_edge_existing_for_this_subedge
                #------------------------------------------------------------------------------------------
                totnum_subedges_for_this_id_edge_existing = len(list_id_subedge_for_id_edge_existing[id_edge_existing_for_this_subedge])

                assert totnum_subedges_for_this_id_edge_existing != 0


                #------------------------------------------------------------------------------------------
                # update list_dict_ref[id_edge] & list_dict_edge[id_edge]
                #------------------------------------------------------------------------------------------
                list_dict_this_subedge = subedges[id_subedge]

                ###
                if totnum_subedges_for_this_id_edge_existing == 1:

                    ###
                    dict_subedge_for_edge = {"id_edge": id_edge_existing_for_this_subedge,
                                             "list_dict_this_subedge": list_dict_this_subedge
                                             }

                    dict_subedge_for_ref = {"id_edge": id_edge_existing_for_this_subedge,
                                            "list_dict_this_subedge": list_dict_this_subedge,
                                            "b_active": True
                                            }

                    ### update (edge)
                    list_dict_edge[id_edge_existing_for_this_subedge].append(dict_subedge_for_edge)

                    ### update (ref)
                    list_dict_ref[id_edge_existing_for_this_subedge] = dict_subedge_for_ref

                else: # totnum_subedges_for_this_id_edge_existing >= 2
                    #------------------------------------------------------------------------------
                    # register new edge
                    #------------------------------------------------------------------------------

                    ### assign new edge id
                    new_id_edge_for_this_subedge = max_id_edge_sofar + 1
                    max_id_edge_sofar = new_id_edge_for_this_subedge


                    ###
                    dict_subedge_for_edge = {"id_edge": new_id_edge_for_this_subedge,
                                             "list_dict_this_subedge": list_dict_this_subedge
                                             }

                    dict_subedge_for_ref = {"id_edge": new_id_edge_for_this_subedge,
                                            "list_dict_this_subedge": list_dict_this_subedge,
                                            "b_active": True
                                            }

                    ### update (edge)
                    list_dict_edge[new_id_edge_for_this_subedge].append(dict_subedge_for_edge)

                    ### update (ref)
                    list_dict_ref[new_id_edge_for_this_subedge] = dict_subedge_for_ref


                    #------------------------------------------------------------------------------
                    # make id_edge_existing_for_this_subedge INACTIVE
                    #------------------------------------------------------------------------------
                    list_dict_ref[id_edge_existing_for_this_subedge]["b_active"] = False


                    #------------------------------------------------------------------------------
                    # update list_type_node[] (for node) & list_id_node_X_for_edge[] (for edge)
                    #------------------------------------------------------------------------------
                    new_id_node      = new_id_edge_for_this_subedge
                    existing_id_node = id_edge_existing_for_this_subedge

                    list_type_node[new_id_node]      = TYPE_node.END
                    list_type_node[existing_id_node] = TYPE_node.SWITCH

                    list_id_node_a_for_edge[new_id_edge_for_this_subedge] = existing_id_node
                    list_id_node_b_for_edge[new_id_edge_for_this_subedge] = new_id_node
                #end
            #end
        #end
            # completed to set
            #   list_dict_edge
            #   list_type_node
            #   list_id_node_a_for_edge[id_edge] = id_node
            #   list_id_node_b_for_edge[id_edge] = id_node


        ###---------------------------------------------------------------------------------------------------------
        ### visualize
        ###---------------------------------------------------------------------------------------------------------
        if 0:
            self._visualize_debug_edge_ori(list_dict_edge, max_id_edge_sofar, list_dict_all_sections)
            self._visualize_debug_edge_ipm(list_dict_edge, max_id_edge_sofar, list_dict_all_sections)
        #end


        return list_dict_edge, max_id_edge_sofar, list_type_node, list_id_node_a_for_edge, list_id_node_b_for_edge
        #------------------------------------------------------------------------------------------------------
        # list_dict_edge:
        #   list_dict_edge[i]: having {list:N}
        #                      where, each list in {list:N} is {dict:2},
        #                      where, the dict is {'id_edge': X, 'list_dict_this_subedge' is {list:M}}
        #                      where, each list in {list:M} is a point for this subedge.
        #
        # list_type_node:
        #   list_type_node[id_node] : TYPE_node.START/SWITCH/END
        #
        # list_id_node_a_for_edge:
        # list_id_node_b_for_edge:
        #   list_id_node_a_for_edge[id_edge]: having id_node_a (for id_edge)
        #   list_id_node_a_for_edge[id_edge]: having id_node_b (for id_edge)
        #------------------------------------------------------------------------------------------------------
    #end

    ###=========================================================================================================
    ### MyUtils_RailPathGraph::
    ###=========================================================================================================
    def _get_feasible_paths_from_railpathgraph(self, max_id_edge_sofar, list_type_node, list_id_node_a_for_edge, list_id_node_b_for_edge):
        """

        :param list_dict_edge:
        :param max_id_edge_sofar:
        :param list_type_node:
        :param list_id_node_a_for_edge:
        :param list_id_node_b_for_edge:
        :return:
        """

        # list_all_nodes  = [-1, 20, 30, 40]
        # list_type_nodes = [(TYPE_node.START), (TYPE_node.SWITCH), (TYPE_node.END), (TYPE_node.END)]
        # list_all_node_pairs_for_edges = [[-1, 20], [20, 30], [20, 40]]

        totnum_nodes = max_id_edge_sofar + 1
        totnum_edges = max_id_edge_sofar + 1


        ###=============================================================================================
        ### create obj
        ###=============================================================================================
        obj_rpg = DLList_RPG()


        ###=============================================================================================
        ###  create nodes
        ###=============================================================================================

        ### seed node
        obj_rpg.create_node(-1, TYPE_node.START)


        ###
        for id_node in range(totnum_nodes):
            ###
            type_node = list_type_node[id_node]
            #print("id_node: [%d], type_node [%s]" % (id_node, type_node))

            ###
            obj_rpg.create_node(id_node, type_node)
        #end


        ###=============================================================================================
        ### update connections
        ###=============================================================================================
        for id_edge in range(totnum_edges):
            ###
            id_node_a = list_id_node_a_for_edge[id_edge]
            id_node_b = list_id_node_b_for_edge[id_edge]
            #print('id_node_a: [%d], id_node_b: [%d]' % (id_node_a, id_node_b))

            ###
            obj_rpg.update_connections_in_node(id_node_a, id_node_b)
        #end

        # <to-be-added>
        #   condition: id_node_a (near) - id_node_b (far)
        #   self.prev has a single id?


        ###=============================================================================================
        ### get feasible path (as node set)
        ###=============================================================================================
        list_paths_as_node_set = obj_rpg.get_feasible_paths_as_node_set()

        # At this point, completed to set
        #   list_paths_as_node_set: a set of nodes


        return list_paths_as_node_set

        #---------------------------------------------------------------------------------------
        # list_paths_as_node_set:
        #   list_paths_as_node_set[i]: has N nodes for this path
        #---------------------------------------------------------------------------------------
    #end


    ###=========================================================================================================
    ### MyUtils_RailPathGraph::
    ###=========================================================================================================
    def _convert_to_paths_as_vertices(self, list_dict_edge, list_paths_as_node_set):
        """

        :param list_dict_edge:
        :param list_paths_as_node_set:
        :return:
        """

        # This function includes a routine for filtering-out too-short paths.

        # Note that id_node in list_paths_as_node_set also indicates id_edge.


        #param_valid_y_min = 10.0    # (meter)
        param_valid_y_min = self.m_param_rpg_path_vertices_valid_y_min


        ###
        #list_paths_as_vertices_out = [None]*totnum_paths
        list_paths_as_vertices_out = []


        ###
        totnum_paths = len(list_paths_as_node_set)

        for idx_path in range(totnum_paths):
            ###---------------------------------------------------------------------------------------------
            ###
            ###---------------------------------------------------------------------------------------------
            dict_path_this = {"xy_cen_img"  : [],
                              "xy_left_img" : [],
                              "xy_right_img": [],
                              "xyz_cen_3d"  : [],
                              "xyz_left_3d" : [],
                              "xyz_right_3d": []
                              }

            ###---------------------------------------------------------------------------------------------
            ###
            ###---------------------------------------------------------------------------------------------
            list_nodes_for_this_path = list_paths_as_node_set[idx_path]   # nodes for one path

            for idx_node in range(1, len(list_nodes_for_this_path)):
                id_node = list_nodes_for_this_path[idx_node]
                id_edge = id_node

                list_edge_this = list_dict_edge[id_edge]

                for idx_subedge in range(len(list_edge_this)):
                    list_pnts = list_edge_this[idx_subedge]["list_dict_this_subedge"]

                    for idx_pnts in range(len(list_pnts)):
                        ###
                        xy_cen_img = list_pnts[idx_pnts]["info_pnt"]["xy_cen_img"]
                        xy_left_img = list_pnts[idx_pnts]["info_pnt"]["xy_left_img"]
                        xy_right_img = list_pnts[idx_pnts]["info_pnt"]["xy_right_img"]
                        xyz_cen_3d = list_pnts[idx_pnts]["info_pnt"]["xyz_cen_3d"]
                        xyz_left_3d = list_pnts[idx_pnts]["info_pnt"]["xyz_left_3d"]
                        xyz_right_3d = list_pnts[idx_pnts]["info_pnt"]["xyz_right_3d"]


                        ###
                        dict_path_this["xy_cen_img"].append(xy_cen_img)
                        dict_path_this["xy_left_img"].append(xy_left_img)
                        dict_path_this["xy_right_img"].append(xy_right_img)
                        dict_path_this["xyz_cen_3d"].append(xyz_cen_3d)
                        dict_path_this["xyz_left_3d"].append(xyz_left_3d)
                        dict_path_this["xyz_right_3d"].append(xyz_right_3d)
                    #end
                #end
            #end
                # completed to set
                #       dict_path_this{}


            ###---------------------------------------------------------------------------------------------
            ### convert to ndarr
            ###---------------------------------------------------------------------------------------------
            temp0 = dict_path_this["xy_cen_img"]
            temp1 = dict_path_this["xy_left_img"]
            temp2 = dict_path_this["xy_right_img"]
            temp3 = dict_path_this["xyz_cen_3d"]
            temp4 = dict_path_this["xyz_left_3d"]
            temp5 = dict_path_this["xyz_right_3d"]

            dict_path_this["xy_cen_img"] = np.array(temp0)
            dict_path_this["xy_left_img"] = np.array(temp1)
            dict_path_this["xy_right_img"] = np.array(temp2)
            dict_path_this["xyz_cen_3d"] = np.array(temp3)
            dict_path_this["xyz_left_3d"] = np.array(temp4)
            dict_path_this["xyz_right_3d"] = np.array(temp5)


            ###---------------------------------------------------------------------------------------------
            ### filter out
            ###---------------------------------------------------------------------------------------------
            arr_xyz_cen_3d = dict_path_this["xyz_cen_3d"]
            arr_y = arr_xyz_cen_3d[:, 1]
            max_y_this = max(arr_y)


            if max_y_this < param_valid_y_min:
                continue
            #end


            ###---------------------------------------------------------------------------------------------
            ### store
            ###---------------------------------------------------------------------------------------------
            list_paths_as_vertices_out.append(dict_path_this)
        #end


        return list_paths_as_vertices_out
        #---------------------------------------------------------------------------------------------------
        # list_paths_as_vertices_out:
        #   list_paths_as_vertices_out[i]: ith path, having the following {dict:6}
        #               'xy_cen_img': ndarr:(N, 2), where e.g. N is 350.
        #               'xy_left_img': ndarr:(N, 2)
        #               'xy_right_img': ndarr:(N, 2)
        #               'xyz_cen_3d': ndarr:(N, 3)
        #               'xyz_left_3d': ndarr:(N, 3)
        #               'xyz_right_3d': ndarr:(N, 3)
        #---------------------------------------------------------------------------------------------------
    #end


    ###=========================================================================================================
    ### MyUtils_RailPathGraph::
    ###=========================================================================================================
    def _convert_to_paths_as_vertices_v2(self, list_dict_edge, list_paths_as_node_set, list_type_node):
        """

        :param list_dict_edge:
        :param list_paths_as_node_set:
        :param list_type_node:
        :return:
        """

        # This function includes a routine for filtering-out too-short paths.
        # Note that id_node in list_paths_as_node_set also indicates id_edge.


        #param_valid_y_min = 10.0    # (meter)
        param_valid_y_min = self.m_param_rpg_path_vertices_valid_y_min


        ###
        list_paths_as_vertices_out = []


        ###
        totnum_paths = len(list_paths_as_node_set)

        for idx_path in range(totnum_paths):
            ###---------------------------------------------------------------------------------------------
            ###
            ###---------------------------------------------------------------------------------------------
            dict_path_this = {"id_edge"        : [],
                              "xy_cen_img"     : [],
                              "xy_left_img"    : [],
                              "xy_right_img"   : [],
                              "xyz_cen_3d"     : [],
                              "xyz_left_3d"    : [],
                              "xyz_right_3d"   : [],
                              ###
                              "id_node_switch": [],   # switch (id_node)
                              "xy_switch_img"  : [],  # switch (img)
                              "xyz_switch_3d"  : [],  # switch (3d)
                              ###
                              "xy_switch_img_edge_start": [],
                              "xyz_switch_3d_edge_start": [],
                              "xy_switch_img_edge_end":   [],    # equal to "xy_switch_img"
                              "xyz_switch_3d_edge_end":   []     # equal to "xyz_switch_3d"
                              }

            ###---------------------------------------------------------------------------------------------
            ###
            ###---------------------------------------------------------------------------------------------
            list_nodes_for_this_path = list_paths_as_node_set[idx_path]   # nodes for one path


            for idx_node in range(1, len(list_nodes_for_this_path)):
                id_node = list_nodes_for_this_path[idx_node]
                id_edge = id_node
                type_node = list_type_node[id_node]

                list_edge_this = list_dict_edge[id_edge]

                ###
                b_is_first = True
                xy_cen_img_first = None
                xyz_cen_3d_first = None

                for idx_subedge in range(len(list_edge_this)):  # For subedges in this edge
                    list_pnts = list_edge_this[idx_subedge]["list_dict_this_subedge"]

                    for idx_pnts in range(len(list_pnts)):
                        ###
                        xy_cen_img   = list_pnts[idx_pnts]["info_pnt"]["xy_cen_img"]
                        xy_left_img  = list_pnts[idx_pnts]["info_pnt"]["xy_left_img"]
                        xy_right_img = list_pnts[idx_pnts]["info_pnt"]["xy_right_img"]
                        xyz_cen_3d   = list_pnts[idx_pnts]["info_pnt"]["xyz_cen_3d"]
                        xyz_left_3d  = list_pnts[idx_pnts]["info_pnt"]["xyz_left_3d"]
                        xyz_right_3d = list_pnts[idx_pnts]["info_pnt"]["xyz_right_3d"]

                        ###
                        dict_path_this["id_edge"].append(id_edge)
                        dict_path_this["xy_cen_img"].append(xy_cen_img)
                        dict_path_this["xy_left_img"].append(xy_left_img)
                        dict_path_this["xy_right_img"].append(xy_right_img)
                        dict_path_this["xyz_cen_3d"].append(xyz_cen_3d)
                        dict_path_this["xyz_left_3d"].append(xyz_left_3d)
                        dict_path_this["xyz_right_3d"].append(xyz_right_3d)


                        if b_is_first is True:
                            xy_cen_img_first = xy_cen_img
                            xyz_cen_3d_first = xyz_cen_3d
                            b_is_first = False
                        #end

                    #end
                #end


                ### store the last elements in this edge (for switch)
                if type_node is TYPE_node.SWITCH:
                    dict_path_this["id_node_switch"].append(id_node)
                    dict_path_this["xy_switch_img"].append(xy_cen_img)
                    dict_path_this["xyz_switch_3d"].append(xyz_cen_3d)

                    dict_path_this["xy_switch_img_edge_start"].append(xy_cen_img_first)
                    dict_path_this["xyz_switch_3d_edge_start"].append(xyz_cen_3d_first)
                    dict_path_this["xy_switch_img_edge_end"].append(xy_cen_img)
                    dict_path_this["xyz_switch_3d_edge_end"].append(xyz_cen_3d)
                #end
            #end
                # completed to set
                #       dict_path_this{}


            ###---------------------------------------------------------------------------------------------
            ### convert to ndarr
            ###---------------------------------------------------------------------------------------------
            temp0 = dict_path_this["xy_cen_img"]
            temp1 = dict_path_this["xy_left_img"]
            temp2 = dict_path_this["xy_right_img"]
            temp3 = dict_path_this["xyz_cen_3d"]
            temp4 = dict_path_this["xyz_left_3d"]
            temp5 = dict_path_this["xyz_right_3d"]
            temp6 = dict_path_this["xy_switch_img"]
            temp7 = dict_path_this["xyz_switch_3d"]

            dict_path_this["xy_cen_img"]    = np.array(temp0)
            dict_path_this["xy_left_img"]   = np.array(temp1)
            dict_path_this["xy_right_img"]  = np.array(temp2)
            dict_path_this["xyz_cen_3d"]    = np.array(temp3)
            dict_path_this["xyz_left_3d"]   = np.array(temp4)
            dict_path_this["xyz_right_3d"]  = np.array(temp5)
            dict_path_this["xy_switch_img"] = np.array(temp6)
            dict_path_this["xyz_switch_3d"] = np.array(temp7)


            ###---------------------------------------------------------------------------------------------
            ### filter out
            ###---------------------------------------------------------------------------------------------
            # arr_xyz_cen_3d = dict_path_this["xyz_cen_3d"]
            # arr_y = arr_xyz_cen_3d[:, 1]
            # max_y_this = max(arr_y)
            # print(dict_path_this["xy_cen_img"])

            arr_xyz_cen_3d = dict_path_this["xy_cen_img"]
            arr_y = arr_xyz_cen_3d[:, 1]
            max_y_this = max(arr_y)

            if max_y_this < param_valid_y_min:
                print(idx_path)
                continue
            #end


            ###---------------------------------------------------------------------------------------------
            ### store
            ###---------------------------------------------------------------------------------------------
            list_paths_as_vertices_out.append(dict_path_this)
        #end


        ### debug
        # if totnum_paths >= 2:
        #     aaa = 1
        # #end


        return list_paths_as_vertices_out
        #---------------------------------------------------------------------------------------------------
        # list_paths_as_vertices_out:
        #   list_paths_as_vertices_out[i]: ith path, having the following {dict:6}
        #               'xy_cen_img': ndarr:(N, 2), where e.g. N is 350.
        #               'xy_left_img': ndarr:(N, 2)
        #               'xy_right_img': ndarr:(N, 2)
        #               'xyz_cen_3d': ndarr:(N, 3)
        #               'xyz_left_3d': ndarr:(N, 3)
        #               'xyz_right_3d': ndarr:(N, 3)
        #
        #               'id_edge'
        #
        #               'xy_switch_img'
        #               'xyz_switch_3d'
        #               'id_node_switch'
        #---------------------------------------------------------------------------------------------------
    #end

    def _ransac_polyfit(self, x, y, order=3, n=20, k=100, t=0.1, f=0.8):
        # Thanks https://en.wikipedia.org/wiki/Random_sample_consensus

        # n – minimum number of data points required to fit the model
        # k – maximum number of iterations allowed in the algorithm
        # t – threshold value to determine when a data point fits a model
        # d – number of close data points required to assert that a model fits well to data
        # f – fraction of close data points required

        besterr = np.inf
        bestfit = None
        for kk in range(k):
            maybeinliers = np.random.randint(len(x), size=n)
            maybemodel = np.polyfit(x[maybeinliers], y[maybeinliers], order)
            alsoinliers = np.abs(np.polyval(maybemodel, x) - y) <= t
            if sum(alsoinliers) >= len(x) * f:
                bettermodel = np.polyfit(x[alsoinliers], y[alsoinliers], order)
                thiserr = np.sum(np.abs(np.polyval(bettermodel, x[alsoinliers]) - y[alsoinliers]))
                if thiserr < besterr:
                    bestfit = bettermodel
                    besterr = thiserr
        return bestfit

    ###=========================================================================================================
    ### MyUtils_RailPathGraph::
    ###=========================================================================================================
    def _get_paths_by_polynomial_fitting(self, list_paths_as_vertices, n_seg_classes, seg_im, flg):
        """

        :param list_paths_as_vertices:
        :return:
        """

        #param_y_max = 150.0
        #param_degree_poly = 2
        param_y_max = self.m_param_rpg_poly_fitting_y_max
        param_degree_poly = self.m_param_rpg_poly_fitting_degree




        totnum_paths = len(list_paths_as_vertices)

        ###
        list_paths_out = []

        for idx_path in range(totnum_paths):
            ###---------------------------------------------------------------------------------------
            ### get data for this path
            ###---------------------------------------------------------------------------------------
            dict_path_this = list_paths_as_vertices[idx_path]

            ###
            arr_xy_left  = dict_path_this["xy_left_img"]
            arr_xy_right = dict_path_this["xy_right_img"]


            arr_x_left_ori  = arr_xy_left[:, 1]
            arr_x_left_ori  = arr_x_left_ori[::-1]

            arr_y_left_ori  = arr_xy_left[:, 0]
            arr_y_left_ori = arr_y_left_ori[::-1]

            arr_x_right_ori  = arr_xy_right[:, 1]
            arr_x_right_ori = arr_x_right_ori[::-1]

            arr_y_right_ori  = arr_xy_right[:, 0]
            arr_y_right_ori = arr_y_right_ori[::-1]

            # arr_y_cen_orii = 0.5*(arr_y_left_ori+arr_y_right_ori)
            # print(arr_y_cen_orii)




            if n_seg_classes == 3:
                rgb_rail = 232                #254
            elif n_seg_classes == 19:
                rgb_rail = 254                #254

            if flg is True:
                for counter in range(len(arr_y_left_ori)-2,-1,-1):
                    y_left_this = arr_y_left_ori[counter]
                    x_left_this = arr_x_left_ori[counter]
                    flag_no_change = 0
                    for offset in range(6):
                        if y_left_this-offset>0 and y_left_this+offset<960:
                            if seg_im[x_left_this,y_left_this-offset,0] == rgb_rail or seg_im[x_left_this,y_left_this+offset,0] == rgb_rail:                                #or seg_im[x_left_this,y_right_this-offset,0] == rgb_rail or seg_im[x_left_this,y_right_this+offset,0] == rgb_rail:
                                flag_no_change = 1
                                break
                    if flag_no_change == 0:
                        for offset in range(6,50):
                            if y_left_this - offset > 0 and y_left_this + offset < 960:
                                if seg_im[x_left_this, y_left_this - offset, 0] == rgb_rail and seg_im[x_left_this, y_left_this + offset, 0] != rgb_rail and abs(y_left_this - offset - arr_y_left_ori[counter+1])<10:
                                    arr_y_left_ori[counter] = y_left_this - offset
                                    break
                                elif seg_im[x_left_this, y_left_this - offset, 0] != rgb_rail and seg_im[x_left_this, y_left_this + offset, 0] == rgb_rail and abs(y_left_this + offset - arr_y_left_ori[counter+1])<10:
                                    arr_y_left_ori[counter] = y_left_this + offset
                                    break

            if flg is True:
                for counter in range(len(arr_y_right_ori)-2,-1,-1):
                    y_right_this = arr_y_right_ori[counter]
                    x_right_this = arr_x_right_ori[counter]
                    flag_no_change = 0
                    for offset in range(6):
                        if y_right_this - offset > 0 and y_right_this + offset < 960:
                            if seg_im[x_right_this,y_right_this-offset,0] == rgb_rail or seg_im[x_right_this,y_right_this+offset,0] == rgb_rail:                             #or seg_im[x_right_this,y_left_this-offset,0] == rgb_rail or seg_im[x_right_this,y_left_this+offset,0] == rgb_rail:
                                flag_no_change = 1
                                break
                    if flag_no_change == 0:
                        for offset in range(6,50):
                            if y_right_this - offset > 0 and y_right_this + offset < 960:
                                if seg_im[x_right_this, y_right_this - offset, 0] == rgb_rail and seg_im[x_right_this, y_right_this + offset, 0] != rgb_rail and abs(y_right_this - offset - arr_y_right_ori[counter+1])<10:
                                    arr_y_right_ori[counter] = y_right_this - offset
                                    break
                                elif seg_im[x_right_this, y_right_this - offset, 0] != rgb_rail and seg_im[x_right_this, y_right_this + offset, 0] == rgb_rail and abs(y_right_this + offset - arr_y_right_ori[counter+1])<10:
                                    arr_y_right_ori[counter] = y_right_this + offset
                                    break



            x_start = min(arr_x_left_ori[-1],arr_x_right_ori[-1])
            x_end   = max(arr_x_left_ori[0],arr_x_right_ori[0])

            min_residuals = 100000000
            for param_deg_poly in range(1,4):
                coeff_poly_left = np.polyfit(arr_x_left_ori, arr_y_left_ori, param_deg_poly, full=True)
                residual_this   = coeff_poly_left[1][0]
                if residual_this <= min_residuals:
                    min_residuals = residual_this
                    coeff_this = coeff_poly_left[0]
                    poly_this = np.poly1d(coeff_this)
            # sample_arr_x_left_new = np.linspace(arr_x_left_ori[-1], arr_x_left_ori[0], arr_x_left_ori[0]-arr_x_left_ori[-1]+1)
            sample_arr_x_left_new = np.linspace(x_start, x_end, x_end - x_start + 1)
            sample_arr_y_left_new = poly_this(sample_arr_x_left_new)
            sample_arr_xyz_left_ori_ = np.vstack((sample_arr_y_left_new, sample_arr_x_left_new))
            sample_arr_xyz_left_ori = sample_arr_xyz_left_ori_.T

            min_residuals = 100000000
            for param_deg_poly in range(1,4):
                coeff_poly_right = np.polyfit(arr_x_right_ori, arr_y_right_ori, param_deg_poly, full=True)
                residual_this   = coeff_poly_right[1][0]
                if residual_this <= min_residuals:
                    min_residuals = residual_this
                    coeff_this = coeff_poly_right[0]
                    poly_this = np.poly1d(coeff_this)
            # sample_arr_x_right_new = np.linspace(arr_x_right_ori[-1], arr_x_right_ori[0], arr_x_right_ori[0]-arr_x_right_ori[-1]+1)
            sample_arr_x_right_new = np.linspace(x_start, x_end, x_end - x_start + 1)
            sample_arr_y_right_new = poly_this(sample_arr_x_right_new)
            sample_arr_xyz_right_ori_ = np.vstack((sample_arr_y_right_new, sample_arr_x_right_new))
            sample_arr_xyz_right_ori = sample_arr_xyz_right_ori_.T


            dict_path_poly_this = {
                                   "xy_left"             : sample_arr_xyz_left_ori,
                                   "xy_right"            : sample_arr_xyz_right_ori,
                                   "coeff_poly_left"     : coeff_poly_left,
                                   "coeff_poly_right"    : coeff_poly_right}

            ###---------------------------------------------------------------------------------------
            ### store
            ###---------------------------------------------------------------------------------------
            dict_path_final = {"extracted": dict_path_this,
                               "polynomial": dict_path_poly_this}

            list_paths_out.append(dict_path_final)
        #end


        if 0:
            self._visualize_debug_final_path_ipm(list_paths_out)
        #end

        return list_paths_out
    # end
    #---------------------------------------------------------------------------------------------------
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
    #---------------------------------------------------------------------------------------------------


    ###=========================================================================================================
    ### MyUtils_RailPathGraph::
    ###=========================================================================================================
    def _visualize_debug_final_path_ipm(self, list_paths_final):
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
        #   "polynomial": dict_path_poly_this = {"xyz_cen_3d": arr_xyz_sample_ori,
        #                                        "xyz_left_3d": [],
        #                                        "xyz_right_3d": []}
        #---------------------------------------------------------------------------------------------

        ###
        assert (self.m_obj_utils_3D is not None)
        assert (self.m_img_raw_rsz_uint8 is not None)


        ###------------------------------------------------------------------------------------------------
        ### create img for visualization
        ###------------------------------------------------------------------------------------------------
        img_vis_ipm_rgb   = self.m_obj_utils_3D.create_img_IPM(self.m_img_raw_rsz_uint8)
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
        h_img_temp = self.m_img_raw_rsz_uint8.shape[0]
        w_img_temp = self.m_img_raw_rsz_uint8.shape[1]

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

            arr_xyz_cen = dict_polynomial["xyz_cen_3d"]
            arr_xyz_left = dict_polynomial["xyz_left_3d"]
            arr_xyz_right = dict_polynomial["xyz_right_3d"]


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
        cv2.imshow('visualize_debug_final_path_ipm', img_vis_ipm_gray3)
        cv2.waitKey(1)


        ### save (temp)
        fname_output_temp = "/home/yu1/Desktop/dir_temp/temp_res0/final_path_ipm/" + "final_path_ipm_" + str(self.m_temp_idx_img_final_path_ipm) + '.jpg'
        cv2.imwrite(fname_output_temp, img_vis_ipm_gray3)
        self.m_temp_idx_img_final_path_ipm += 1

    #end



    ###=========================================================================================================
    ### MyUtils_RailPathGraph::
    ###=========================================================================================================
    def _visualize_debug_edge_ori(self, list_dict_edge, max_id_edge_sofar, list_dict_all_sections):

        # dict_ref = {"id_edge": id_edge_for_this_subedge_ref,
        #             "list_dict_this_subedge": list_dict_this_subedge}
        #
        #           "subedges" has
        #               dict_pnt = {"id_subedge": id_cluster_for_this_pnt_ref,
        #                           "info_pnt": dict_pnt_this}

        ###
        assert (self.m_obj_utils_3D is not None)
        assert (self.m_img_raw_rsz_uint8 is not None)


        ###------------------------------------------------------------------------------------------------
        ### create img for visualization
        ###------------------------------------------------------------------------------------------------
        img_vis_gray1 = cv2.cvtColor(self.m_img_raw_rsz_uint8, cv2.COLOR_BGR2GRAY)

        img_vis_gray3 = np.zeros_like(self.m_img_raw_rsz_uint8)  # img_ipm_gray3: 3-ch gray img
        img_vis_gray3[:, :, 0] = img_vis_gray1
        img_vis_gray3[:, :, 1] = img_vis_gray1
        img_vis_gray3[:, :, 2] = img_vis_gray1


        ###------------------------------------------------------------------------------------------------
        ### create supplementary img for visualization
        ###------------------------------------------------------------------------------------------------
        self._create_img_supp_ori_for_visualization(img_vis_gray3, list_dict_all_sections)
            # completed to set
            #       self.m_img_supp_ori_for_vis

        ###
        alpha = 0.9
        img_vis_final = cv2.addWeighted(src1=img_vis_gray3, alpha=alpha, src2=self.m_img_supp_ori_for_vis,
                                        beta=(1.0 - alpha), gamma=0)


        ###------------------------------------------------------------------------------------------------
        ### process each section
        ###------------------------------------------------------------------------------------------------
        for id_edge in range(max_id_edge_sofar + 1):
            list_edge_this = list_dict_edge[id_edge]

            val_r = int(self.m_rgb_cluster[id_edge, 0])
            val_g = int(self.m_rgb_cluster[id_edge, 1])
            val_b = int(self.m_rgb_cluster[id_edge, 2])

            for idx_subedge in range(len(list_edge_this)):
                list_pnts = list_edge_this[idx_subedge]["list_dict_this_subedge"]

                for idx_pnts in range(len(list_pnts)):
                    xy_cen_img = list_pnts[idx_pnts]["info_pnt"]["xy_cen_img"]

                    cv2.circle(img_vis_final, center=(xy_cen_img[0], xy_cen_img[1]),
                               radius=3, color=(val_b, val_g, val_r), thickness=-1)
                #end
            #end
        #end


        ###------------------------------------------------------------------------------------------------
        ### show
        ###------------------------------------------------------------------------------------------------
        cv2.imshow('visualize_debug1_ori', img_vis_final)
        cv2.waitKey(1)


        ### save (temp)
        fname_output_temp = "/home/yu1/Desktop/dir_temp/temp_res0/debug1_ori/" + "debug1_ori_" + str(self.m_temp_idx_img_debug1_ori) + '.jpg'
        cv2.imwrite(fname_output_temp, img_vis_final)
        self.m_temp_idx_img_debug1_ori += 1
    #end


    ###=========================================================================================================
    ### MyUtils_RailPathGraph::
    ###=========================================================================================================
    def _visualize_debug_edge_ipm(self, list_dict_edge, max_id_edge_sofar, list_dict_all_sections):

        # dict_ref = {"id_edge": id_edge_for_this_subedge_ref,
        #             "list_dict_this_subedge": list_dict_this_subedge}
        #
        #           "subedges" has
        #               dict_pnt = {"id_subedge": id_cluster_for_this_pnt_ref,
        #                           "info_pnt": dict_pnt_this}

        ###
        assert (self.m_obj_utils_3D is not None)
        assert (self.m_img_raw_rsz_uint8 is not None)


        ###------------------------------------------------------------------------------------------------
        ### create img for visualization
        ###------------------------------------------------------------------------------------------------
        img_vis_ipm_rgb   = self.m_obj_utils_3D.create_img_IPM(self.m_img_raw_rsz_uint8)
        img_vis_ipm_gray1 = cv2.cvtColor(img_vis_ipm_rgb, cv2.COLOR_BGR2GRAY)

        img_vis_ipm_gray3 = np.zeros_like(img_vis_ipm_rgb)  # img_ipm_gray3: 3-ch gray img
        img_vis_ipm_gray3[:, :, 0] = img_vis_ipm_gray1
        img_vis_ipm_gray3[:, :, 1] = img_vis_ipm_gray1
        img_vis_ipm_gray3[:, :, 2] = img_vis_ipm_gray1

        ###
        h_img_bev, w_img_bev = self.m_obj_utils_3D.get_size_img_bev()


        ###------------------------------------------------------------------------------------------------
        ### create supplementary img for visualization
        ###------------------------------------------------------------------------------------------------
        self._create_img_supp_ipm_for_visualization(img_vis_ipm_gray3, list_dict_all_sections, h_img_bev, w_img_bev)
            # completed to set
            #       self.m_img_supp_for_vis

        ###
        alpha = 0.9
        img_vis_final = cv2.addWeighted(src1=img_vis_ipm_gray3, alpha=alpha, src2=self.m_img_supp_ipm_for_vis,
                                        beta=(1.0 - alpha), gamma=0)


        ###------------------------------------------------------------------------------------------------
        ### process each section
        ###------------------------------------------------------------------------------------------------
        for id_edge in range(max_id_edge_sofar + 1):
            list_edge_this = list_dict_edge[id_edge]

            val_r = int(self.m_rgb_cluster[id_edge, 0])
            val_g = int(self.m_rgb_cluster[id_edge, 1])
            val_b = int(self.m_rgb_cluster[id_edge, 2])

            for idx_subedge in range(len(list_edge_this)):
                list_pnts = list_edge_this[idx_subedge]["list_dict_this_subedge"]

                for idx_pnts in range(len(list_pnts)):
                    xy_cen_img = list_pnts[idx_pnts]["info_pnt"]["xy_cen_img"]

                    x_cen_bev, y_cen_bev = self.m_obj_utils_3D.convert_pnt_img_ori_to_pnt_bev( np.array([[xy_cen_img[0]], [xy_cen_img[1]], [1.0]]) )

                    x_cen_bev_int = int(round(x_cen_bev))
                    y_cen_bev_int = int(round(y_cen_bev))


                    ### draw pnts
                    if (0 <= x_cen_bev_int) and (x_cen_bev_int < w_img_bev) and (0 <= y_cen_bev_int) and (y_cen_bev_int < h_img_bev):
                        cv2.circle(img_vis_final, center=(x_cen_bev_int, y_cen_bev_int),
                                   radius=3, color=(val_b, val_g, val_r), thickness=-1)
                    #end
                #end
            #end
        #end


        ###------------------------------------------------------------------------------------------------
        ### show
        ###------------------------------------------------------------------------------------------------
        cv2.imshow('visualize_debug1_ipm', img_vis_final)
        cv2.waitKey(1)


        ### save (temp)
        fname_output_temp = "/home/yu1/Desktop/dir_temp/temp_res0/debug1_ipm/" + "debug1_ipm_" + str(self.m_temp_idx_img_debug1_ipm) + '.jpg'
        cv2.imwrite(fname_output_temp, img_vis_final)
        self.m_temp_idx_img_debug1_ipm += 1

    #end



    ###=========================================================================================================
    ### MyUtils_RailPathGraph::_create_img_bg_for_visualization()
    ###=========================================================================================================
    def _create_img_supp_ori_for_visualization(self, img_vis_gray3, list_dict_all_sections):

        ###
        if self.m_img_supp_ori_for_vis is not None:     # to create m_img_supp_ori_for_vis only once during running
            return
        #end


        ###
        self.m_img_supp_ori_for_vis = np.zeros_like(img_vis_gray3)


        ###
        w_img = img_vis_gray3.shape[1]
        totnum_sections = len(list_dict_all_sections)

        for id_section in range(totnum_sections):
            ###
            y_min = list_dict_all_sections[id_section]["y_min"]
            y_max = list_dict_all_sections[id_section]["y_max"]

            ### mark this region
            if id_section % 2 == 0:
                for y in range(y_min, y_max + 1):
                    for x in range(0, w_img):
                        self.m_img_supp_ori_for_vis[y, x, :] = (255, 255, 255)
                    #end
                #end
            #end
        #end
        # completed to set
        #       self.m_img_supp_for_vis

    #end


    ###=========================================================================================================
    ### MyUtils_RailPathGraph::_create_img_ipm_supp_for_visualization()
    ###=========================================================================================================
    def _create_img_supp_ipm_for_visualization(self, img_vis_ipm_gray3, list_dict_all_sections, h_img_bev, w_img_bev):

        ###
        if self.m_img_supp_ipm_for_vis is not None:     # to create m_img_supp_ipm_for_vis only once during running
            return
        #end


        ###
        self.m_img_supp_ipm_for_vis = np.zeros_like(img_vis_ipm_gray3)


        ###
        totnum_sections = len(list_dict_all_sections)

        for id_section in range(totnum_sections):
            ###
            y_min_ori = list_dict_all_sections[id_section]["y_min"]
            y_max_ori = list_dict_all_sections[id_section]["y_max"]
            x_dum_ori = int(w_img_bev/2)

            ###
            x_dum_min_bev, y_min_bev = self.m_obj_utils_3D.convert_pnt_img_ori_to_pnt_bev( np.array([[x_dum_ori], [y_min_ori], [1.0]]) )
            x_dum_max_bev, y_max_bev = self.m_obj_utils_3D.convert_pnt_img_ori_to_pnt_bev( np.array([[x_dum_ori], [y_max_ori], [1.0]]) )

            y_min_bev_int = int(round(y_min_bev))
            y_max_bev_int = int(round(y_max_bev))


            ### mark this region
            if id_section % 2 == 0:
                for y in range(y_min_bev_int, y_max_bev_int + 1):
                    for x in range(0, w_img_bev):
                        if (0 <= x) and (x < w_img_bev) and (0 <= y) and (y < h_img_bev):
                            self.m_img_supp_ipm_for_vis[y, x, :] = (255, 255, 255)
                        #end
                    #end
                #end
            #end
        #end
        # completed to set
        #       self.m_img_supp_ipm_for_vis

    #end


    ###=========================================================================================================
    ### MyUtils_RailPathGraph::_visualize_subedges_in_all sections()
    ###=========================================================================================================
    def _visualize_debug_subedges_in_all_sections_ori(self, list_dict_all_sections):
        """

        :param list_dict_all_sections:
        :return:
        """

        #------------------------------------------------------------------------------------------------
        # list_dict_all_sections[], having each list of
        #       dict_one_section = {"id_section": id_section,
        #                     "y_min": y_min,
        #                     "y_max": y_max,
        #                     "subedges": None}
        #       where,
        #           "subedges" has
        #               dict_pnt = {"id_subedge": id_cluster_for_this_pnt_ref,
        #               "info_pnt": dict_pnt_this}
        #
        #       where,
        #               dict_pnt_this = {"centerness": c_this,
        #                                "xy_cen_img": [x_cen, y_cen],
        #                                "xy_left_img": [x_left, y_cen],
        #                                "xy_right_img": [x_right, y_cen],
        #                                "xyz_cen_3d": [x_cen_3d, y_cen_3d, z_cen_3d],
        #                                "xyz_left_3d": [x_left_3d, y_left_3d, z_left_3d],
        #                                "xyz_right_3d": [x_right_3d, y_right_3d, z_right_3d]}
        #------------------------------------------------------------------------------------------------


        ###
        assert (self.m_obj_utils_3D is not None)
        assert (self.m_img_raw_rsz_uint8 is not None)


        ###------------------------------------------------------------------------------------------------
        ### create img for visualization
        ###------------------------------------------------------------------------------------------------
        img_vis_gray1 = cv2.cvtColor(self.m_img_raw_rsz_uint8, cv2.COLOR_BGR2GRAY)

        img_vis_gray3 = np.zeros_like(self.m_img_raw_rsz_uint8)  # img_ipm_gray3: 3-ch gray img
        img_vis_gray3[:, :, 0] = img_vis_gray1
        img_vis_gray3[:, :, 1] = img_vis_gray1
        img_vis_gray3[:, :, 2] = img_vis_gray1


        ###------------------------------------------------------------------------------------------------
        ### create supplementary img for visualization
        ###------------------------------------------------------------------------------------------------
        self._create_img_supp_ori_for_visualization(img_vis_gray3, list_dict_all_sections)
            # completed to set
            #       self.m_img_supp_ori_for_vis

        ###
        alpha = 0.9
        img_vis_final = cv2.addWeighted(src1=img_vis_gray3, alpha=alpha, src2=self.m_img_supp_ori_for_vis,
                                        beta=(1.0 - alpha), gamma=0)


        ###------------------------------------------------------------------------------------------------
        ### process each section
        ###------------------------------------------------------------------------------------------------
        totnum_sections = len(list_dict_all_sections)

        for id_section in range(totnum_sections):
            ###
            subedges = list_dict_all_sections[id_section]["subedges"]

            ###
            if subedges is None:
                continue
            #end

            ###
            for id_cluster in range(len(subedges)):   # len(subedges): totnum_clusters_in_this_section
                list_dict_pnts_this_cluster = subedges[id_cluster]

                val_r = int(self.m_rgb_cluster[id_cluster, 0])
                val_g = int(self.m_rgb_cluster[id_cluster, 1])
                val_b = int(self.m_rgb_cluster[id_cluster, 2])

                for idx_pnt in range(len(list_dict_pnts_this_cluster)):   # len(list_dict_pnts_this_cluster): totnum_pnts_in_this_cluster
                    dict_pnt_this_cluster = list_dict_pnts_this_cluster[idx_pnt]["info_pnt"]
                    xy_cen_img = dict_pnt_this_cluster["xy_cen_img"]

                    cv2.circle(img_vis_final, center=(xy_cen_img[0], xy_cen_img[1]),
                               radius=3, color=(val_b, val_g, val_r), thickness=-1)

                #end
            #end
        #end


        ###------------------------------------------------------------------------------------------------
        ### show
        ###------------------------------------------------------------------------------------------------
        cv2.imshow('img_vis_subedges_ori', img_vis_final)
        cv2.waitKey(0)


        ### save (temp)
        # fname_output_temp = "/home/yu1/Desktop/dir_temp/temp_res0/subedge_ori/" + "subedge_ori_" + str(self.m_temp_idx_img_vis_ori) + '.jpg'
        # cv2.imwrite(fname_output_temp, img_vis_final)
        # self.m_temp_idx_img_vis_ori += 1


    #end


    ###=========================================================================================================
    ### MyUtils_RailPathGraph::_visualize_subedges_in_all sections()
    ###=========================================================================================================
    def _visualize_debug_subedges_in_all_sections_ipm(self, list_dict_all_sections):
        """

        :param list_dict_all_sections:
        :return:
        """

        #------------------------------------------------------------------------------------------------
        # list_dict_all_sections[], having each list of
        #       dict_one_section = {"id_section": id_section,
        #                     "y_min": y_min,
        #                     "y_max": y_max,
        #                     "subedges": None}
        #       where,
        #           "subedges" has
        #               dict_pnt = {"id_subedge": id_cluster_for_this_pnt_ref,
        #               "info_pnt": dict_pnt_this}
        #
        #       where,
        #               dict_pnt_this = {"centerness": c_this,
        #                                "xy_cen_img": [x_cen, y_cen],
        #                                "xy_left_img": [x_left, y_cen],
        #                                "xy_right_img": [x_right, y_cen],
        #                                "xyz_cen_3d": [x_cen_3d, y_cen_3d, z_cen_3d],
        #                                "xyz_left_3d": [x_left_3d, y_left_3d, z_left_3d],
        #                                "xyz_right_3d": [x_right_3d, y_right_3d, z_right_3d]}
        #------------------------------------------------------------------------------------------------

        ###
        assert (self.m_obj_utils_3D is not None)
        assert (self.m_img_raw_rsz_uint8 is not None)


        ###------------------------------------------------------------------------------------------------
        ### create img for visualization
        ###------------------------------------------------------------------------------------------------
        img_vis_ipm_rgb   = self.m_obj_utils_3D.create_img_IPM(self.m_img_raw_rsz_uint8)
        img_vis_ipm_gray1 = cv2.cvtColor(img_vis_ipm_rgb, cv2.COLOR_BGR2GRAY)

        img_vis_ipm_gray3 = np.zeros_like(img_vis_ipm_rgb)  # img_ipm_gray3: 3-ch gray img
        img_vis_ipm_gray3[:, :, 0] = img_vis_ipm_gray1
        img_vis_ipm_gray3[:, :, 1] = img_vis_ipm_gray1
        img_vis_ipm_gray3[:, :, 2] = img_vis_ipm_gray1

        ###
        h_img_bev, w_img_bev = self.m_obj_utils_3D.get_size_img_bev()


        ###------------------------------------------------------------------------------------------------
        ### create supplementary img for visualization
        ###------------------------------------------------------------------------------------------------
        self._create_img_supp_ipm_for_visualization(img_vis_ipm_gray3, list_dict_all_sections, h_img_bev, w_img_bev)
            # completed to set
            #       self.m_img_supp_for_vis

        ###
        alpha = 0.9
        img_vis_final = cv2.addWeighted(src1=img_vis_ipm_gray3, alpha=alpha, src2=self.m_img_supp_ipm_for_vis,
                                        beta=(1.0 - alpha), gamma=0)


        ###------------------------------------------------------------------------------------------------
        ### process each section
        ###------------------------------------------------------------------------------------------------
        totnum_sections = len(list_dict_all_sections)

        for id_section in range(totnum_sections):
            ###
            subedges = list_dict_all_sections[id_section]["subedges"]

            ###
            if subedges is None:
                continue
            #end

            ###
            for id_cluster in range(len(subedges)):   # len(subedges): totnum_clusters_in_this_section
                list_dict_pnts_this_cluster = subedges[id_cluster]

                val_r = int(self.m_rgb_cluster[id_cluster, 0])
                val_g = int(self.m_rgb_cluster[id_cluster, 1])
                val_b = int(self.m_rgb_cluster[id_cluster, 2])

                for idx_pnt in range(len(list_dict_pnts_this_cluster)):   # len(list_dict_pnts_this_cluster): totnum_pnts_in_this_cluster
                    dict_pnt_this_cluster = list_dict_pnts_this_cluster[idx_pnt]["info_pnt"]
                    xy_cen_img = dict_pnt_this_cluster["xy_cen_img"]

                    x_cen_bev, y_cen_bev = self.m_obj_utils_3D.convert_pnt_img_ori_to_pnt_bev( np.array([[xy_cen_img[0]], [xy_cen_img[1]], [1.0]]) )

                    x_cen_bev_int = int(round(x_cen_bev))
                    y_cen_bev_int = int(round(y_cen_bev))


                    ### draw pnts
                    if (0 <= x_cen_bev_int) and (x_cen_bev_int < w_img_bev) and (0 <= y_cen_bev_int) and (y_cen_bev_int < h_img_bev):
                        cv2.circle(img_vis_final, center=(x_cen_bev_int, y_cen_bev_int),
                                   radius=3, color=(val_b, val_g, val_r), thickness=-1)
                    #end
                #end
            #end
        #end


        ###------------------------------------------------------------------------------------------------
        ### show
        ###------------------------------------------------------------------------------------------------
        cv2.imshow('img_vis_subedges_ipm', img_vis_final)
        cv2.waitKey(1)


        ### save (temp)
        fname_output_temp = "/home/yu1/Desktop/dir_temp/temp_res0/subedge_ipm/" + "subedge_ipm_" + str(self.m_temp_idx_img_vis_ori) + '.jpg'
        cv2.imwrite(fname_output_temp, img_vis_final)
        self.m_temp_idx_img_vis_ipm += 1

    #end



    ###=========================================================================================================
    ### MyUtils_RailPathGraph::process_a()
    ###=========================================================================================================
    # def process_old(self, list_triplet_points_local_max, obj_utils_3D, img_raw_rsz_uint8):
    #     """
    #     - create cluster-graph
    #     - decode paths from cluster-graph
    #     :return:
    #     """
    #
    #     ###
    #     self.m_obj_utils_3D = obj_utils_3D
    #     self.m_img_raw_rsz_uint8 = img_raw_rsz_uint8
    #
    #     ###
    #     h_img = img_raw_rsz_uint8.shape[0]
    #     w_img = img_raw_rsz_uint8.shape[1]
    #
    #     ###
    #     list_dict_pnt_cluster, \
    #     max_id_cluster_sofar = self._obsolete_create_clusters_initial(list_triplet_points_local_max, h_img, w_img)
    #
    #     ###
    #     self._obsolete_fit_polynomial(list_dict_pnt_cluster, max_id_cluster_sofar)
    #
    # #end


    ###=========================================================================================================
    ### MyUtils_RailPathGraph::create_graph_cluster_initial()
    ###=========================================================================================================
    # def _obsolete_create_clusters_initial(self, list_triplet_points_local_max, h_img, w_img):
    #     """
    #
    #     :param list_triplet_points_local_max:
    #     :param h_img:
    #     :param w_img:
    #     :return:
    #     """
    #     # ------------------------------------------------------------------------------------------------
    #     # Please, see extract_triplet_localmax() in <my_utils_img.py>
    #     #   to see the definition of list_triplet_points_local_max.
    #     # ------------------------------------------------------------------------------------------------
    #     # dict_pnt_this = {"centerness": c_this,
    #     #                  "xy_cen_img": [x_cen, y_cen],
    #     #                  "xy_left_img": [x_left, y_cen],
    #     #                  "xy_right_img": [x_right, y_cen],
    #     #                  "xyz_cen_3d": [x_cen_3d, y_cen_3d, z_cen_3d],
    #     #                  "xyz_left_3d": [x_left_3d, y_left_3d, z_left_3d],
    #     #                  "xyz_right_3d": [x_right_3d, y_right_3d, z_right_3d]}
    #     # ------------------------------------------------------------------------------------------------
    #     # <<similarity for initial clusters>>
    #     #   - small dx (in img and 3d) between a pair of pixels
    #     #   - small dy (in img) between a pair of pixels
    #     # ------------------------------------------------------------------------------------------------
    #     # h_img = len(list_triplet_points_local_max)
    #     # ------------------------------------------------------------------------------------------------
    #     # note that
    #     #   this is 1(pnt_ref) <-> n(pnt_this) matching.
    #     #   so, pnts at a certain row can be assigned the identical pnt_ref.
    #     # ------------------------------------------------------------------------------------------------
    #
    #     ###
    #     param_thres_dx_3d = 0.5
    #     param_thres_dy_img = 10
    #
    #     ###
    #     max_id_cluster_sofar = -1
    #     list_dict_pnt_cluster = [[] for _ in range(w_img)]  # can use any large number, instead of w_img
    #     list_dict_pnt_ref = [None] * w_img  # can use any large number, instead of w_img
    #     # list_dict_pnt_ref: having dicts of
    #     #       dict_pnt_ref = {"id_cluster": id_cluster_for_this_pnt_ref,
    #     #                       "info_pnt": dict_pnt_this}
    #
    #     ### sweep from near to far
    #     for y in range(h_img - 1, -1, -1):
    #
    #         ### get pnt at this row
    #         list_thisrow = list_triplet_points_local_max[y]
    #
    #         if len(list_thisrow) == 0:
    #             continue
    #         # end
    #
    #         ### process for each pnt at this row
    #         for idx_dict_pnt_this, dict_pnt_this in enumerate(list_thisrow):
    #             ###---------------------------------------------------------------------------------
    #             ### get info of this pnt
    #             ###---------------------------------------------------------------------------------
    #             xy_cen_img_pnt_this = dict_pnt_this["xy_cen_img"]
    #             xyz_cen_3d_pnt_this = dict_pnt_this["xyz_cen_3d"]
    #
    #             ###---------------------------------------------------------------------------------
    #             ### check if this pnt belongs to an existing cluster or newly appeared pnt
    #             ###---------------------------------------------------------------------------------
    #             b_new_pnt = True
    #             id_cluster_for_pnt_this = None
    #
    #             ###
    #             if max_id_cluster_sofar == -1:
    #                 pass
    #             else:  # max_id_cluster_sofar >= 0
    #
    #                 ### find existing cluster for min
    #                 dx_img_min = w_img * 1000
    #                 dy_img_for_min = None
    #                 dx_3d_for_min = None
    #
    #                 for id_cluster_existing in range(0, max_id_cluster_sofar + 1):
    #                     ###
    #                     dict_pnt_ref = list_dict_pnt_ref[id_cluster_existing]["info_pnt"]
    #                     xy_cen_img_pnt_ref = dict_pnt_ref["xy_cen_img"]
    #                     xyz_cen_3d_pnt_ref = dict_pnt_ref["xyz_cen_3d"]
    #
    #                     ###
    #                     dx_img = abs(xy_cen_img_pnt_this[0] - xy_cen_img_pnt_ref[0])
    #                     dy_img = abs(xy_cen_img_pnt_this[1] - xy_cen_img_pnt_ref[1])
    #                     dx_3d = abs(xyz_cen_3d_pnt_this[0] - xyz_cen_3d_pnt_ref[0])
    #
    #                     ###
    #                     if dx_img <= dx_img_min:
    #                         id_cluster_for_pnt_this = id_cluster_existing
    #
    #                         dx_img_min = dx_img
    #                         dy_img_for_min = dy_img
    #                         dx_3d_for_min = dx_3d
    #                     # end
    #                 # end
    #                 # completed to set
    #                 #   id_cluster_for_pnt_this
    #                 #   dx_img_min
    #                 #   dy_img_for_min
    #                 #   dx_3d_for_min
    #
    #                 ### determine if this pnt is new/existing
    #                 # param_thres_dx_3d: e.g. 0.5
    #                 # param_thres_dy_img: 10
    #
    #                 if dx_3d_for_min < param_thres_dx_3d and dy_img_for_min < param_thres_dy_img:
    #                     b_new_pnt = False
    #                 else:
    #                     b_new_pnt = True
    #                 # end
    #             # end
    #
    #             ###---------------------------------------------------------------------------------
    #             ### update
    #             ###---------------------------------------------------------------------------------
    #             # <if is a newly appeared pnt>
    #             #   set this pnt as a reference pnt
    #             #   update reference pnt info
    #             # ---------------------------------------------------------------------
    #             # <if it is an existing cluster>
    #             #   assign existing cluster-id
    #             #   update reference pnt info
    #             # ---------------------------------------------------------------------
    #             id_cluster_for_this_pnt_ref = -1
    #
    #             ### set id_cluster_for_this_pnt_ref
    #             if b_new_pnt is True:
    #                 id_cluster_for_this_pnt_ref = max_id_cluster_sofar + 1
    #                 max_id_cluster_sofar = id_cluster_for_this_pnt_ref
    #             else:
    #                 id_cluster_for_this_pnt_ref = id_cluster_for_pnt_this
    #             # end
    #
    #             ### set updated info for this_pnt_ref
    #             dict_pnt_ref = {"id_cluster": id_cluster_for_this_pnt_ref,
    #                             "info_pnt": dict_pnt_this}
    #
    #             ### update (this_pnt_ref)
    #             list_dict_pnt_ref[id_cluster_for_this_pnt_ref] = dict_pnt_ref
    #
    #             ### update
    #             list_dict_pnt_cluster[id_cluster_for_this_pnt_ref].append(dict_pnt_ref)
    #         # end
    #     # end
    #     # completed to set
    #     #   list_dict_pnt_cluster
    #     #   max_id_cluster_sofar
    #
    #     ###----------------------------------------------------------------------------------------------
    #     ### <debugging>>
    #     ###----------------------------------------------------------------------------------------------
    #     # code from extract_triplet_pnts_localmax() in <my_utils_img.py>
    #     if 1:
    #         ###
    #         assert(self.m_obj_utils_3D is not None)
    #         assert(self.m_img_raw_rsz_uint8 is not None)
    #
    #
    #         ### create bg img
    #         img_ipm_rgb = self.m_obj_utils_3D.create_img_IPM(self.m_img_raw_rsz_uint8)
    #         img_ipm_gray1 = cv2.cvtColor(img_ipm_rgb, cv2.COLOR_BGR2GRAY)
    #
    #         img_ipm_gray3 = np.zeros_like(img_ipm_rgb)  # img_ipm_gray3: 3-ch gray img
    #         img_ipm_gray3[:, :, 0] = img_ipm_gray1
    #         img_ipm_gray3[:, :, 1] = img_ipm_gray1
    #         img_ipm_gray3[:, :, 2] = img_ipm_gray1
    #
    #         ###
    #         h_img_bev, w_img_bev = self.m_obj_utils_3D.get_size_img_bev()
    #
    #         ###
    #         for id_cluster in range(0, max_id_cluster_sofar + 1):
    #
    #             ###
    #             list_dict_pnt_cluster_this = list_dict_pnt_cluster[id_cluster]
    #
    #             ###
    #             val_r = int(self.m_rgb_cluster[id_cluster, 0])
    #             val_g = int(self.m_rgb_cluster[id_cluster, 1])
    #             val_b = int(self.m_rgb_cluster[id_cluster, 2])
    #
    #             ###
    #             for idx_dict_this, dict_this in enumerate(list_dict_pnt_cluster_this):
    #                 xy_cen_img = dict_this["info_pnt"]["xy_cen_img"]
    #
    #                 x_cen_bev, y_cen_bev = self.m_obj_utils_3D.convert_pnt_img_ori_to_pnt_bev(
    #                     np.array([[xy_cen_img[0]], [xy_cen_img[1]], [1.0]]))
    #
    #                 x_cen_bev_int = int(round(x_cen_bev))
    #                 y_cen_bev_int = int(round(y_cen_bev))
    #
    #                 ### draw pnts
    #                 if (0 <= x_cen_bev_int) and (x_cen_bev_int < w_img_bev) and (0 <= y_cen_bev_int) and (
    #                         y_cen_bev_int < h_img_bev):
    #                     cv2.circle(img_ipm_gray3, center=(x_cen_bev_int, y_cen_bev_int), radius=3,
    #                                color=(val_b, val_g, val_r), thickness=-1)
    #                 # end
    #             # end
    #         # end
    #
    #         cv2.imshow('img_ipm_cluster', img_ipm_gray3)
    #         cv2.waitKey(1)
    #     # end
    #
    #     return list_dict_pnt_cluster, max_id_cluster_sofar
    # #end


    ###=========================================================================================================
    ### MyUtils_RailPathGraph::
    ###=========================================================================================================
    # def _obsolete_fit_polynomial(self, list_dict_pnt_cluster, max_id_cluster_sofar):
    #
    #     ###----------------------------------------------------------------------------------------------
    #     ### fit polynomial
    #     ###----------------------------------------------------------------------------------------------
    #     list_poly = [None]*(max_id_cluster_sofar + 1)
    #
    #
    #     ###
    #     for id_cluster in range(0, max_id_cluster_sofar + 1):
    #
    #         ###
    #         list_dict_pnt_cluster_this = list_dict_pnt_cluster[id_cluster]
    #         totnum_pnts = len(list_dict_pnt_cluster_this)
    #
    #         if totnum_pnts <= 3:
    #             continue
    #         #end
    #
    #
    #         ###
    #         arr_x_3d = np.zeros(shape=(totnum_pnts,))
    #         arr_y_3d = np.zeros(shape=(totnum_pnts,))
    #
    #         for idx_dict_this, dict_this in enumerate(list_dict_pnt_cluster_this):
    #             xyz_cen_3d = dict_this["info_pnt"]["xyz_cen_3d"]
    #
    #             arr_x_3d[idx_dict_this] = xyz_cen_3d[0]
    #             arr_y_3d[idx_dict_this] = xyz_cen_3d[1]
    #         # end
    #             # completed to set
    #             #   arr_x_3d, arr_y_3d
    #
    #
    #         ###
    #         arr_x_3d_new = arr_y_3d
    #         arr_y_3d_new = -1.0*arr_x_3d
    #
    #         coeff_poly = np.polyfit(arr_x_3d_new, arr_y_3d_new, 2)
    #         poly_this = np.poly1d(coeff_poly)
    #
    #         ###
    #         list_poly[id_cluster] = poly_this
    #     #end
    #         # completed to set
    #         #       list_poly[]
    #
    #
    #     ###----------------------------------------------------------------------------------------------
    #     ### <debugging>>
    #     ###----------------------------------------------------------------------------------------------
    #     # code from extract_triplet_pnts_localmax() in <my_utils_img.py>
    #     if 1:
    #         ###
    #         assert(self.m_obj_utils_3D is not None)
    #         assert(self.m_img_raw_rsz_uint8 is not None)
    #
    #         ### create bg img
    #         img_ipm_rgb = self.m_obj_utils_3D.create_img_IPM(self.m_img_raw_rsz_uint8)
    #         img_ipm_gray1 = cv2.cvtColor(img_ipm_rgb, cv2.COLOR_BGR2GRAY)
    #
    #         img_ipm_gray3 = np.zeros_like(img_ipm_rgb)  # img_ipm_gray3: 3-ch gray img
    #         img_ipm_gray3[:, :, 0] = img_ipm_gray1
    #         img_ipm_gray3[:, :, 1] = img_ipm_gray1
    #         img_ipm_gray3[:, :, 2] = img_ipm_gray1
    #
    #         ###
    #         h_img_bev, w_img_bev = self.m_obj_utils_3D.get_size_img_bev()
    #
    #         ###
    #         for id_cluster in range(0, max_id_cluster_sofar + 1):
    #
    #             ###
    #             list_dict_pnt_cluster_this = list_dict_pnt_cluster[id_cluster]
    #
    #             ###
    #             val_r = int(self.m_rgb_cluster[id_cluster, 0])
    #             val_g = int(self.m_rgb_cluster[id_cluster, 1])
    #             val_b = int(self.m_rgb_cluster[id_cluster, 2])
    #
    #
    #             ### draw pnts (raw)
    #             for idx_dict_this, dict_this in enumerate(list_dict_pnt_cluster_this):
    #                 xy_cen_img = dict_this["info_pnt"]["xy_cen_img"]
    #
    #                 x_cen_bev, y_cen_bev = self.m_obj_utils_3D.convert_pnt_img_ori_to_pnt_bev( np.array([[xy_cen_img[0]], [xy_cen_img[1]], [1.0]]) )
    #
    #                 x_cen_bev_int = int(round(x_cen_bev))
    #                 y_cen_bev_int = int(round(y_cen_bev))
    #
    #
    #                 ### draw pnt
    #                 if (0 <= x_cen_bev_int) and (x_cen_bev_int < w_img_bev) and (0 <= y_cen_bev_int) and (y_cen_bev_int < h_img_bev):
    #                     cv2.circle(img_ipm_gray3, center=(x_cen_bev_int, y_cen_bev_int), radius=3, color=(val_b, val_g, val_r), thickness=-1)
    #                 #end
    #             #end
    #
    #
    #             ### draw pnts on polynomial
    #             poly_this = list_poly[id_cluster]
    #
    #             if poly_this is None:
    #                 continue
    #             #end
    #
    #
    #             arr_x_3d_new = np.linspace(0, 100, 500)
    #             arr_y_3d_new = poly_this(arr_x_3d_new)
    #
    #             for idx in range(arr_x_3d_new.size):
    #                 x_3d_new = arr_x_3d_new[idx]
    #                 y_3d_new = arr_y_3d_new[idx]
    #
    #                 x_3d = -1.0*y_3d_new
    #                 y_3d = x_3d_new
    #
    #                 x_bev, y_bev = self.m_obj_utils_3D.convert_pnt_world_to_pnt_bev( np.array([[x_3d], [y_3d], [1.0]]) )
    #
    #                 x_bev_int = int(round(x_bev))
    #                 y_bev_int = int(round(y_bev))
    #
    #                 ### draw pnt
    #                 if (0 <= x_bev_int) and (x_bev_int < w_img_bev) and (0 <= y_bev_int) and (y_bev_int < h_img_bev):
    #                     cv2.circle(img_ipm_gray3, center=(x_bev_int, y_bev_int), radius=1, color=(val_b, val_g, val_r), thickness=-1)
    #                 #end
    #             #end
    #         #end
    #
    #         cv2.imshow('img_ipm_cluster_curve', img_ipm_gray3)
    #         cv2.waitKey(1)
    #     #end
    #
    # #end


#END


############################################################################################################################################################
############################################################################################################################################################
############################################################################################################################################################
############################################################################################################################################################

    # ############################################################################################################
    # ### test2: numpy.polyfit()
    # ############################################################################################################
    # # https://numpy.org/doc/stable/reference/generated/numpy.polyfit.html
    # # https://www.w3schools.com/python/python_ml_polynomial_regression.asp
    #
    # import numpy as np
    # import matplotlib.pyplot as plt
    #
    # x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    # y = np.array([0.0, 0.8, 0.9, 0.1, -0.8, -1.0])
    # z = np.polyfit(x, y, 3)
    #
    # p = np.poly1d(z)
    #
    # xp = np.linspace(-2, 6, 100)
    # plt.plot(x, y, '.', xp, p(xp), '.')
    # plt.ylim(-2, 2)
    #
    # plt.show()

############################################################################################################################################################
############################################################################################################################################################
############################################################################################################################################################
############################################################################################################################################################






