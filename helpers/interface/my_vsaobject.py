# Sept 4 2020
# Jungwon Kang


### VSAObject
# from helpers.interface.vsadatatypes_yorku   import *
# from helpers.interface.constants_yorku      import *

from helpers.utils.my_common_type   import *


########################################################################################################################
### dummy VSA object
########################################################################################################################
class Polygon_dummy():
    def __init__(self, vertices=None, coefficients=None):
        self.vertices = vertices
        self.coefficients = coefficients
    #end
#end


########################################################################################################################
###
########################################################################################################################
def create_VSAObject_from_PE_results(list_res_paths):
    ###===================================================================================================
    ### create VSAObject from PE results
    ###===================================================================================================
    detections = []


    for idx_path in range(len(list_res_paths)):
        list_path_this = list_res_paths[idx_path]

        type_path = list_path_this["type_path"]

        if type_path is TYPE_path.EGO:  # create for only EGO-path
            ### get
            xyz_left_3d             = list_path_this["polynomial"]["xyz_left_3d"]
            xyz_right_3d            = list_path_this["polynomial"]["xyz_right_3d"]

            coeff_poly_left_3d_new  = list_path_this["polynomial"]["coeff_poly_left_3d_new"]
            coeff_poly_right_3d_new = list_path_this["polynomial"]["coeff_poly_right_3d_new"]


            ###
            detections.append(Polygon_dummy(vertices=xyz_left_3d, coefficients=coeff_poly_left_3d_new))
            detections.append(Polygon_dummy(vertices=xyz_right_3d, coefficients=coeff_poly_right_3d_new))
        #end
    #end


    return detections
#END

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


