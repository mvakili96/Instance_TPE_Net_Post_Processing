# 2020/9/4
# Jungwon Kang

import numpy as np


########################################################################################################################
###
########################################################################################################################
def rectify_pixel_value(val_pixel_int):

    if val_pixel_int > 255:
        val_pixel_int = 255
    #end

    if val_pixel_int < 0:
        val_pixel_int = 0
    #end

    return val_pixel_int

########################################################################################################################
###
########################################################################################################################
def adjust_rgb_for_region(b_old_uint8, g_old_uint8, r_old_uint8, type_region = 0):
    """
    :param b_old_uint8:
    :param g_old_uint8:
    :param r_old_uint8:
    :param type_region: 0(normal region), 1(switch region)
    :return:
    """

    ###
    db_int = 0
    dg_int = 0
    dr_int = 0

    if type_region == 0:
        db_int = 0
        dg_int = 100
        dr_int = 0
    elif type_region == 1:
        db_int = -20
        dg_int = -20
        dr_int = 50
    #end


    ###
    b_new_int = int(b_old_uint8) + db_int
    g_new_int = int(g_old_uint8) + dg_int
    r_new_int = int(r_old_uint8) + dr_int

    b_new_int = rectify_pixel_value(b_new_int)
    g_new_int = rectify_pixel_value(g_new_int)
    r_new_int = rectify_pixel_value(r_new_int)

    b_new_uint8 = np.uint8(b_new_int)
    g_new_uint8 = np.uint8(g_new_int)
    r_new_uint8 = np.uint8(r_new_int)

    return b_new_uint8, g_new_uint8, r_new_uint8


    # ### class_ADErail_this: RAIL_EGO/RAIL_SPLIT_FROM_EGO/RAIL_MERGE_TO_EGO/RAIL_OTHER
    # if class_ADErail_this == 'RAIL_EGO':
    #     db_int = 0
    #     dg_int = 100
    #     dr_int = 0
    #
    # elif class_ADErail_this == 'RAIL_SPLIT_FROM_EGO':
    #     db_int = 0
    #     dg_int = 0
    #     dr_int = 100
    #
    # elif class_ADErail_this == 'RAIL_MERGE_TO_EGO':
    #     db_int = 100
    #     dg_int = -50
    #     dr_int = -50
    #
    # else:  # class_ADErail_this == RAIL_OTHER
    #     db_int =  100
    #     dg_int = -25
    #     dr_int =  25


