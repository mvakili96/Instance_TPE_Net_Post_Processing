# 2020/7/10
# Jungwon Kang

import copy


from helpers.models.TPEnet_a import TPEnet_a
from helpers.models.TPEnet_a import TPEnet_b


########################################################################################################################
###
########################################################################################################################
def get_model(model_dict, dim_ins_vec, dim_seg_vec):
    """get model"""

    name        = model_dict["arch"]
    model       = _get_model_instance(name)
    param_dict  = copy.deepcopy(model_dict)
    param_dict.pop("arch")

    model       = model(n_classes_ins = dim_ins_vec, n_classes_seg = dim_seg_vec, **param_dict)

    return model
#end


########################################################################################################################
###
########################################################################################################################
def _get_model_instance(name):
    """get model instance"""

    try:
        return {
            "TPEnet_a": TPEnet_a,
            "TPEnet_b": TPEnet_b,
        }[name]
    except:
        raise ("Model {} not available".format(name))
    #end

#end


########################################################################################################################
