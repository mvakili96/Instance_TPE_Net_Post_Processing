# 2020/7/12
# Jungwon Kang


import torch
from collections import OrderedDict


########################################################################################################################
###
########################################################################################################################
class MyUtils_Net:


    m_fname_weights_to_be_loaded = None


    ###############################################################################################################
    ###
    ###############################################################################################################
    def __init__(self, dict_args):
        self.m_fname_weights_to_be_loaded = dict_args["file_weight"]
    #end


    ###############################################################################################################
    ###
    ###############################################################################################################
    def load_weights_to_model(self, model):
        """
        load trained weights to a model

        :param model
        :param fname_weights_to_be_loaded
        :return: model
        """


        #////////////////////////////////////////////////////////////////////////////////////////////////////////////
        # model: empty model, which we want to fill in by weights-to-be-loaded
        # fname_weights_to_be_loaded: path to a file of weights-to-be-loaded
        #////////////////////////////////////////////////////////////////////////////////////////////////////////////

        # <terminology>
        #  state_dict_model  : network structure from model
        #  state_dict_weights: network weights from file


        ###================================================================================================
        ### 1. load weights-to-be-loaded from a file
        ###================================================================================================
        state_dict_weights0 = torch.load(self.m_fname_weights_to_be_loaded)["model_state"]
        print('loaded weights-to-be-loaded form %s !' % self.m_fname_weights_to_be_loaded)
            # completed to set
                #       state_dict_weights0{}: weights loaded from weights-to-be-loaded file


        ###================================================================================================
        ### 2. selective copy (1): copy state_dict_weights0[] -> state_dict_weights[]
        ###================================================================================================
        state_dict_weights = OrderedDict()

        for key in state_dict_weights0:
            if key.startswith('module') and not key.startswith('module_list'):
                #state_dict[key[7:]] = state_dict_[key]     # <original code>
                state_dict_weights[key] = state_dict_weights0[key]          # <edited by Jungwon>
            else:
                state_dict_weights[key] = state_dict_weights0[key]
            #end
        #end
            # completed to set
            #       state_dict_weights{}


        ###================================================================================================
        ### 3. selective copy (2): copy state_dict_model[] -> state_dict_weights[]
        ###================================================================================================
        # note that
        #   state_dict{}       : from weights_to_be_loaded
        #   model_state_dict{} : from model

        state_dict_model = model.state_dict()
            # completed to set
            #       model_state_dict[]: empty model, which we want to fill in by pretrained-weights


        ###
        for key in state_dict_weights:              # state_dict[]:       modules loaded from pretrained-weights file
            if key in state_dict_model:     # model_state_dict[]: modules from empty model
                ###------------------------------------------------------------------------------
                ### if shape is not consistent, just skip.
                ###------------------------------------------------------------------------------
                if state_dict_weights[key].shape != state_dict_model[key].shape:
                    print('Skip loading parameter {}, required shape{}, loaded shape{}.'.format(
                        key, state_dict_model[key].shape, state_dict_weights[key].shape))
                    state_dict_weights[key] = state_dict_model[key]         # copy dummy into state_dict[]
                #end
            else:
                ###------------------------------------------------------------------------------
                ### if key in state dict[] does not exist in model_state_dict[], just ignore.
                ###------------------------------------------------------------------------------
                print('Drop parameter {}.'.format(key))
            #end
        #end


        ###
        for key in state_dict_model:
            if key not in state_dict_weights:
                ###------------------------------------------------------------------------------
                ### if key in state_dict_model{} does not exist in state_dict_weights{}, just ignore.
                ###------------------------------------------------------------------------------
                print('No param {}.'.format(key))
                state_dict_weights[key] = state_dict_model[key]         # copy dummy into state_dict_weights[]
            #end
        #end
            # completed to set
            #       state_dict[]: final pretrained-weights


        ###================================================================================================
        ### 4. fill in model with final weights_to_be_loaded
        ###================================================================================================
        model.load_state_dict(state_dict_weights, strict=False)


        return model
            # model: model filled by weights_to_be_loaded
    #end


########################################################################################################################



