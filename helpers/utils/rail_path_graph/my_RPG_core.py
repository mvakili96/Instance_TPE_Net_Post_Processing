# Aug 23 2020
# Updated on Sept 3 2020
# Jungwon Kang


import enum


# <to-be-added>
#   condition: id_node_a (near) - id_node_b (far)
#   self.prev has a single id?


########################################################################################################################
###
########################################################################################################################
class TYPE_node(enum.Enum):
    START = 0
    END = 1
    SWITCH = 2
#End


########################################################################################################################
###
########################################################################################################################
class Node_RPG:
    def __init__(self, id_node, type_node):
        self.id_node = id_node
        self.type_node = type_node

        self.prev = []
        self.next = []
    #end
#END


########################################################################################################################
###
########################################################################################################################
class DLList_RPG:
    ###======================================================================================================
    ### DLList_RPG
    ###======================================================================================================
    def __init__(self):
        self.nodes = {}
    #end


    ###======================================================================================================
    ### DLList_RPG
    ###======================================================================================================
    def create_node(self, id_node, type_node):
        ###
        node_new = Node_RPG(id_node, type_node)

        if str(id_node) in self.nodes:  # str(id_node) should be new in self.nodes
            assert 0
        #end

        ###
        self.nodes[str(id_node)] = node_new
    #end


    ###======================================================================================================
    ### DLList_RPG
    ###======================================================================================================
    def update_connections_in_node(self, id_node_a, id_node_b):
        # condition: id_node_a (near) - id_node_b (far)

        self.nodes[str(id_node_a)].next.append(id_node_b)
        self.nodes[str(id_node_b)].prev.append(id_node_a)
    #end


    ###======================================================================================================
    ### DLList_RPG
    ###======================================================================================================
    # assume that there is only one connection from a far node to a near node.
    def get_feasible_paths_as_node_set(self):

        list_paths = []


        for key_node_here in self.nodes:
            ###
            node_here = self.nodes[key_node_here]

            ###
            if node_here.type_node is not TYPE_node.END:
                continue
            #end

            ### set init node for search
            list_path_this = []

            id_node_this = node_here.id_node
            id_node_prev = node_here.prev[0]

            while 1:
                ### register
                list_path_this.append(id_node_this)

                ### move to prev
                id_node_this = id_node_prev

                if self.nodes[str(id_node_this)].type_node is TYPE_node.START:
                    list_path_this.append(id_node_this)
                    break
                #end

                id_node_prev = self.nodes[str(id_node_this)].prev[0]
            #end

            list_path_this_reverse = list_path_this[::-1]
            list_paths.append(list_path_this_reverse)
        #end


        return list_paths
    #end


#END


########################################################################################################################
###
########################################################################################################################
def get_sample_paths():

    ###=============================================================================================
    ### sample input
    ###=============================================================================================
    # list_all_nodes  = [0, 1, 2, 3]
    # list_type_nodes = [(TYPE_node.START), (TYPE_node.SWITCH), (TYPE_node.END), (TYPE_node.END)]
    # list_all_node_pairs_for_edges = [[0,1], [1,2], [1,3]]

    list_all_nodes  = [-1, 20, 30, 40]
    list_type_nodes = [(TYPE_node.START), (TYPE_node.SWITCH), (TYPE_node.END), (TYPE_node.END)]
    list_all_node_pairs_for_edges = [[-1, 20], [20, 30], [20, 40]]


    ###=============================================================================================
    ### create obj
    ###=============================================================================================
    obj_rpg = DLList_RPG()


    ###=============================================================================================
    ###  create nodes
    ###=============================================================================================
    for i in range(len(list_all_nodes)):
        ###
        id_node   = list_all_nodes[i]
        type_node = list_type_nodes[i]
        #print("id_node: [%d], type_node [%s]" % (id_node, type_node))

        ###
        obj_rpg.create_node(id_node, type_node)
    #end


    ###=============================================================================================
    ### update connections
    ###=============================================================================================
    for i in range(len(list_all_node_pairs_for_edges)):
        ###
        id_node_pair = list_all_node_pairs_for_edges[i]
        id_node_a    = id_node_pair[0]
        id_node_b    = id_node_pair[1]
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
#END

########################################################################################################################
###
########################################################################################################################
if __name__ == '__main__':

    ###
    get_sample_paths()

