import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

################ Usage ###############
# 1. Read the .pl .sdc into dataset 
# 2. draw_macro_placement_result(args, dataset, logger)
######################################

def draw_macro_placement_result(args, dataset_obj, logger):    
    # build resource to site type dictionary
    resourceToSiteType = {} # e.g., DSP48E2 => DSP; LUT3, FDRE, CARRY8 => SLICE
    for siteType in dataset_obj.sitetypes:
        for resource in siteType.resource.keys():
            resourceToSiteType[resource] = siteType.name
    node_span = {'DSP': 2, 'BRAM': 5} # hard-coded, span in y-direction x,y, x,y+?

    labels = ['NULL', 'DSP', 'BRAM'] # R-xxx: resource for xxx
    res_to_idx = {'NULL':0, 'DSP': 1, 'BRAM': 2}
    # Create a colormap for the integer values
    cmap = plt.cm.get_cmap('viridis', len(labels))
    logger.info("Drawing the macro placement result...")

    ###################### Mark site usage ######################
    logger.info("Marking Macro Placement...")
    site_map_matrix = np.zeros((dataset_obj.sitemap_width, dataset_obj.sitemap_height), dtype=int) # initialize the matrix with NULL

    for nodeid in range(len(dataset_obj.nodes)):
        node_inst = dataset_obj.nodes[nodeid]
        if(nodeid % 50000 == 0):
            print("Checking: {}%{}".format(nodeid, len(dataset_obj.nodes)) )
        if node_inst.is_macro == False:
            continue
        site_type = resourceToSiteType[node_inst.resourcetype]
        for add_y in range(node_span[site_type]):
            site_map_matrix[node_inst.locX][node_inst.locY + add_y] = res_to_idx[site_type]
    
    ###################### Mark macro net connections ######################
    logger.info("Marking net connection between the two macros...")
    macro_connect = []
    distance_plot_thres = 450
    for netid in range(len(dataset_obj.nets)):
        #print(netid)
        net = dataset_obj.nets[netid]
        for pinid_i in range(len(net.macropins)):
            for pinid_j in range(pinid_i+1, len(net.macropins)):
                nodeid_i = net.macropins[pinid_i]
                nodeid_j = net.macropins[pinid_j]
                if dataset_obj.nodes[nodeid_i].is_macro and dataset_obj.nodes[nodeid_j].is_macro:
                    X_i = dataset_obj.nodes[nodeid_i].locX
                    X_j = dataset_obj.nodes[nodeid_j].locX
                    Y_i = dataset_obj.nodes[nodeid_i].locY
                    Y_j = dataset_obj.nodes[nodeid_j].locY
                    if abs(X_i-X_j) + abs(Y_i-Y_j) > distance_plot_thres:
                        macro_connect.append([X_i, X_j, Y_i, Y_j])
    logger.info("The connection larger than {} is {}".format(distance_plot_thres, len(macro_connect)))

    ###################### Plot ######################
    logger.info("Plotting the macro placement result...")
    # Plotting the matrix as a colored grid
    plt.imshow(np.transpose(site_map_matrix), cmap=cmap, origin='lower')
    # Plotting the connection
    for id in range(len(macro_connect)):
        plt.plot([macro_connect[id][0], macro_connect[id][1]], [macro_connect[id][2],macro_connect[id][3]], 'r-')
    # Creating custom legend patches for each label
    patches = [mpatches.Patch(color=cmap(i), label=label) for i, label in enumerate(labels)]
    # Adding a colorbar legend
    plt.legend(handles=patches, loc='lower right')
    # Save the figure as a PNG image
    output_path = os.path.join(args.result_dir, args.exp_id, "site_map_{}.png".format(dataset_obj.params["design_name"]))
    plt.savefig(output_path)
    # Display the plot
    plt.show()