import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

################ Usage ###############
# 1. Read the .pl .sdc into dataset 
# 2. draw_macro_placement_result(dataset, logger)
######################################

def draw_macro_placement_result(dataset_obj, logger):    
    # build resource to site type dictionary
    resourceToSiteType = {} # e.g., DSP48E2 => DSP; LUT3, FDRE, CARRY8 => SLICE
    for siteType in dataset_obj.sitetypes:
        for resource in siteType.resourcecap.keys():
            resourceToSiteType[resource] = siteType.name
    node_span = {'DSP': 2, 'BRAM': 5, 'URAM': 15} # hard-coded, span in y-direction x,y, x,y+?

    labels = ['NULL', 'DSP', 'BRAM', 'URAM'] # R-xxx: resource for xxx
    res_to_idx = {'NULL':0, 'DSP': 1, 'BRAM': 2, 'URAM': 3}
    # Create a colormap for the integer values
    cmap = plt.cm.get_cmap('viridis', len(labels))
    logger.info("Drawing the macro placement result...")

    ###################### Mark site usage ######################
    logger.info("\t Marking Macro Placement...")
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
    
    ###################### Plot ######################
    logger.info("\t Plotting the macro placement result...")
    # Plotting the matrix as a colored grid
    plt.imshow(np.transpose(site_map_matrix), cmap=cmap, origin='lower')
    # Creating custom legend patches for each label
    patches = [mpatches.Patch(color=cmap(i), label=label) for i, label in enumerate(labels)]
    # Adding a colorbar legend
    plt.legend(handles=patches, loc='lower right')
    # Save the figure as a PNG image
    plt.savefig("site_map_{}.png".format(dataset_obj.params["design_name"]))
    # Display the plot
    plt.show()