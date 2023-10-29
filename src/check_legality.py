import os
from utils import *
from src.db import *
from src.MacroPl import *
from src import *

# Check whether the macros (simple and cascaded) are placed on the correct site
def CheckCoordinateLegality(db, error_str, logger):
    is_legal = True
    for id in range(len(db.nodes)):
        # Only check the simple macros that are placed
        if db.nodes[id].isPlace and db.nodes[id].cascade_id == -1:
            # check whether the nodes are in the sitemap
            if db.nodes[id].locX < 0:
                error_str += ("Invalid site ("+ str(db.nodes[id].locX) + ","+str(db.nodes[id].locY)+") for macro "+db.nodes[id].name+" , left below the 0"+"\n")
                is_legal = False
            elif db.nodes[id].locX > db.sitemap_width:
                error_str += ("Invalid site ("+ str(db.nodes[id].locX) + ","+str(db.nodes[id].locY)+") for macro "+db.nodes[id].name+" , right above the "+str(db.sitemap_width)+"\n")
                is_legal = False                
            elif db.nodes[id].locY < 0:
                error_str += ("Invalid site ("+ str(db.nodes[id].locX) + ","+str(db.nodes[id].locY)+") for macro "+db.nodes[id].name+" , down below the 0"+"\n")
                is_legal = False
            elif db.nodes[id].locY > db.sitemap_height:
                error_str += ("Invalid site ("+ str(db.nodes[id].locX) + ","+str(db.nodes[id].locY)+") for macro "+db.nodes[id].name+" , up above the "+str(db.sitemap_height)+"\n")
                is_legal = False
                
            # check whether the nodes are in the reginal constraint
            if not db.nodes[id].IsinRegionConstr():
                error_str += ("The location ("+str(db.nodes[id].locX) + ","+str(db.nodes[id].locY)+") for macro "+db.nodes[id].name+" not in the region constraints"+"\n")              
                is_legal = False

            # check whether the nodes are placed on the suitable site            
            siteid = db.nodes[id].site
            resourcetype = db.nodes[id].resourcetype
            #print(siteid, resourcetype)
            if db.sites[siteid].resource_supply[resourcetype]<=0:
                error_str += ("Invalid site type"+db.sites[siteid].sitetype+"for macro:"+db.nodes[id].name+" site:("+str(db.sites[siteid].locX)+","+str(db.sites[siteid].locY)+")\n")
                is_legal = False
    
    # check the cascade macros
    for id in range(len(db.cascademacros)):
        ref_node_id = db.cascademacros[id].reference_node
        if db.nodes[ref_node_id].isPlace:
            # check whether the macros are in the sitemap
            if db.cascademacros[id].locX < 0:
                error_str += ("Invalid site ("+ str(db.cascademacros[id].locX) + ","+str(db.cascademacros[id].locY)+") for cascaded macro "+db.cascademacros[id].name+" , left below the 0"+"\n")
                is_legal = False
            elif db.cascademacros[id].locX + db.cascademacros[id].width > db.sitemap_width:
                error_str += ("Invalid site ("+ str(db.cascademacros[id].locX) + ","+str(db.cascademacros[id].locY)+") for cascaded macro "+db.cascademacros[id].name+" , right above the "+str(db.sitemap_width)+"\n")
                is_legal = False                
            elif db.cascademacros[id].locY < 0:
                error_str += ("Invalid site ("+ str(db.cascademacros[id].locX) + ","+str(db.cascademacros[id].locY)+") for cascaded macro "+db.cascademacros[id].name+" , down below the 0"+"\n")
                is_legal = False
            elif db.cascademacros[id].locY + db.cascademacros[id].height > db.sitemap_height:
                error_str += ("Invalid site ("+ str(db.cascademacros[id].locX) + ","+str(db.cascademacros[id].locY)+") for cascaded macro "+db.cascademacros[id].name+" , up above the "+str(db.sitemap_height)+"\n")
                is_legal = False
            
            # check whether the macros are placed on the suitable site
            siteid = db.nodes[ref_node_id].site
            resourcetype = db.nodes[ref_node_id].resourcetype
            if db.sites[siteid].resource_supply[resourcetype] <= 0:
                error_str += ("Invalid site type"+db.sites[siteid].sitetype+"for cascaded macro:"+db.cascademacros[id].name+" site:("+str(db.sites[siteid].locX)+","+str(db.sites[siteid].locY)+")\n")
                is_legal = False

    return is_legal, error_str


# Check whether the resource of the site is overflow
def CheckSiteResourceOverflow(db, error_str, logger):
    is_legal = True
    for i in range(len(db.sites)):
        for j in range(len(list(db.sites[i].resource_supply.keys()))):
            res_name = list(db.sites[i].resource_supply.keys())[j]
            if db.sites[i].resource_usage[res_name] > db.sites[i].resource_supply[res_name]:
                error_str += ("Excessive resource demand in Site:("+str(db.sites[i].locX)+","+str(db.sites[i].locY)+")"+" Demand:"+str(db.sites[i].resource_usage[res_name])+" Supply:"+str(db.sites[i].resource_supply[res_name])+"\n")
                is_legal = False
    return is_legal, error_str
    
# Check whether the nodes in cascaded macros are placed continuously
def CheckMacroShape(db, error_str, logger):
    is_legal = True
    for i in range(len(db.cascademacros)):
        macro_node_col = db.cascademacros[i].Macronodecol
        macro_node_id_col = []
        for j in range(len(macro_node_col)):
            macro_node_id_col.append(macro_node_col[j].id)
        ref_node = db.nodes[macro_node_id_col[0]]
        site_id = ref_node.site
        for j in range(0,len(macro_node_id_col)):
            nodeid = macro_node_id_col[j]
            new_site_id = site_id + j
            gt_site_locX = db.sites[new_site_id].locX
            gt_site_locY = db.sites[new_site_id].locY
            cal_site_locX = db.nodes[nodeid].locX
            cal_site_locY = db.nodes[nodeid].locY
            if not (gt_site_locX==cal_site_locX and gt_site_locY==cal_site_locY):
                error_str += ("The placed sites for cells in Macro:"+ db.cascademacros[i].name+"are not neighboring\n")
                is_legal = False
                break
    return is_legal, error_str


# Check the legality of the placement 
def CheckLegality(db, log_dir, logger):
    logger.info("====Legality Check====")
    is_legal = True
    placelegal_path = os.path.join(log_dir, "PlaceError.log")
    if not os.path.exists(os.path.dirname(placelegal_path)):
       os.makedirs(os.path.dirname(placelegal_path))
    f_legal = open(placelegal_path, "w")
    error_str = ""
    is_legal_sub, error_str = CheckCoordinateLegality(db, error_str, logger)
    if not is_legal_sub:
        logger.info("Cell Coordinate Legality Check does not pass")
        is_legal = False

    is_legal_sub, error_str = CheckSiteResourceOverflow(db, error_str, logger)        
    if not is_legal_sub:
        logger.info("Site Resource Check does not pass")
        is_legal = False
        
    is_legal_sub, error_str = CheckMacroShape(db, error_str, logger) 
    if not is_legal_sub:
        logger.info("Macro Placement Shape Check does not pass")
        is_legal = False
        
    if is_legal:
        logger.info("Legality Check pass!!")
    else:
        logger.info("Legality Check not pass!!")
        f_legal.write(error_str)
        
    f_legal.close()
    return is_legal