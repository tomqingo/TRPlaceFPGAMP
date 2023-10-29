from src import *
from utils import *
import copy
import math
import time
import multiprocessing
from random import choice
import os

# augment the benchmark
def augment_data(args, logger, dataset, augment_pos_list, augment_neg_list, MacroHPWL_q):
    input_path = os.path.join(args.solution_dir, args.design_name, "solution")
    MacroHPWL_list = []
    for sample_cnt in augment_pos_list:
        logger.info("generating the augmented benchmark"+str(sample_cnt))
        dataset_aug = copy.deepcopy(dataset)
        dataset_aug = RandomAugment(dataset_aug, args.augment_small_range, args.augment_small_per, logger)
        output_path = os.path.join(input_path, "solution_random_"+str(sample_cnt)+"/macroplacement.pl")
        error_path = os.path.join(input_path, "solution_random_"+str(sample_cnt))
        if CheckLegality(dataset_aug, error_path, logger):
            totalMacroHPWL = dataset_aug.calMacroHPWL()
            logger.info("Macro HPWL "+str(totalMacroHPWL))
            dataset_aug.OutputSolutionpl(output_path)
            MacroHPWL_list.append([sample_cnt, totalMacroHPWL])
    
    for sample_cnt in augment_neg_list:
        logger.info("generating the augmented benchmark"+str(sample_cnt))
        dataset_aug = copy.deepcopy(dataset)
        dataset_aug.RandomAugment(args.augment_large_range, args.augment_large_per, logger)
        output_path = os.path.join(input_path, "solution_random_"+str(sample_cnt)+"/macroplacement.pl")
        error_path = os.path.join(input_path, "solution_random_"+str(sample_cnt))
        if CheckLegality(dataset_aug, error_path, logger):
            totalMacroHPWL = dataset_aug.calMacroHPWL()
            logger.info("Macro HPWL "+str(totalMacroHPWL))      
            dataset_aug.OutputSolutionpl(output_path)
            MacroHPWL_list.append([sample_cnt, totalMacroHPWL])
    
    MacroHPWL_q.put(MacroHPWL_list)


# augment the single case
def augment_single_data(args, logger):
    logger.info("Augment all designs in dataset %s." % args.dataset)
    dataset = load_dataset(args, logger)
    dataset.readSamplePl(args.solution, logger)
    totalMacroHPWL_gt = dataset.calMacroHPWL()
    logger.info("Macro HPWL for vivado case:"+str(totalMacroHPWL_gt))
    aug_file_num = list(range(0, args.augment_pos_num + args.augment_neg_num))
    aug_pos_file_num = aug_file_num[0:args.augment_pos_num]
    aug_neg_file_num = aug_file_num[args.augment_pos_num:]
    MacroHPWL_q = multiprocessing.Queue()
    augment_data(args, logger, dataset, aug_pos_file_num, aug_neg_file_num, MacroHPWL_q)
    aug_MacroHPWL_col = []
    aug_MacroHPWL_col.extend(MacroHPWL_q.get())
    aug_MacroHPWL_col = sorted(aug_MacroHPWL_col, key=(lambda x: x[0]))
    logger.info("Finish augment design"+dataset.params["design_name"])
    input_path = dataset.params["sample_dir"]
    writehpwl(args, logger, input_path, aug_MacroHPWL_col, totalMacroHPWL_gt)

# write the HPWL to the 
def writehpwl(args, logger, base_dir_path, hpwl_col, hpwl_gt):
    macroHPWL_path = base_dir_path + "MacroHPWL.txt"
    f_hpwl = open(macroHPWL_path, "w")
    hpwl_str = base_dir_path + "solution_gt "
    hpwl_str += str(hpwl_gt)
    hpwl_str += "\n"
    for id in range(len(hpwl_col)):
        hpwl_str += (base_dir_path + "solution_gt_" + str(hpwl_col[id][0]))
        hpwl_str += " "
        hpwl_str += str(hpwl_col[id][1])
        hpwl_str += "\n"
    f_hpwl.write(hpwl_str)
    f_hpwl.close()

def splitfilecol2job(aug_file_num, njobs):
    aug_file_num_slice = []
    num_file_per_slice = math.ceil(len(aug_file_num)*1.0/njobs)
    for id in range(njobs):
        if id == njobs-1:
            aug_file_num_slice.append(aug_file_num[id*num_file_per_slice:])
        else:
            aug_file_num_slice.append(aug_file_num[id*num_file_per_slice:(id+1)*num_file_per_slice])
    return aug_file_num_slice

def augment_all_data_parallel(args, logger):
    logger.info("Augment all designs in dataset %s." % args.dataset)
    mul_params = get_multiple_design_params(args.dataset_root, args.dataset)
    for i, params in enumerate(mul_params):
        logger.info("Augment"+params["design_name"])
        cur_args = copy.deepcopy(args)
        cur_args.design_name = params["design_name"]
        dataset = load_dataset(cur_args, logger)
        dataset.readSamplePl(args.solution, logger)
        totalMacroHPWL_gt = dataset.calMacroHPWL()
        logger.info("Macro HPWL for vivado case:"+str(totalMacroHPWL_gt))
        aug_file_num = list(range(0, args.augment_pos_num + args.augment_neg_num))        
        aug_pos_file_num = aug_file_num[0:args.augment_pos_num]
        aug_neg_file_num = aug_file_num[args.augment_pos_num:]
        aug_pos_slice = splitfilecol2job(aug_pos_file_num, cur_args.runs)
        aug_neg_slice = splitfilecol2job(aug_neg_file_num, cur_args.runs)
        jobs = []
        MacroHPWL_q = multiprocessing.Queue()
        for j in range(cur_args.runs):
            p = multiprocessing.Process(target=augment_data, args=(cur_args, logger, dataset, aug_pos_slice[j], aug_neg_slice[j], MacroHPWL_q))
            jobs.append(p)
            p.start()
            time.sleep(2)
        for proc in jobs:
            proc.join()
        aug_MacroHPWL_col = []
        for job_id in jobs:
            aug_MacroHPWL_col.extend(MacroHPWL_q.get())
        aug_MacroHPWL_col = sorted(aug_MacroHPWL_col, key=(lambda x: x[0]))
        logger.info("Finish augment design"+params["design_name"])
        input_path = dataset.params["sample_dir"]
        writehpwl(cur_args, logger, input_path, aug_MacroHPWL_col, totalMacroHPWL_gt)

def RandomAugment(db, displacement_thres, augment_numper_thres, logger):
    # Number of the macros that are adjusted
    num_macro_adjust_thres = int(db.num_macro*augment_numper_thres)
    num_macro_adjust = choice(list(range(4, num_macro_adjust_thres)))
    # Number of the cascaded macros (30%), simple macros (70%) that are adjusted
    num_cascade_macro_adjust = choice(list(range(0, min(int(num_macro_adjust*0.3), db.num_cascade_macro)+1)))
    
    # The site and site column col to place the macros
    restype_loc = {"LUT":[], "FF":[], "CARRY8":[], "DSP48E2":[], "RAMB36E2":[], "IO":[]}
    restype_column = {"LUT":[], "FF":[], "CARRY8":[], "DSP48E2":[], "RAMB36E2":[], "IO":[]}
    # The location contained in the column
    col2loc = {}
    left_right_region_slack = 1.0 # The slack distance from the region left/right boundary
    bram_up_down_region_slack = 5.0 # The slack distance from the region up/down boundary for bram
    dsp_up_down_region_slack = 3.0 # The slack distance from the region up/down boundary for dsp

    # Add the columns and the sites in the restype_loc and restype_column
    for id in range(len(db.sites)):
        col_id = db.sites[id].locX
        if not col_id in col2loc:
            col2loc[col_id] = [id]
        else:
            col2loc[col_id].append(id)
        for j in range(len(list(db.sites[id].resource_supply.keys()))):
            res_name = list(db.sites[id].resource_supply.keys())[j]
            if db.sites[id].resource_supply[res_name] > 0:
                restype_loc[res_name].append(id)
                if not col_id in restype_column[res_name]:
                    restype_column[res_name].append(col_id)

    # temporally clear the uncascaded macro on the sites
    for id in range(len(db.sites)):
        site_current = db.sites[id]
        nodecol = site_current.nodecol
        for nodeid in nodecol:
            if db.nodes[nodeid].cascade_id == -1:
                db.sites[id].removeNode(db.nodes[nodeid])

    # choose the cascaded macros to be replaced in a range
    macro_candidate = db.cascademacros[:]
    cnt = 0
    while len(macro_candidate) > 0 and cnt < num_cascade_macro_adjust:
        candidate_id = choice(list(range(len(macro_candidate))))
        macro = macro_candidate[candidate_id]
        macroid = macro.id
        # calculate the slack for each macro
        macro_reference_id = macro.reference_node
        macro_res_type = db.nodes[macro_reference_id].resourcetype
        if macro_res_type == "RAMB36E2":
            up_down_region_slack = bram_up_down_region_slack
        elif macro_res_type == "DSP48E2":
            up_down_region_slack = dsp_up_down_region_slack
        
        nodecol = macro.Macronodecol
        macrolength = macro.num_col
        # Original Location for the macro
        Xcorr_macro, Ycorr_macro, _, _ = db.nodes[macro_reference_id].getLocation()
        site_macro = db.nodes[macro_reference_id].getLocation()

            
        # Construct candidate set for the sites
        column_candidate = []
        site_candidate = []
        for column_id in restype_column[macro_res_type]:
            if abs(column_id - Xcorr_macro) <= displacement_thres:
                column_candidate.append(column_id)
        for column_id in column_candidate:
            loc = col2loc[column_id]
            for site_id in loc:
                site_current = db.sites[site_id]
                Xcorr_site_current, Ycorr_site_current, _, _  = site_current.getLocation()
                # If the location of the site is the same as befoure or it is ouside the placement threshold
                if (Xcorr_site_current == Xcorr_macro and Ycorr_site_current == Ycorr_macro) \
                or abs(Xcorr_site_current - Xcorr_macro) + abs(Ycorr_site_current - Ycorr_macro) > displacement_thres:
                    continue
                
                subflag = True
                for j in range(0, macrolength):
                    immed_site_id = site_id+j
                    X_immed, Y_immed, _, _ = db.sites[immed_site_id].getLocation()
                    if (db.sites[immed_site_id].CheckIsFull(macro_res_type) or X_immed != Xcorr_site_current) or \
                        db.checkRegionFull(macro_res_type, X_immed, Y_immed, left_right_region_slack, up_down_region_slack):
                        subflag = False
                        break                
                if subflag:
                    site_candidate.append(site_id)
        
        # The macro is not considered as candidate when there is no available sites
        if len(site_candidate) == 0:
            macro_candidate.remove(macro)
            continue
            
        # Randomly selected a site and adjust the location
        site_chosen_id = choice(site_candidate)
        Xcorr_site_chosen, Ycorr_site_chosen, _, _ = db.sites[site_chosen_id].getLocation()
        for j in range(0, macrolength):
            nodeid = nodecol[j].id
            # chosen site id
            immed_site_id = site_chosen_id+j
            # original site id
            immed_site_id_org = site_macro+j
            immed_site_X, immed_site_Y, immed_realX, immed_realY = db.sites[immed_site_id].getLocation()
            db.nodes[nodeid].SetPlaceLocation(immed_site_X, immed_site_Y, immed_realX, immed_realY, immed_site_id)
            db.cascademacros[macroid].Macronodecol[j].SetPlaceLocation(immed_site_X, immed_site_Y, immed_realX, immed_realY, immed_site_id)
            db.sites[immed_site_id].addNode(db.nodes[nodeid])
            # change in the region constraint
            for regionid in range(len(db.regionconstrtype)):
                region = db.regionconstrtype[regionid]
                if(region.IsinRegion(immed_site_X, immed_site_Y, left_right_region_slack, up_down_region_slack)):
                    db.regionconstrtype[regionid].AddNode(db.nodes[nodeid])
            # remove the nodes beloning to the site
            db.sites[immed_site_id_org].removeNode(db.nodes[nodeid])
            if immed_site_id in restype_loc[macro_res_type]:
                restype_loc[macro_res_type].remove(immed_site_id)
                col2loc[immed_site_X].remove(immed_site_id)
        cnt += 1
        macro_candidate.remove(macro)

    # Reduce the site that is full after placing the cascaded macro
    for id in range(len(db.sites)):
        siteid = db.sites[id].id
        colid = db.sites[id].locX
        for j in range(len(list(db.sites[id].resource_supply.keys()))):
            res_name = list(db.sites[id].resource_supply.keys())[j]
            if db.sites[id].resource_supply[res_name] > 0 and \
            (db.sites[id].CheckIsFull(res_name) and (siteid in restype_loc[res_name])):
                restype_loc[res_name].remove(siteid)
                col2loc[colid].remove(siteid)

    # Calculate the number of non-cascaded macro
    cnt_cascade_macro_adjust = cnt
    num_non_cascade_macro_adjust = num_macro_adjust - cnt_cascade_macro_adjust
    num_non_cascade_macro_adjust = min(num_non_cascade_macro_adjust, db.num_basic_macro)
    logger.info("Augment cascaded macros:"+str(cnt_cascade_macro_adjust)+",basic macros:"+str(num_non_cascade_macro_adjust))
    

    # Find overlapping of the location non-cascade macros with the cascaded macros
    non_cascade_macro_candidate = []
    nonoverflow_with_regionConstr = []
    nonoverflow_wo_regionConstr = []
    
    for nodeid in range(len(db.nodes)):
        node_current = db.nodes[nodeid]
        node_restype = node_current.resourcetype
        if node_current.is_macro and node_current.cascade_id == -1:
            site_current = node_current.site
            # Xcorr_current = node_current.locX
            # check whether the region is overflow
            if not db.sites[site_current].CheckIsFull(node_restype):
                if node_current.regionconstr_type == -1:
                    nonoverflow_with_regionConstr.append(nodeid)
                else:
                    nonoverflow_wo_regionConstr.append(nodeid)
            else:
                non_cascade_macro_candidate.append(nodeid)
        
    #print(non_cover)
    num_choice = num_non_cascade_macro_adjust - len(non_cascade_macro_candidate)
    num_choice_nodes_with_RC = int(num_choice*0.4)
    num_choice_nodes_with_RC = min(num_choice_nodes_with_RC, len(nonoverflow_with_regionConstr))
    num_choice_nodes_wo_RC = num_choice - num_choice_nodes_with_RC
    num_choice_nodes_wo_RC = min(num_choice_nodes_wo_RC, len(nonoverflow_wo_regionConstr))

    if len(nonoverflow_with_regionConstr) > 0:
        for id in range(num_choice_nodes_with_RC):
            node_choice_id = choice(nonoverflow_with_regionConstr)
            non_cascade_macro_candidate.append(node_choice_id)
            nonoverflow_with_regionConstr.remove(node_choice_id)

    if len(nonoverflow_wo_regionConstr) > 0:
        for id in range(num_choice_nodes_wo_RC):
            node_choice_id = choice(nonoverflow_wo_regionConstr)
            non_cascade_macro_candidate.append(node_choice_id)
            nonoverflow_wo_regionConstr.remove(node_choice_id)

    # Place the nodes that are not selected back to the sites        
    for nodeid in nonoverflow_with_regionConstr:
        site_current = db.nodes[nodeid].site
        column_current = db.nodes[nodeid].locX
        db.sites[site_current].addNode(db.nodes[nodeid])
        restype = db.nodes[nodeid].resourcetype
        restype_loc[restype].remove(site_current)
        col2loc[column_current].remove(site_current)

    for nodeid in nonoverflow_wo_regionConstr:
        site_current = db.nodes[nodeid].site
        column_current = db.nodes[nodeid].locX
        db.sites[site_current].addNode(db.nodes[nodeid])
        restype = db.nodes[nodeid].resourcetype
        restype_loc[restype].remove(site_current)
        col2loc[column_current].remove(site_current)
        
    # Randomly find the corresponding place for each region-constrained node
    for nodeid in non_cascade_macro_candidate:
        if db.nodes[nodeid].regionconstr_type == -1:
            continue
        init_displacement_thres = displacement_thres
        site_candidate = []
        macro_locX, macro_locY, macro_realX, macro_realY = db.nodes[nodeid].getLocation()
        macro_siteid = db.nodes[nodeid].site
        macro_res_type = db.nodes[nodeid].resourcetype
        while len(site_candidate) == 0 and init_displacement_thres < 1000:
            # get the candidate site
            for site_id in restype_loc[macro_res_type]:
                site_current = db.sites[site_id]
                site_locX, site_locY, site_realX, site_realY = db.sites[site_id].getLocation()
                #if macro_locX == site_locX and macro_locY == site_locY:
                #    continue
                db.nodes[nodeid].SetPlaceLocation(site_locX, site_locY, site_realX, site_realY, site_id)
                if db.nodes[nodeid].IsBRAM():
                    up_down_region_slack = bram_up_down_region_slack
                if db.nodes[nodeid].IsDSP():
                    up_down_region_slack = dsp_up_down_region_slack                        
                if not db.nodes[nodeid].IsinRegionConstr(left_right_region_slack, up_down_region_slack):
                    db.nodes[nodeid].SetPlaceLocation(macro_locX, macro_locY, macro_realX, macro_realY, macro_siteid)
                    continue
                db.nodes[nodeid].SetPlaceLocation(macro_locX, macro_locY, macro_realX, macro_realY, macro_siteid)
                if abs(macro_locX - site_locX) + abs(macro_locY - site_locY) <= init_displacement_thres:
                    site_candidate.append(site_id)
            init_displacement_thres = init_displacement_thres * 1.5
            
        # placed the cells
        if len(site_candidate) > 0:
            placed_site_id = choice(site_candidate)
            placed_X, placed_Y, placed_RealX, placed_RealY = db.sites[placed_site_id].getLocation()
            db.nodes[nodeid].SetPlaceLocation(placed_X, placed_Y, placed_RealX, placed_RealY, placed_site_id)
            db.sites[placed_site_id].addNode(db.nodes[nodeid])
            restype_loc[macro_res_type].remove(placed_site_id)
            col2loc[placed_X].remove(placed_site_id)
        else:
            place_id = db.nodes[nodeid].site
            db.sites[place_id].addNode(db.nodes[nodeid])


    # Randomly find the corresponding place for each non-region_constrained node
    for nodeid in non_cascade_macro_candidate:
        if db.nodes[nodeid].regionconstr_type != -1:
            continue
        # Generate the feasible location set for each node
        init_displacement_thres = displacement_thres
        site_candidate = []
        macro_locX, macro_locY, _, _ = db.nodes[nodeid].getLocation()
        macro_res_type = db.nodes[nodeid].resourcetype
        while len(site_candidate) == 0:
            for site_id in restype_loc[macro_res_type]:
                site_current = db.sites[site_id]
                site_locX, site_locY, _, _ = db.sites[site_id].getLocation()
                if (macro_locX != site_locX or macro_locY != site_locY) and \
                abs(macro_locX - site_locX) + abs(macro_locY - site_locY) <= init_displacement_thres:
                    site_candidate.append(site_id)
            init_displacement_thres = init_displacement_thres * 1.5
            
        placed_site_id = choice(site_candidate)
        placed_X = db.sites[placed_site_id].locX
        placed_Y = db.sites[placed_site_id].locY
        db.nodes[nodeid].ReSetPlaceLocation(placed_X, placed_Y, placed_site_id)
        db.sites[placed_site_id].addNode(db.nodes[nodeid])
        restype_loc[macro_res_type].remove(placed_site_id)
        col2loc[placed_X].remove(placed_site_id)
        


    

