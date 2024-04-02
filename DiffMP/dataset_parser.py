#conding=utf8  
import os
import json
import numpy as np

if not os.path.exists("./dataset/site2idx.json"):
    print('Creating site-to-index mapping...')
    dict_DSP = {}
    dict_BRAM = {}
    count_DSP = 1
    count_BRAM = 1
    # Read design.scl
    with open("./dataset/design.scl", 'r') as site_map:
        site_map = site_map.readlines()
        for i in range(54, len(site_map)):
            if 'DSP' in site_map[i]:
                site_x = site_map[i].strip().split(' ')[0]
                site_y = site_map[i].strip().split(' ')[1]
                dict_DSP[site_x+','+site_y] = str(count_DSP)
                count_DSP += 1
            elif 'BRAM' in site_map[i]:
                site_x = site_map[i].strip().split(' ')[0]
                site_y = site_map[i].strip().split(' ')[1]
                dict_BRAM[site_x+','+site_y] = str(count_BRAM)
                count_BRAM += 1
    with open('./dataset/site2idx.json', mode='w', encoding='utf-8') as f:
        json.dump([dict_DSP, dict_BRAM], f)
else:
    print('Loading site-to-index mapping...')
    with open('./dataset/site2idx.json', mode='r', encoding='utf-8') as f:
        dicts = json.load(f)
        # the conversion between the 2d DSP sites to 1d DSP site
        dict_DSP = dicts[0]
        dict_BRAM = dicts[1]
        print('len(dict_DSP):', len(dict_DSP))
        print('len(dict_BRAM):', len(dict_BRAM))


if __name__ == '__main__':

    # Using the regional constraints

    with open('./dataset/all_solutions.txt', 'w') as dataset_sol:
        with open('./dataset/all_constraints.txt', 'w') as dataset_con:
            with open('./dataset/all_metrics.txt', 'w') as dataset_met:
                with open('./dataset/all_netlists.txt', 'w') as dataset_net:
                    with open('./dataset/all_ordering.txt', 'w') as dataset_order:
                        
                        for root, ds, _ in os.walk('/research/d1/gds/qluo22/dataset_col/mlcad2023_v2'):
                            for d in ds:
                                if 'Design' in d:
                                    base_design_name = d
                                    print('processing:', base_design_name)

                                    # read macro information
                                    node2placementunitid = {}
                                    node_info_path = os.path.join(root, base_design_name, 'netlist_feature/PU_info.txt')
                                    with open(node_info_path, mode="r", encoding="utf-8") as node_info:
                                        node_info = node_info.readlines()
                                        for i_ in range(len(node_info)):
                                            if "Begin" in node_info[i_]:
                                                placeunitid = int(node_info[i_].split(' ')[2])
                                            elif "End" in node_info[i_]:
                                                continue
                                            else:
                                                node2placementunitid[node_info[i_][:-1]] = placeunitid

                                    # read macro feature
                                    node_feature_path = os.path.join(root, base_design_name, 'netlist_feature/PU_feature.txt')
                                    with open(node_feature_path, mode='r', encoding='utf-8') as node_feature:
                                        node_feature = node_feature.readlines()
                                        BRAM_size_list = []
                                        BRAM_degree_list = []
                                        DSP_size_list = []
                                        DSP_degree_list = []
                                        BRAM_regionarea_list = []
                                        DSP_regionarea_list = []
                                        BRAM_name2size = {}
                                        DSP_name2size = {}
                                        
                                        for i_ in range(len(node_feature)):
                                            macro_name = node_feature[i_].strip().split(' ')[0]
                                            macro_feature = node_feature[i_].strip().split(' ')[-1].split(',')

                                            # macro size / degree
                                            macro_size = macro_feature[4]
                                            macro_degree = macro_feature[5]
                                            macro_constr_area = macro_feature[-2]

                                            if macro_feature[0] == '1.0': ### BRAM
                                                BRAM_size_list.append(float(macro_size))
                                                BRAM_degree_list.append(float(macro_degree))
                                                BRAM_regionarea_list.append(float(macro_constr_area))
                                                BRAM_name2size[int(macro_name)] = float(macro_size)
                                            elif macro_feature[1] == '1.0': ### DSP
                                                DSP_size_list.append(float(macro_size))
                                                DSP_degree_list.append(float(macro_degree))
                                                DSP_regionarea_list.append(float(macro_constr_area))
                                                DSP_name2size[int(macro_name)] = float(macro_size)
                                            else:
                                                raise NotImplementedError("Unrecognizable macro type")

                                        # if base_design_name == "Design_5":
                                        #     print(BRAM_name2size)
                                        #     print(DSP_name2size)
                                        # the BRAM and DSP
                                        BRAM_order = [i for i in range(len(BRAM_size_list))]
                                        DSP_order = [i for i in range(len(DSP_size_list))]
                                        # print('DSP_order:', DSP_order)

                                        BRAM_zip = zip(BRAM_order, BRAM_size_list, BRAM_regionarea_list, BRAM_degree_list)
                                        DSP_zip = zip(DSP_order, DSP_size_list, DSP_regionarea_list, DSP_degree_list)
                                        # print('DSP_zip:', DSP_zip)

                                        # sorted according to the size, the area of the regional constraint, degree of the macro
                                        BRAM_order_sorted = sorted(BRAM_zip, key=lambda x:(x[1],-x[2],x[3]), reverse=True)
                                        DSP_order_sorted = sorted(DSP_zip, key=lambda x:(x[1],-x[2],x[3]), reverse=True)
                                        # print('DSP_order_sorted1:', DSP_order_sorted)

                                        # the order of the placement
                                        BRAM_order_sorted = [list(x) for x in zip(*BRAM_order_sorted)][0]
                                        DSP_order_sorted = [list(x) for x in zip(*DSP_order_sorted)][0]
                                        # print('DSP_order_sorted2:', DSP_order_sorted)

                                        # raise NotImplementedError
                                        BRAM_order_sorted = ",".join([str(i) for i in BRAM_order_sorted])
                                        DSP_order_sorted = ",".join([str(i) for i in DSP_order_sorted])

                                    # region constraints
                                    region_constraint_path = os.path.join(root, base_design_name, 'design.regions')
                                    metric_path = os.path.join(root, base_design_name, 'AMF_solution', 'MacroHPWL.txt')

                                    with open(metric_path, mode='r', encoding='utf-8') as f_metric:
                                        f_metric = f_metric.readlines()
                                        if len(f_metric) > 0:
                                            all_values = [float(i.strip().split(' ')[-1]) for i in f_metric]
                                            max_value = max(all_values)
                                            min_value = min(all_values)
                                            range_ = max_value - min_value ### For normalization
                                            f_metric_dict = {}
                                            for m in range(len(f_metric)):
                                                name = f_metric[m].strip().split(' ')[0].split('/')[-1]
                                                value = f_metric[m].strip().split(' ')[-1]
                                                # f_metric_dict[name] = value
                                                f_metric_dict[name] = str((float(value) - min_value) / range_)

                                    if not os.path.exists(region_constraint_path):
                                        continue

                                    else:
                                        with open(region_constraint_path, mode='r', encoding='utf-8') as f_region:
                                            f_region = f_region.readlines()
                                            idx2region = {}
                                            for i in range(len(f_region)-1):
                                                if 'RegionConstraint BEGIN' in f_region[i] and '#' not in f_region[i]:
                                                    constraint_idx = f_region[i].strip().split(' ')[2]
                                                    # constraint_rect = ','.join(f_region[i+1].strip().split(' ')[1:])
                                                    # int
                                                    constraint_rect = np.array([float(i) for i in f_region[i+1].strip().split(' ')[1:]]) / np.array([206,300,206,300])
                                                    constraint_rect = np.clip(constraint_rect, 0., 1.).tolist()
                                                    constraint_rect = ','.join([str(i) for i in constraint_rect])

                                                    idx2region[constraint_idx] = constraint_rect
                                                if 'InstanceToRegionConstraintMapping' in f_region[i] and '#' not in f_region[i]:
                                                    break
                                            
                                            constraint_dict_DSP = {}
                                            constraint_dict_BRAM = {}
                                            for i in range(len(f_region)):
                                                if 'DSP' in f_region[i]:
                                                    macro_name = f_region[i].strip().split(' ')[0]
                                                    constraint_idx = f_region[i].strip().split(' ')[-1]
                                                    constraint_dict_DSP[macro_name] = idx2region[constraint_idx]
                                                elif 'BRAM' in f_region[i]:
                                                    macro_name = f_region[i].strip().split(' ')[0]
                                                    constraint_idx = f_region[i].strip().split(' ')[-1]
                                                    constraint_dict_BRAM[macro_name] = idx2region[constraint_idx]

                                    base_folder = ('/research/d1/gds/qluo22/dataset_col/mlcad2023_v2/%s/AMF_solution' % base_design_name) 
                                    for sub_folder in os.listdir('/research/d1/gds/qluo22/dataset_col/mlcad2023_v2/%s/AMF_solution' % base_design_name):
                                        sub_f = os.path.join(base_folder, sub_folder, "macroplacement.pl")
                                        if os.path.exists(sub_f):
                                            design_name = base_design_name + '@' + sub_folder
                                            
                                            DSP = []
                                            BRAM = []
                                            # place location for DSP and BRAM Macros
                                            place_DSP = []
                                            place_BRAM = []
                                            # regional constraints for DSP and BRAM Macros
                                            con_DSP = []
                                            con_BRAM = []
                                            # whole_region = '0,0,206,300'
                                            whole_region = '0.0,0.0,1.0,1.0'  ### Normalized, Why the region constraints needs to be normalized
                                                
                                            with open(sub_f, 'r') as f_sol:
                                                f_sol = f_sol.readlines()
                                                for i in range(len(f_sol)):
                                                    macro_name = f_sol[i].strip().split(' ')[0]
                                                    site_x = f_sol[i].strip().split(' ')[1]
                                                    site_y = f_sol[i].strip().split(' ')[2]
                                                    if 'DSP' in macro_name:
                                                        DSP.append(macro_name)
                                                        # The i-th DSP site (conversion between (site_x, site_y) and siteid)
                                                        place_DSP.append(dict_DSP[site_x+','+site_y])
                                                        # Special operation for DSP_config cells
                                                        if "DSP_config" in macro_name:
                                                            macro_name_col = macro_name.split("/")
                                                            macro_name = macro_name_col[0] + "/" + macro_name_col[1]
                                                        # cascade macro (not exceed the boundary)
                                                        if "CASCADE" in macro_name:
                                                            left  = str(0.0)
                                                            right = str(1.0)
                                                            macroid = node2placementunitid[macro_name]
                                                            up = str((300.0-DSP_name2size[macroid]*2.5)/300.0)
                                                            down = str(0.0)
                                                            region = [left, down, right, up]
                                                            region_str = ",".join(region)
                                                            con_DSP.append(region_str)
                                                            # if base_design_name == "Design_5":
                                                            #     print(macro_name, DSP_name2size[macroid], region_str)
                                                        elif macro_name in constraint_dict_DSP.keys():
                                                            con_DSP.append(constraint_dict_DSP[macro_name])
                                                        else:
                                                            con_DSP.append(whole_region)
                                                    elif 'BRAM' in macro_name:
                                                        BRAM.append(macro_name)
                                                        place_BRAM.append(dict_BRAM[site_x+','+site_y])
                                                        # cascade macro (not exceed the boundary)
                                                        if "CASCADE" in macro_name:
                                                            left  = str(0.0)
                                                            right = str(1.0)
                                                            macroid = node2placementunitid[macro_name]
                                                            up = str((300.0-BRAM_name2size[macroid]*5.0)/300.0)
                                                            down = str(0.0)
                                                            region = [left, down, right, up]
                                                            region_str = ",".join(region)
                                                            con_BRAM.append(region_str)
                                                                # if base_design_name == "Design_5":
                                                                #     print(macro_name, BRAM_name2size[macroid], region_str)
                                                            # simple macros with regional constraints
                                                        elif macro_name in constraint_dict_BRAM.keys():
                                                            con_BRAM.append(constraint_dict_BRAM[macro_name])
                                                        else:
                                                            con_BRAM.append(whole_region)
                                                    else:
                                                        print('Not a macro, ignore...')
                                                        continue

                                                # if len(DSP_order) == len(place_DSP):
                                                if True:
                                                    DSP = " ".join(DSP)
                                                    place_DSP = " ".join(place_DSP)
                                                    con_DSP = " ".join(con_DSP)
                                                    BRAM = " ".join(BRAM)
                                                    place_BRAM = " ".join(place_BRAM)
                                                    con_BRAM = " ".join(con_BRAM)

                                                    dataset_sol.write(design_name+' | ')
                                                    # dataset_sol.write(DSP+' | ')
                                                    dataset_sol.write(place_DSP+' | ')
                                                    # dataset_sol.write(BRAM+' | ')
                                                    dataset_sol.write(place_BRAM+'\n')
                                                    # dataset_sol.write(URAM+' | ')
                                                    # dataset_sol.write(place_URAM+'\n')

                                                    dataset_con.write(design_name+' | ')
                                                    # dataset_con.write(DSP+' | ')
                                                    dataset_con.write(con_DSP+' | ')
                                                    # dataset_con.write(BRAM+' | ')
                                                    dataset_con.write(con_BRAM+'\n')

                                                    dataset_met.write(design_name+' | ')


                                                    if sub_folder in f_metric_dict:
                                                        dataset_met.write(f_metric_dict[sub_folder]+'\n')
                                                    else:
                                                        dataset_met.write('none'+'\n')

                                                    dataset_net.write(design_name+' | ')
                                                    dataset_net.write(base_design_name+'\n')


                                                    dataset_order.write(design_name+' | ')
                                                    dataset_order.write(DSP_order_sorted+' | ')
                                                    dataset_order.write(BRAM_order_sorted+'\n')
                                                

                                # raise NotImplementedError

                                                

                                        
