#conding=utf8  
import os
import json
import numpy as np

if not os.path.exists("./dataset/site2idx.json"):
    print('Creating site-to-index mapping...')
    dict_DSP = {}
    dict_BRAM = {}
    dict_URAM = {}
    count_DSP = 1
    count_BRAM = 1
    count_URAM = 1
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
            elif 'URAM' in site_map[i]:
                site_x = site_map[i].strip().split(' ')[0]
                site_y = site_map[i].strip().split(' ')[1]
                dict_URAM[site_x+','+site_y] = str(count_URAM)
                count_URAM += 1
    with open('./dataset/site2idx.json', mode='w', encoding='utf-8') as f:
        json.dump([dict_DSP, dict_BRAM, dict_URAM], f)

else:
    print('Loading site-to-index mapping...')
    with open('./dataset/site2idx.json', mode='r', encoding='utf-8') as f:
        dicts = json.load(f)
        dict_DSP = dicts[0]
        dict_BRAM = dicts[1]
        dict_URAM = dicts[2]
        print('len(dict_DSP):', len(dict_DSP))
        print('len(dict_BRAM):', len(dict_BRAM))
        print('len(dict_URAM):', len(dict_URAM))


if __name__ == '__main__':

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

                                    node_feature_path = os.path.join(root, base_design_name, 'netlist_feature/PU_feature.txt')
                                    with open(node_feature_path, mode='r', encoding='utf-8') as node_feature:
                                        node_feature = node_feature.readlines()
                                        BRAM_size_list = []
                                        BRAM_degree_list = []
                                        DSP_size_list = []
                                        DSP_degree_list = []
                                        
                                        for i_ in range(len(node_feature)):
                                            macro_name = node_feature[i_].strip().split(' ')[0]
                                            macro_feature = node_feature[i_].strip().split(' ')[-1].split(',')
                                            
                                            macro_size = macro_feature[5]
                                            macro_degree = macro_feature[6]

                                            if macro_feature[0] == '1': ### BRAM
                                                BRAM_size_list.append(int(macro_size))
                                                BRAM_degree_list.append(int(macro_degree))
                                            elif macro_feature[1] == '1': ### DSP
                                                DSP_size_list.append(int(macro_size))
                                                DSP_degree_list.append(int(macro_degree))
                                            else:
                                                raise NotImplementedError("Unrecognizable macro type")

                                        BRAM_order = [i for i in range(len(BRAM_size_list))]
                                        DSP_order = [i for i in range(len(DSP_size_list))]
                                        # print('DSP_order:', DSP_order)

                                        BRAM_zip = zip(BRAM_order, BRAM_size_list, BRAM_degree_list)
                                        DSP_zip = zip(DSP_order, DSP_size_list, DSP_degree_list)
                                        # print('DSP_zip:', DSP_zip)

                                        BRAM_order_sorted = sorted(BRAM_zip, key=lambda x:(x[1],x[2]), reverse=True)
                                        DSP_order_sorted = sorted(DSP_zip, key=lambda x:(x[1],x[2]), reverse=True)
                                        # print('DSP_order_sorted1:', DSP_order_sorted)

                                        BRAM_order_sorted = [list(x) for x in zip(*BRAM_order_sorted)][0]
                                        DSP_order_sorted = [list(x) for x in zip(*DSP_order_sorted)][0]
                                        # print('DSP_order_sorted2:', DSP_order_sorted)

                                        # raise NotImplementedError
                                        BRAM_order_sorted = ",".join([str(i) for i in BRAM_order_sorted])
                                        DSP_order_sorted = ",".join([str(i) for i in DSP_order_sorted])


                                    region_constraint_path = os.path.join(root, base_design_name, 'design.regions')
                                    metric_path = os.path.join(root, base_design_name, 'solution', 'MacroHPWL.txt')

                                    with open(metric_path, mode='r', encoding='utf-8') as f_metric:
                                        f_metric = f_metric.readlines()
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

                                                    constraint_rect = np.array([int(i) for i in f_region[i+1].strip().split(' ')[1:]]) / np.array([206,300,206,300])
                                                    constraint_rect = np.clip(constraint_rect, 0., 1.).tolist()
                                                    constraint_rect = ','.join([str(i) for i in constraint_rect])

                                                    idx2region[constraint_idx] = constraint_rect
                                                if 'InstanceToRegionConstraintMapping' in f_region[i] and '#' not in f_region[i]:
                                                    break
                                            
                                            constraint_dict_DSP = {}
                                            constraint_dict_BRAM = {}
                                            constraint_dict_URAM = {}
                                            for i in range(len(f_region)):
                                                if 'DSP' in f_region[i]:
                                                    macro_name = f_region[i].strip().split(' ')[0]
                                                    constraint_idx = f_region[i].strip().split(' ')[-1]
                                                    constraint_dict_DSP[macro_name] = idx2region[constraint_idx]
                                                elif 'BRAM' in f_region[i]:
                                                    macro_name = f_region[i].strip().split(' ')[0]
                                                    constraint_idx = f_region[i].strip().split(' ')[-1]
                                                    constraint_dict_BRAM[macro_name] = idx2region[constraint_idx]
                                                elif 'URAM' in f_region[i]:
                                                    macro_name = f_region[i].strip().split(' ')[0]
                                                    constraint_idx = f_region[i].strip().split(' ')[-1]
                                                    constraint_dict_URAM[macro_name] = idx2region[constraint_idx]

                                    for sub_root, sub_ds, sub_fs in os.walk('/research/d1/gds/qluo22/dataset_col/mlcad2023_v2/%s/solution' % base_design_name):
                                        for sub_f in sub_fs:
                                            if 'solution_gt' in sub_f:
                                                solution_path = os.path.join(sub_root, sub_f)
                                                design_name = base_design_name + '@' + sub_f.split('.')[0]

                                                DSP = []
                                                BRAM = []
                                                URAM = []
                                                place_DSP = []
                                                place_BRAM = []
                                                place_URAM = []
                                                con_DSP = []
                                                con_BRAM = []
                                                con_URAM = []
                                                # whole_region = '0,0,206,300'
                                                whole_region = '0,0,1,1'  ### Normalized
                                                
                                                with open(solution_path, 'r') as f_sol:
                                                    f_sol = f_sol.readlines()

                                                    for i in range(len(f_sol)):
                                                        macro_name = f_sol[i].strip().split(' ')[0]
                                                        site_x = f_sol[i].strip().split(' ')[1]
                                                        site_y = f_sol[i].strip().split(' ')[2]
                                                        if 'DSP' in macro_name:
                                                            DSP.append(macro_name)
                                                            place_DSP.append(dict_DSP[site_x+','+site_y])
                                                            # Special operation for DSP_config cells
                                                            if "DSP_config" in macro_name:
                                                                macro_name_col = macro_name.split("/")
                                                                macro_name = macro_name_col[0] + "/" + macro_name_col[1]
                                                            if macro_name in constraint_dict_DSP.keys():
                                                                con_DSP.append(constraint_dict_DSP[macro_name])
                                                            else:
                                                                con_DSP.append(whole_region)
                                                        elif 'BRAM' in macro_name:
                                                            BRAM.append(macro_name)
                                                            place_BRAM.append(dict_BRAM[site_x+','+site_y])
                                                            if macro_name in constraint_dict_BRAM.keys():
                                                                con_BRAM.append(constraint_dict_BRAM[macro_name])
                                                            else:
                                                                con_BRAM.append(whole_region)
                                                        elif 'URAM' in macro_name:
                                                            URAM.append(macro_name)
                                                            place_URAM.append(dict_URAM[site_x+','+site_y])
                                                            if macro_name in constraint_dict_URAM.keys():
                                                                con_URAM.append(constraint_dict_URAM[macro_name])
                                                            else:
                                                                con_URAM.append(whole_region)
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
                                                        URAM = " ".join(URAM)
                                                        place_URAM = " ".join(place_URAM)
                                                        con_URAM = " ".join(con_URAM)

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
                                                        # dataset_con.write(URAM+' | ')
                                                        # dataset_con.write(con_URAM+'\n')

                                                        dataset_met.write(design_name+' | ')
                                                        if sub_f.split('.')[0] in f_metric_dict:
                                                            dataset_met.write(f_metric_dict[sub_f.split('.')[0]]+'\n')
                                                        else:
                                                            dataset_met.write('none'+'\n')

                                                        dataset_net.write(design_name+' | ')
                                                        dataset_net.write(base_design_name+'\n')


                                                        dataset_order.write(design_name+' | ')
                                                        dataset_order.write(DSP_order_sorted+' | ')
                                                        dataset_order.write(BRAM_order_sorted+'\n')
                                                    
                                                    else:
                                                        continue
                                                

                                # raise NotImplementedError

                                                

                                        
