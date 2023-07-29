import os
import pdb

def find_benchmark(dataset_root, benchmark):
    bm_to_root = {
        "mlcad2023": dataset_root
    }
    root = bm_to_root[benchmark]
    # Firstly exclude some testcases from the benchmark
    # uram_benchmark = [3,4,8,9,13,14,18,19,23,24,28,29,33,34,39,38,43,44,48,49,63,64,68,69,73,74,78,79,83,84,88,89,93,94,98,99,103,104,123,124,128,129,133,134,138,139,143,144,148,149,153,154,158,159,183,184,188,189,193,194,198,199,203,204,208,209,213,214,218,219,223,224]
    #benchmark_ignore = [106,107,110,111,112,115,116,117,161,162,165,166,167,170,171,172,175,176,180,226,227,230,231,232,235,236,240,51,52,55,56,57,60]
    benchmark_ignore = []
    #for id in range(0,176):
    #benchmark_ignore.append(id)
    #uram_benchmark_str = []
    benchmark_ignore_str = []
    #for id in range(len(uram_benchmark)):
    #    uram_file_name = "Design_"+str(uram_benchmark[id])
    #    uram_benchmark_str.append(uram_file_name)
    for id in range(len(benchmark_ignore)):
        benchmark_ignore_filename = "Design_"+str(benchmark_ignore[id])
        benchmark_ignore_str.append(benchmark_ignore_filename)
    file_col = os.listdir(root)
    file_col_prior = []
    for id in range(len(file_col)):
        if file_col[id] in benchmark_ignore_str:
            continue
        file_col_prior.append(file_col[id])
    all_designs = [i for i in file_col_prior if os.path.isdir(os.path.join(root, i))]
    #pdb.set_trace()
    all_designs.sort(key=lambda x: int(x[7:]))
    #uram_designs = [i for i in uram_benchmark_str if os.path.isdir(os.path.join(root, i))]
    #all_designs.extend(uram_designs)
    return root, all_designs

def get_single_design_params(dataset_root, benchmark, design_name, placement=None):
    if benchmark == "mlcad2023":
        return single_mlcad2023(dataset_root, design_name, placement)

def get_multiple_design_params(dataset_root, benchmark):
    root, all_designs = find_benchmark(dataset_root, benchmark)
    params_mul = []
    for design_name in all_designs:
        params = get_single_design_params(dataset_root, benchmark, design_name)
        params_mul.append(params)
    #pdb.set_trace()
    return params_mul

def single_mlcad2023(dataset_root, design_name, placement=None):
    benchmark = "mlcad2023"
    root, all_designs = find_benchmark(dataset_root, benchmark)
    if design_name not in all_designs:
        raise ValueError("Design Name %s should in %s" % (design_name, root))
    params = {
        "benchmark": benchmark,
        "bookshelf_variety": "mlcad2023",
        "nodes": "%s/%s/design.nodes" % (root, design_name),
        "nets": "%s/%s/design.nets" % (root,design_name),
        "lib": "%s/%s/design.lib" % (root,design_name),
        "sitemap": "%s/%s/design.scl" % (root,design_name),
        "fixed": "%s/%s/design.pl" % (root,design_name),
        "cascade_shape": "%s/%s/design.cascade_shape" % (root,design_name),
        "cascade_instance": "%s/%s/design.cascade_shape_instances" % (root, design_name),
        "region_constr": "%s/%s/design.regions" % (root, design_name),
        #"sample": "/data/ssd/qluo/benchmark/checkpoint/place_results/%s/solution_gt.pl" % design_name,
        #"sample": "/data/ssd/qluo/benchmark/checkpoint/design_2_new/solution_v2.pl",
        #"sample": "/data/ssd/qluo/benchmark/checkpoint/design_2/solution_v2.pl",
        #"sample": "/data/ssd/qluo/cumple/result/8/log/Design_2/solution.pl",
        #"sample": "/data/ssd/qluo/AMF-Placer-new/AMFPlacer-MLCAD/build/%s/solution_newrouteadj.pl" % design_name,
        "sample": "/data/ssd/qluo/docker_practice/Cumple/DiffMP/result/Design_12_solution.pl",
        "sample_dir": "/data/ssd/qluo/benchmark/checkpoint/place_results/%s/" % design_name,  
        #"sample_dir": "/data/ssd/qluo/benchmark/checkpoint/place_results_test/design_5/",
        #"sample_dir": "/data/ssd/qluo/benchmark/mlcad2023_v2/%s/netlist_feature" % design_name,
        "design_name":design_name,
    }
    #print(params)
    return params

def get_custom_design_params(args):
    params = dict([
        [item.strip() for item in token.strip().split(":")] 
        for token in args.custom_path.split(",") if len(token) > 0
    ])
    if "benchmark" not in params.keys():
        raise ValueError("Cannot find 'benchmark' in args.custom_path")
    if "design_name" not in params.keys():
        raise ValueError("Cannot find 'design_name' in args.custom_path")
    args.dataset = params["benchmark"]
    args.design_name = params["design_name"]
    return params

def checkparam(param):
    if not os.path.exists(param["nodes"]):
        logger.info(param["nodes"]+" is not in the "+param["design_name"])
        return False
    else:
        return True
