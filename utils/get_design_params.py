import os
import pdb

# Find all the cases of mlcad2023
def find_benchmark(dataset_root, benchmark):
    bm_to_root = {
        "mlcad2023": dataset_root
    }
    root = bm_to_root[benchmark]
    file_col = os.listdir(root)
    all_designs = [i for i in file_col if os.path.isdir(os.path.join(root, i))]
    #print(all_designs)
    all_designs.sort(key=lambda x: int(x[7:]))
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
        "macros": "%s/%s/design.macros" % (root, design_name), 
        "design_name":design_name,
    }
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

def checkparam(param, logger):
    if not os.path.exists(param["nodes"]):
        logger.info(param["nodes"]+" is not in the "+param["design_name"])
        return False
    else:
        return True
