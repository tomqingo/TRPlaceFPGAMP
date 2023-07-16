import re
import argparse
import os

def get_option():
	parser = argparse.ArgumentParser("Extract Macro Location")
	parser.add_argument("--input_dir", type=str, default="/data/ssd/qluo/benchmark/mlcad2023/Design_1/place_results/", help="the folder for the initial placement results")
	parser.add_argument("--output_dir", type=str, default="/data/ssd/qluo/benchmark/checkpoint/place_results/Design_1/", help="the folder for the output placement results")
	args = parser.parse_args()
	return args

def Convert(solution_pl_org_path, solution_pl_path):
	Xtile_Xccorr_conversion = {1:2, 6:11, 10:17, 11:20, 14:26, 16:30, 19:35, 20:38, 23:44, 25:48, 28:53, 32:60, 33:63, 38:72, 39:75, 42:81, 51:98, 53:101, 55:105, 56:108, 59:114, 61:118, 64:123, 68:130, 69:133, 74:142, 75:145, 78:151, 87:168, 89:171, 90:174, 96:186, 98:189, 102:195, 105:202}
	out_str = ""
	with open(solution_pl_org_path) as f_org:
		all_lines = f_org.read().splitlines()
		for id in range(len(all_lines)):
			cur_line = all_lines[id]
			cur_line_col = cur_line.strip().split()
			macro_name = cur_line_col[0]
			macro_tile = cur_line_col[1]
			macro_site = cur_line_col[2]

			macro_tile_col = re.split("X|Y", macro_tile)
			macro_site_col = re.split("X|Y", macro_site)


			X_tile = int(macro_tile_col[-2])
			Y_tile = int(macro_tile_col[-1])

			X_corr = Xtile_Xccorr_conversion[X_tile]


			if "DSP" in macro_name:
				Y_site = int(macro_site_col[-1])
				Y_corr = int(Y_site*2.5)
				bel_corr = 0
			elif "URAM_CASCADE" in macro_name:
				Y_corr = Y_tile
				uram_name = macro_name.strip().split("/")[-1]
				uram_id = int(uram_name.strip().split("t")[-1])
				bel_corr = (uram_id-1)%4
			else:
				Y_corr = Y_tile
				bel_corr = 0


			out_str += (macro_name+" ")
			out_str += (str(X_corr)+" ")
			out_str += (str(Y_corr)+" ")
			out_str += str(bel_corr)
			out_str += "\n"

	f_out = open(solution_pl_path, "w")
	f_out.write(out_str)
	f_out.close()


if __name__=="__main__":
	args = get_option()
	if not os.path.exists(args.output_dir):
		os.mkdirs(args.output_dir)
	solution_pl_org_path = os.path.join(args.input_dir, "solution_gt_org.pl")
	solution_pl_path = os.path.join(args.output_dir, "solution_gt.pl")
	Convert(solution_pl_org_path, solution_pl_path)




