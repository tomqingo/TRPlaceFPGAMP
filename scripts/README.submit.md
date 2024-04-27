# Requirements
The binary is tested on 
- Ubuntu 18.04 and Centos 7

# How to run
## 1. Generate macro placement solution in bookshelf (output_file name: macroplacement.pl)
```
./Cumple path-to-input-design path-to-solution
E.g.: ./Cumple benchmarks/mlcad2023/Design_2 solutions/TeamCumple/Design_2
```
## 2. Convert macro placement solution from bookshelf to tcl script (macroplacement.pl -> place_macro.tcl)
```
python macroplacement_bookshelf2vivado.py --bm_dir path-to-input-design --sol_dir path-to-solution
E.g.: python macroplacement_bookshelf2vivado.py --bm_dir benchmarks/mlcad2023/Design_2 --sol_dir solutions/TeamCumple/Design_2
```

## 3. (optional) Convert io placement solution from bookshelf to tcl script (design.pl -> place_io.tcl)
```
g++ io_map.xx -o io_map
./io_map path-to-input-design/design.pl path-to-input-design/place_io.tcl
```

## 4. Run vivado to place and route
```
set_param -name place.hardVerbose -value 735361
set dcp_path path-to-input-design/design.dcp
set io_tcl_path path-to-input-design/place_io.tcl
set macro_tcl_path path-to-solution/place_macro.tcl
#set placed_checkpoint_path "merge.placed.dcp"
#set routed_checkpoint_path "merge.routed.dcp"
puts "Opening Unrouted Checkpoint..."

open_checkpoint $dcp_path

set_param place.timingDriven false

puts "Placing IOs"
source $io_tcl_path

puts "DSPs, and BRAMs..."
source $macro_tcl_path

puts "Placing Design..."
place_design -verbose

#puts "Writing Placed Checkpoints..."
#write_checkpoint -force $placed_checkpoint_path

puts "Routing Design..."
route_design -no_timing_driven -verbose

#puts "Writing Routed Checkpoints..."
#write_checkpoint -force $routed_checkpoint_path

puts "Report Congestiong Level..."
report_design_analysis -congestion -min_congestion_level 3

puts "Reporting Routing Status..."
report_route_status
exit
```