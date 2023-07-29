#Load the synthesized netlist
open_checkpoint /data/ssd/qluo/benchmark/mlcad2023_v2/Design_12/design.dcp

set_param place.timingDriven false

#Place design using the bookshelf macro placement
place_design  -macro_placement /data/ssd/qluo/docker_practice/Cumple/DiffMP/result/Design_12_solution.pl -verbose

#route design
route_design -no_timing_driven -verbose

#generate routing Report
report_route_status