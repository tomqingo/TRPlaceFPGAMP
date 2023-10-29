set pl_path "solution_gt_org.pl"
set dcp_path "../design.dcp"
set dcp_place_path "design_place"

open_checkpoint $dcp_path
set_param place.timingDriven false

place_design -verbose

set fp [open $pl_path w+]

set macrotypecol {BRAM* DSP* URAM*}
foreach macrotype $macrotypecol {
	current_instance
	foreach macroinst [get_cells $macrotype] {
		current_instance $macroinst
		foreach inst [get_cells -filter {NAME =~ *my_dsp} -hier] {
		  puts $fp "[get_cells $inst] [get_tiles  -of_objects [get_sites -of_objects [get_cells $inst]]] [get_sites -of_objects [get_cells $inst]]"
		}
		foreach inst [get_cells -filter {NAME =~ *_instance_name* && Name =~ *i_primitive} -hier] {
		  puts $fp "[get_cells $inst] [get_tiles  -of_objects [get_sites -of_objects [get_cells $inst]]] [get_sites -of_objects [get_cells $inst]]"
		}
		foreach inst [get_cells -filter {TYPE =~ BLOCKRAM} -hier] {
		  puts $fp "[get_cells $inst] [get_tiles  -of_objects [get_sites -of_objects [get_cells $inst]]] [get_sites -of_objects [get_cells $inst]]"
 	  }
   current_instance
	}
}

close $fp

write_checkpoint dcp_place_path -force
quit