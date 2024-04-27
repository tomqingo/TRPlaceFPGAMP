#!/bin/bash

designs=(2 5 6 7 10 11 12 15 16 17 20 21 22 25 26 27 30 31 32 35 36 37 40 41 42 45 46 47 50 51 52 55 56 57 60 61 62 65 66 67 70 71 72 75 76 77 80 81 82 85 86 87 90 91 92 95 96 97 100 101 102 105 106 107 110 111 112 115 116 117 120 121 122  125 126 127 130 131 132 135 136 137 140 141 142 145 147 150 151 152 155 156 160 161 162 165 166 167 170 171 172 175 176 180 181 182 185 186 187 190 191 192 195 196 197 200 201 202 205 206 207 210 211 212 215 216 217 220 221 222 225 226 227 230 231 232 235 236 237 240)

cd ~/disk/AMFPlacer-MLCAD/benchmarks/mlcad2023_v2
for design in ${designs[@]:0:140}
do
	cnt=$(( cnt + 1))
	echo $cnt/${#designs[@]} "Design_"$design
	cd Design_$design
    mkdir -p vivado_solution/solution_1
    cp -r /uac/gds/qluo22/disk/AMFPlacer-MLCAD/run/vivado_place/d$design/* vivado_solution/solution_1/
    cd ..
done