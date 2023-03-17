#!/bin/bash

ghdl -a ../xnor/xnor_gate.vhdl
ghdl -a ../xnor_array/xnor_gate_array.vhdl
ghdl -a ../xnor_array_columns/xnor_array_columns.vhdl
ghdl -a controller_sota.vhdl
ghdl -a controller_sota_tb.vhdl
ghdl -e controller_sota_tb
ghdl -r controller_sota_tb --vcd=testbench.vcd
# gtkwave testbench.vcd
