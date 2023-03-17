#!/bin/bash

ghdl -a xnor_gate.vhdl
ghdl -a xnor_gate_array.vhdl
ghdl -a xnor_array_columns.vhdl
ghdl -a controller_lta.vhdl
ghdl -a controller_lta_tb.vhdl
ghdl -e controller_lta_tb
ghdl -r controller_lta_tb --vcd=testbench.vcd
# gtkwave testbench.vcd
