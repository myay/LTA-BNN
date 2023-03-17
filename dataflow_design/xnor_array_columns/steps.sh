#!/bin/bash

ghdl -a ../xnor/xnor_gate.vhdl
ghdl -a ../xnor_array/xnor_gate_array.vhdl
ghdl -a xnor_array_columns.vhdl
ghdl -a xnor_array_columns_tb.vhdl
ghdl -e xnor_array_columns_tb
ghdl -r xnor_array_columns_tb --vcd=testbench.vcd
# gtkwave testbench.vcd
