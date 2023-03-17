library IEEE;
use IEEE.STD_LOGIC_1164.ALL;

package array_pack is
  type array_2d_data is array(0 to 1) of std_logic_vector(3 downto 0);
end package;

library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use ieee.numeric_std.all;
use work.array_pack.all;

entity xnor_array_columns is
  generic(
    nr_computing_columns : integer := 2; -- Number of computing columns (m)
    nr_xnor_gates : integer := 4 -- Number of XNOR gates used in each computing (n) column
  );
  port(
    xnor_inputs_1 : in array_2d_data; -- First inputs
    xnor_inputs_2 : in array_2d_data; -- Second inputs
    o_result : out array_2d_data -- Outputs
  );
end xnor_array_columns;

architecture rtl of xnor_array_columns is
begin
  -- Create a certain number of computing columns for vm, number specified in nr_computing_columns
  cc_vm_gen: for i in 0 to nr_computing_columns-1 generate
    cc_inst: entity work.xnor_gate_array(rtl)
      generic map(
        nr_xnor_gates => nr_xnor_gates
      )
      port map(
        xnor_inputs_1 => xnor_inputs_1(i),
        xnor_inputs_2 => xnor_inputs_2(i),
        xnor_outputs => o_result(i)
      );
  end generate;
end rtl;
