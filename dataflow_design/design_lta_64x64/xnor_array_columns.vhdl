library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use ieee.numeric_std.all;

entity xnor_array_columns is
  port(
    xnor_inputs_1 : in std_logic_vector(4095 downto 0); -- First inputs
    xnor_inputs_2 : in std_logic_vector(4095 downto 0); -- Second inputs
    o_result : out std_logic_vector(4095 downto 0) -- Outputs
  );
end xnor_array_columns;

architecture rtl of xnor_array_columns is
begin
  -- Create a certain number of computing columns
  cc_vm_gen: for i in 1 to 64 generate
    cc_inst: entity work.xnor_gate_array(rtl)
      port map(
        xnor_inputs_1 => xnor_inputs_1(64*i-1 downto 64*(i-1)),
        xnor_inputs_2 => xnor_inputs_2(64*i-1 downto 64*(i-1)),
        xnor_outputs => o_result(64*i-1 downto 64*(i-1))
      );
  end generate;
end rtl;
