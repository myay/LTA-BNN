library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use ieee.numeric_std.all;

entity controller_lta is
  port(
    clk: in std_logic;
    xnor_input: in std_logic_vector(4095 downto 0);
    weights: in std_logic_vector(4095 downto 0);
    o_result: out std_logic_vector(4095 downto 0)
  );
end controller_lta;

architecture rtl of controller_lta is
  signal mem_i : std_logic_vector(4095 downto 0) := (others => '0');-- input (one input repeated in every entry)
  signal mem_w : std_logic_vector(4095 downto 0) := (others => '0'); -- weights
  signal mem_o : std_logic_vector(4095 downto 0) := (others => '0'); -- outputs
begin
  -- Generate component of xnor array columns
  inst_x_a_c : entity work.xnor_array_columns(rtl)
    port map(
      xnor_inputs_1 => mem_i,
      xnor_inputs_2 => mem_w,
      o_result => mem_o
    );

  -- Single in multiple out buffer (SIMO)
  process(clk) begin
    if rising_edge(clk) then
      mem_i <= xnor_input;
      mem_w <= weights;
    end if;
  end process;
  o_result <= mem_o;
end rtl;
