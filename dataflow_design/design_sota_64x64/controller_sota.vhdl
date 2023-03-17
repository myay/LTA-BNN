library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use ieee.numeric_std.all;

entity controller_sota is
  port(
    clk: in std_logic;
    xnor_input: in std_logic_vector(63 downto 0);
    weights: in std_logic_vector(4095 downto 0);
    o_result: out std_logic_vector(4095 downto 0)
  );
end controller_sota;

architecture rtl of controller_sota is
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
      for i in 1 to 64 loop
        mem_i(64*i-1 downto 64*(i-1)) <= xnor_input;
      end loop;
      mem_w <= weights;
    end if;
  end process;
  o_result <= mem_o;
end rtl;
