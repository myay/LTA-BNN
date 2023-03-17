library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use ieee.numeric_std.all;
use work.array_pack.all;

entity controller_lta is
  generic(
    nr_computing_columns : integer := 2;
    nr_xnor_gates: integer := 4
  );
  port(
    clk: in std_logic;
    -- xnor_input: in std_logic_vector(nr_xnor_gates-1 downto 0);
    xnor_input : in array_2d_data;
    weights: in array_2d_data;
    o_result: out array_2d_data
  );
end controller_lta;

architecture rtl of controller_lta is
  signal mem_i : array_2d_data := (others => (others => '0'));-- input (one input repeated in every entry)
  signal mem_w : array_2d_data := (others => (others => '0')); -- weights
  signal mem_o : array_2d_data := (others => (others => '0')); -- outputs
begin
  -- Generate component of xnor array columns
  inst_x_a_c : entity work.xnor_array_columns(rtl)
    generic map(
      nr_computing_columns => nr_computing_columns,
      nr_xnor_gates => nr_xnor_gates
    )
    port map(
      xnor_inputs_1 => mem_i,
      xnor_inputs_2 => mem_w,
      o_result => mem_o
    );

  -- Single in multiple out buffer (SIMO)
  process(clk) begin
    mem_i <= xnor_input;
  end process;
end rtl;
