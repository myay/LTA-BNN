library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use ieee.numeric_std.all;
use work.array_pack.all;

entity controller_sota_tb is
end controller_sota_tb;

architecture test of controller_sota_tb is
  component controller_sota
    generic(
      nr_computing_columns : integer := 2;
      nr_xnor_gates: integer := 4
    );
    port(
      clk: in std_logic;
      xnor_input : in std_logic_vector(nr_xnor_gates-1 downto 0);
      weights: in array_2d_data;
      o_result: out array_2d_data
    );
  end component;

signal input_1: std_logic_vector(3 downto 0) := (others => '0');
signal weights: array_2d_data := (others => (others => '0'));
signal output_cc: array_2d_data := (others => (others => '0'));

signal clk_t: std_logic := '0';
constant clk_period : time := 2 ns;
constant max_clock_cyles: integer := 40;

begin
  cc_inst: controller_sota
    generic map(
      nr_computing_columns => 2,
      nr_xnor_gates => 4
    )
    port map(
      clk => clk_t,
      xnor_input => input_1,
      weights => weights,
      o_result => output_cc
    );

    process begin

      input_1 <= "1010";
      wait for 2 ns;

      input_1 <= "0101";
      wait for 2 ns;

      input_1 <= "0101";
      wait for 2 ns;

      wait;
    end process;

  -- Clock generation process
  clk_process: process
    variable i: integer := 0;
    begin
      while i<max_clock_cyles loop
        clk_t <= '0';
        wait for clk_period/2;  -- Signal is '0'.
        clk_t <= '1';
        wait for clk_period/2;  -- Signal is '1'
        i := i+1;
      end loop;
      wait;
    end process;

end test;
