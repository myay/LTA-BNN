library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use ieee.numeric_std.all;

entity controller_sota_tb is
end controller_sota_tb;

architecture test of controller_sota_tb is
  component controller_sota
    port(
      clk: in std_logic;
      xnor_input: in std_logic_vector(63 downto 0);
      weights: in std_logic_vector(4095 downto 0);
      o_result: out std_logic_vector(4095 downto 0)
    );
  end component;

signal input_1: std_logic_vector(63 downto 0) := (others => '0');
signal weights: std_logic_vector(4095 downto 0) := (others => '0');
signal output_cc: std_logic_vector(4095 downto 0) := (others => '0');

signal clk_t: std_logic := '0';
constant clk_period : time := 2 ns;
constant max_clock_cyles: integer := 40;

begin
  cc_inst: controller_sota
    port map(
      clk => clk_t,
      xnor_input => input_1,
      weights => weights,
      o_result => output_cc
    );

    process begin

      input_1 <= "1010101010101010101010101010101010101010101010101010101010101010";
      wait for 2 ns;

      input_1 <= "0101010101010101010101010101010101010101010101010101010101010101";
      wait for 2 ns;

      input_1 <= "0101010101010101010101010101010101010101010101010101010101010101";
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
