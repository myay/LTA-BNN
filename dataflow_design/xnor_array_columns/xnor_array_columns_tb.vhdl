library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use ieee.numeric_std.all;
use work.array_pack.all;

entity xnor_array_columns_tb is
end xnor_array_columns_tb;

architecture test of xnor_array_columns_tb is
  component xnor_array_columns
    generic(
      nr_computing_columns : integer := 2; -- Number of computing columns used (m)
      nr_xnor_gates : integer := 4 -- Number of XNOR gates used in each computing column (n)
    );
    port(
      xnor_inputs_1 : in array_2d_data; -- First inputs
      xnor_inputs_2 : in array_2d_data; -- Second inputs
      o_result : out array_2d_data -- Outputs
    );
  end component;

signal rst_t: std_logic;
signal input_1: array_2d_data := (others => (others => '0'));
signal input_2: array_2d_data := (others => (others => '0'));
signal output_cc: array_2d_data := (others => (others => '0'));

signal clk_t: std_logic := '0';
constant clk_period : time := 2 ns;
constant max_clock_cyles: integer := 40;

begin
  computing_columns_test: xnor_array_columns
    generic map(
      nr_computing_columns => 2,
      nr_xnor_gates => 4
    )
    port map(
      xnor_inputs_1 => input_1,
      xnor_inputs_2 => input_2,
      o_result => output_cc
    );

  process begin
    -- reset
    input_1(0) <= "1010";
    input_2(0) <= "1010";
    input_1(1) <= "1010";
    input_2(1) <= "1010";
    rst_t <= '1';
    wait for 2 ns;

    -- add 1
    input_1(0) <= "0101";
    input_2(0) <= "1010";
    input_1(1) <= "0101";
    input_2(1) <= "1010";
    rst_t <= '0';
    wait for 2 ns;

    -- add 2
    input_1(0) <= "0101";
    input_2(0) <= "1010";
    input_1(1) <= "0101";
    input_2(1) <= "1010";
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
