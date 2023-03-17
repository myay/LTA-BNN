library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use ieee.numeric_std.all;
use IEEE.MATH_REAL.all;

entity controller_lta_tb is
end controller_lta_tb;

architecture test of controller_lta_tb is
  component controller_lta
    port(
      clk: in std_logic;
      xnor_input: in std_logic_vector(4095 downto 0);
      weights: in std_logic_vector(4095 downto 0);
      o_result: out std_logic_vector(4095 downto 0)
    );
  end component;

signal input_1: std_logic_vector(4095 downto 0) := (others => '0');
signal weights: std_logic_vector(4095 downto 0) := (others => '0');
signal output_cc: std_logic_vector(4095 downto 0) := (others => '0');

signal clk_t: std_logic := '0';
constant clk_period : time := 2 ns;
constant max_clock_cyles: integer := 1000;

begin
  cc_inst: controller_lta
    port map(
      clk => clk_t,
      xnor_input => input_1,
      weights => weights,
      o_result => output_cc
    );

  -- RNG process
  rng_process: process
    variable seed1, seed2 : integer := 999; -- Seeds for reproducable random numbers
    variable j : integer := 0;

    -- Function for generating random std_logic_vector
    impure function rand_lv(len : integer) return std_logic_vector is
      variable x : real; -- Returned random value in rng function
      variable rlv_val : std_logic_vector(len - 1 downto 0); -- Returned random bit string of length len
    begin
      for i in rlv_val'range loop
        uniform(seed1, seed2, x);
        if x > 0.5 then
          rlv_val(i) := '1';
        else
          rlv_val(i) := '0';
        end if;
      end loop;
      return rlv_val;
    end function;

  begin
    -- Set random weights once and don't change them
    weights <= rand_lv(4096);
    while j<max_clock_cyles loop
      -- Change input every clock cycle
      input_1 <= rand_lv(4096);
      wait for 2 ns;
      j := j+1;
    end loop;
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
