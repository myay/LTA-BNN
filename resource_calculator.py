import math

# crossbar settings
# crossbar columns
m = 64
# xnor gates per column
n = 64

alphas_VGG3 = [64, 2048]
betas_VGG3 = [576, 3136]
deltas_VGG3 = [196, 1]

alphas_VGG7 = [128, 256, 256, 512, 512, 1024]
betas_VGG7 = [1152, 1152, 2304, 2304, 4608, 8192]
deltas_VGG7 = [1024, 256, 256, 64, 64, 1]

# acc: accumulator
# dcomp: digital comparator
# acomp: analog comparator
# adc: analog to digital converter
# A: Area, E: Energy, L: Latency

# crossbar
# latency of xnor gate
L_xnor = 704
# energy usage of one crossbar column
E_col = 1.32266666667

# ADC and analog comparator
A_acomp = 78
A_adc = 2000

E_acomp = 0.163
E_adc = 2.55

L_acomp = 74
L_adc = 1000

# ------ AREA FUNCTIONS ------

def VGG3_sota_area():
    layers = 2
    # for beta=3136
    A_acc = 419.2
    A_reg = 183.12
    A_dcomp = 251.2
    area = m*(A_acomp + A_adc + A_acc + deltas_VGG3[0]*A_reg + A_dcomp)
    return area

def VGG3_lta_area():
    layers = 2
    area = (m+1)*A_acomp
    return area

def VGG3_lta_2_area():
    layers = 2
    # find out how many neurons can be processed in parallel in the cb
    multipliers = []
    for beta in betas_VGG3:
        multipliers.append(math.floor((m*n)/beta))
    multiplier_max = max(multipliers)
    area = (m)*A_acomp + multiplier_max*A_acomp
    return area

def VGG7_sota_area():
    # for beta=8192
    A_acc = 1014.77
    A_reg = 488.78
    A_dcomp = 677.77
    # configuration to be able to process all operations with one crossbar:
    # (beta = 8192, delta = 1024)
    area = m*(A_acomp + A_adc + A_acc + deltas_VGG7[0]*A_reg + A_dcomp)
    return area

def VGG7_lta_area():
    # for beta=8192
    A_acc = 84.37
    A_reg = 41.13
    A_dcomp = 39.49
    # configuration to be able to process all operations with one crossbar:
    # (beta = 8192, delta = 1024)
    area = (m+1)*A_acomp + A_adc + A_acc + A_reg*deltas_VGG7[0] + A_dcomp
    return area

def VGG7_lta_2_area():
    # for beta=8192
    A_acc = 84.37
    A_reg = 41.13
    A_dcomp = 39.49

    multipliers = []
    for beta in betas_VGG7:
        multipliers.append(math.floor((m*n)/beta))
    # replace 0s by 1s
    multipliers = [1 if x == 0 else x for x in multipliers]
    multiplier_max = max(multipliers)

    # configuration to be able to process all operations with one crossbar:
    # (beta = 8192, delta = 1024)
    area = (m)*A_acomp + multiplier_max*A_acomp + A_adc + A_acc + A_reg*deltas_VGG7[0] + A_dcomp
    return area

# ------ ENERGY FUNCTIONS ------
def VGG3_sota_energy():
    layers = 2
    E_cb = E_col*m
    # for beta=3136
    E_acc = 0.40
    E_reg = 0.34 # approximation for now
    E_dcomp = 0.26

    energy = 0
    for i in range(0,layers):
        # print(i)
        cb_inv = deltas_VGG3[i]*math.ceil(alphas_VGG3[i]/m)*math.ceil(betas_VGG3[i]/n)
        energy_layer = cb_inv*E_col*m + cb_inv*m*(E_adc + E_acc + E_reg*deltas_VGG3[i]) + math.ceil(alphas_VGG3[i]/m)*m*deltas_VGG3[i]*E_dcomp
        energy += energy_layer
        # print("Energy l", i, energy_layer)
    return energy

def VGG3_lta_energy():
    layers = 2

    energy =  0
    for i in range(0,layers):
        # calculate the number of columns to be activated
        nr_cols = math.ceil(betas_VGG3[i]/m)
        cb_inv = deltas_VGG3[i]*alphas_VGG3[i]*math.ceil(betas_VGG3[i]/(m*n))
        energy_layer = cb_inv*E_col*nr_cols + (m+1)*cb_inv*E_acomp
        energy += energy_layer
        # print("Energy l", i, energy_layer)
    return energy

def VGG3_lta_2_energy():
    layers = 2

    multipliers = []
    for beta in betas_VGG3:
        multipliers.append(math.floor((m*n)/beta))
    # replace 0s by 1s
    multipliers = [1 if x == 0 else x for x in multipliers]

    energy =  0
    for i in range(0,layers):
        # calculate the number of columns to be activated
        nr_cols = math.ceil(betas_VGG3[i]/m)*multipliers[i]
        cb_inv = math.ceil((deltas_VGG3[i]*alphas_VGG3[i]*math.ceil(betas_VGG3[i]/(m*n)))/multipliers[i])
        energy_layer = cb_inv*E_col*nr_cols + (m)*cb_inv*E_acomp + multipliers[i]*cb_inv*E_acomp
        energy += energy_layer
        # print("Energy l", i, energy_layer)
    return energy

def VGG7_sota_energy():
    layers = 6
    E_cb = E_col*m
    # for beta=8192
    E_acc = 1.18
    E_reg = 0.78
    E_dcomp = 0.67

    energy =  0
    for i in range(0,layers):
        # print(i)
        cb_inv = deltas_VGG7[i]*math.ceil(alphas_VGG7[i]/m)*math.ceil(betas_VGG7[i]/n)
        energy_layer = cb_inv*E_col*m + cb_inv*m*(E_adc + E_acc + E_reg*deltas_VGG7[i]) + math.ceil(alphas_VGG7[i]/m)*m*deltas_VGG7[i]*E_dcomp
        energy += energy_layer
        # print("Energy l", i, energy_layer)
    return energy

def VGG7_lta_energy():
    layers = 6
    # for beta=8192
    E_acc = 0.168
    E_reg = 0.112
    E_dcomp = 0.053

    energy =  0
    for i in range(0,layers):
        # print(i)
        cb_inv = deltas_VGG7[i]*alphas_VGG7[i]*math.ceil(betas_VGG7[i]/(m*n))
        energy_layer = 0
        if betas_VGG7[i] <= m*n:
            nr_cols = math.ceil(betas_VGG7[i]/n)
            energy_layer = cb_inv*E_col*nr_cols + (m+1)*cb_inv*E_acomp
        else:
            energy_layer = cb_inv*E_col*m + cb_inv*(m*E_acomp + E_adc + E_reg*deltas_VGG7[i] + E_acc) + alphas_VGG7[i]*deltas_VGG7[i]*E_dcomp
        energy += energy_layer
        # print("Energy l", i, energy_layer)
    return energy

def VGG7_lta_2_energy():
    layers = 6
    # for beta=8192
    E_acc = 0.168
    E_reg = 0.112
    E_dcomp = 0.053

    multipliers = []
    for beta in betas_VGG7:
        multipliers.append(math.floor((m*n)/beta))
    # replace 0s by 1s
    multipliers = [1 if x == 0 else x for x in multipliers]

    energy =  0
    for i in range(0,layers):
        # print(i)
        cb_inv = math.ceil((deltas_VGG7[i]*alphas_VGG7[i]*math.ceil(betas_VGG7[i]/(m*n)))/multipliers[i])
        energy_layer = 0
        if betas_VGG7[i] <= m*n:
            nr_cols = math.ceil(betas_VGG7[i]/n)*multipliers[i]
            energy_layer = cb_inv*E_col*nr_cols + (m)*cb_inv*E_acomp + multipliers[i]*cb_inv*E_acomp
        else:
            energy_layer = cb_inv*E_col*m + cb_inv*(m*E_acomp + E_adc + E_reg*deltas_VGG7[i] + E_acc) + alphas_VGG7[i]*deltas_VGG7[i]*E_dcomp
        energy += energy_layer
        # print("Energy l", i, energy_layer)
    return energy

# ------ LATENCY FUNCTIONS ------
def VGG3_sota_latency():
    layers = 2

    L_acc = 250
    L_reg = 250
    L_dcomp = 250

    latency = 0
    for i in range(0,layers):
        # print(i)
        cb_inv = deltas_VGG3[i]*math.ceil(alphas_VGG3[i]/m)*math.ceil(betas_VGG3[i]/n)
        latency_layer = cb_inv*L_xnor + cb_inv*(L_adc + L_acc + L_reg) + math.ceil(alphas_VGG3[i]/m)*deltas_VGG3[i]*L_dcomp
        latency += latency_layer
        # print("Energy l", i, energy_layer)
    return latency

def VGG3_lta_latency():
    layers = 2

    latency = 0
    for i in range(0,layers):
        cb_inv = deltas_VGG3[i]*alphas_VGG3[i]*math.ceil(betas_VGG3[i]/(m*n))
        latency_layer = cb_inv*L_xnor + 2*cb_inv*L_acomp
        latency += latency_layer
        # print("Energy l", i, energy_layer)
    return latency

def VGG3_lta_2_latency():
    layers = 2

    multipliers = []
    for beta in betas_VGG3:
        multipliers.append(math.floor((m*n)/beta))
    # replace 0s by 1s
    multipliers = [1 if x == 0 else x for x in multipliers]

    latency = 0
    for i in range(0,layers):
        cb_inv = math.ceil((deltas_VGG3[i]*alphas_VGG3[i]*math.ceil(betas_VGG3[i]/(m*n)))/multipliers[i])
        latency_layer = cb_inv*L_xnor + 2*cb_inv*L_acomp
        latency += latency_layer
        # print("Energy l", i, energy_layer)
    return latency

def VGG7_sota_latency():
    layers = 6
    L_acc = 250
    L_reg = 250
    L_dcomp = 250

    latency = 0
    for i in range(0,layers):
        # print(i)
        cb_inv = deltas_VGG7[i]*math.ceil(alphas_VGG7[i]/m)*math.ceil(betas_VGG7[i]/n)
        latency_layer = cb_inv*L_xnor + cb_inv*(L_adc + L_acc + L_reg) + math.ceil(alphas_VGG7[i]/m)*deltas_VGG7[i]*L_dcomp
        latency += latency_layer
        # print("Energy l", i, energy_layer)
    return latency

def VGG7_lta_latency():
    layers = 6
    L_acc = 250
    L_reg = 250
    L_dcomp = 250

    latency = 0
    for i in range(0,layers):
        # print(i)
        cb_inv = deltas_VGG7[i]*alphas_VGG7[i]*math.ceil(betas_VGG7[i]/(m*n))
        latency_layer = 0
        if betas_VGG7[i] <= m*n:
            latency_layer = cb_inv*L_xnor + 2*cb_inv*L_acomp
        else:
            latency_layer = cb_inv*L_xnor + cb_inv*(L_acomp + L_adc + L_acc + L_reg) + alphas_VGG7[i]*deltas_VGG7[i]*L_dcomp
        latency += latency_layer
        # print("Energy l", i, energy_layer)
    return latency

def VGG7_lta_2_latency():
    layers = 6
    L_acc = 250
    L_reg = 250
    L_dcomp = 250

    multipliers = []
    for beta in betas_VGG7:
        multipliers.append(math.floor((m*n)/beta))
    # replace 0s by 1s
    multipliers = [1 if x == 0 else x for x in multipliers]

    latency = 0
    for i in range(0,layers):
        # print(i)
        cb_inv = math.ceil((deltas_VGG7[i]*alphas_VGG7[i]*math.ceil(betas_VGG7[i]/(m*n)))/multipliers[i])
        latency_layer = 0
        if betas_VGG7[i] <= m*n:
            latency_layer = cb_inv*L_xnor + multipliers[i]*2*cb_inv*L_acomp
        else:
            latency_layer = cb_inv*L_xnor + cb_inv*(L_acomp + L_adc + L_acc + L_reg) + alphas_VGG7[i]*deltas_VGG7[i]*L_dcomp
        latency += latency_layer
        # print("Energy l", i, energy_layer)
    return latency

def main():
    print("--- AREA ---")
    area = VGG3_sota_area()
    print(f"VGG3 SOTA Area: {area} micrometer^2")
    area = VGG3_lta_area()
    print(f"VGG3 LTA Area: {area} micrometer^2")
    area = VGG3_lta_2_area()
    print(f"VGG3 LTA-MU Area: {area} micrometer^2")
    area = VGG7_sota_area()
    print(f"VGG7 SOTA Area: {area} micrometer^2")
    area = VGG7_lta_area()
    print(f"VGG7 LTA Area: {area} micrometer^2")
    area = VGG7_lta_2_area()
    print(f"VGG7 LTA-MU Area: {area} micrometer^2")

    print("--- ENERGY ---")
    energy = VGG3_sota_energy()
    print(f"VGG3 SOTA Energy: {energy} pJ")
    energy = VGG3_lta_energy()
    print(f"VGG3 LTA Energy: {energy} pJ")
    energy = VGG3_lta_2_energy()
    print(f"VGG3 LTA-MU Energy: {energy} pJ")
    energy = VGG7_sota_energy()
    print(f"VGG7 SOTA Energy: {energy} pJ")
    energy = VGG7_lta_energy()
    print(f"VGG7 LTA Energy: {energy} pJ")
    energy = VGG7_lta_2_energy()
    print(f"VGG7 LTA-MU Energy: {energy} pJ")

    print("--- LATENCY ---")
    latency = VGG3_sota_latency()
    print(f"VGG3 SOTA Latency: {latency} ps")
    latency = VGG3_lta_latency()
    print(f"VGG3 LTA Latency: {latency} ps")
    latency = VGG3_lta_2_latency()
    print(f"VGG3 LTA-MU Latency: {latency} ps")
    latency = VGG7_sota_latency()
    print(f"VGG7 SOTA Latency: {latency} ps")
    latency = VGG7_lta_latency()
    print(f"VGG7 LTA Latency: {latency} ps")
    latency = VGG7_lta_2_latency()
    print(f"VGG7 LTA-MU Latency: {latency} ps")

if __name__ == "__main__":
    main()
