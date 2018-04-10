# SEIR-D epidemiology model :
# Zoe Cooper -- CPSC 420
#Represents a systems dynamics model of a infectious disease.

import matplotlib.pyplot as plt
import numpy as np

def runsim(mean_infectious_duration = 5,             # days
    transmissibility = .1,                           # infections/contact
    contact_factor = 4,           
    incubation_period = 9,                          # days
    mortality_rate = .4,                        # days                                              
    plot = True):

    recovery_factor = 1/mean_infectious_duration # 1/day
    death_factor = 1/mean_infectious_duration   # 1/day
    incubation_factor = 1/incubation_period          # per days
    delta_t = .1                                    # days
    time_values = np.arange(0,160,delta_t)

    S = np.empty(len(time_values))
    I = np.empty(len(time_values))
    R = np.empty(len(time_values))
    D = np.empty(len(time_values))
    E = np.empty(len(time_values))
    
    total_pop = 1000
    S[0] = total_pop - 1
    I[0] = 1
    R[0] = 0
    D[0] = 0
    E[0] = 0

  

    for i in range(1,len(time_values)):

        # Flows.
        frac_susceptible =  S[i-1]/(S[0]+E[0]+I[0]+R[0])      # unitless
        SI_contact_rate = frac_susceptible * I[i-1] * contact_factor # contacts/day
        exposure_rate = SI_contact_rate * transmissibility  # infections/day
        recovery_rate = I[i-1] * recovery_factor * (1 - mortality_rate)  # recoveries/day
        dying_rate = I[i-1] * death_factor * mortality_rate #deaths per day
        incubation_rate = E[i-1] * incubation_factor #people per day

        # Primes.
        S_prime = -exposure_rate
        E_prime = exposure_rate - incubation_rate 
        I_prime = incubation_rate - recovery_rate - dying_rate
        R_prime = recovery_rate
        D_prime = dying_rate
        

        # Stocks.
        S[i] = S[i-1] + S_prime * delta_t
        I[i] = I[i-1] + I_prime * delta_t
        R[i] = R[i-1] + R_prime * delta_t
        D[i] = D[i-1] + D_prime * delta_t
        E[i] = E[i-1] + E_prime * delta_t

    if plot:
        plt.plot(time_values,S,color="blue",label="S")
        plt.plot(time_values,I,color="red",label="I")
        plt.plot(time_values,R,color="green",label="R")
        plt.plot(time_values,D,color="black",label="D")
        plt.plot(time_values,E,color="pink",label="E")
        plt.legend()
        plt.show()
    return (I[len(I)-1] + R[len(R)-1])/total_pop*100
    

mids = np.arange(.1,10,.1)
perc_infs = []

for mid in mids:
	perc_inf = runsim(mean_infectious_duration=mid,plot=False)
	perc_infs.append(perc_inf)


plt.plot(mids, perc_infs, color="purple", label="% infected")
plt.xlabel("Mean Infected Duration (days)")
plt.ylabel("% ever infected")
plt.legend()
plt.show()