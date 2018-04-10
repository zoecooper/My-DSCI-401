#Project #1 CPSC 420
#Zoe Cooper 

#Importing...
import numpy as np
import matplotlib.pyplot as plt
import math

delta_t = 5/60
time_values = np.arange(0, 24*91, delta_t) #91 days in April, May, and June

outside_temp_april = 15*np.sin(2*math.pi/24*time_values[0:8736]) + 50 #15 under and over 50, chose 50 bc in middle of the ranges from site I sent
outside_temp_may = 15*np.sin(2*math.pi/24*time_values[8736:17472]) + 60
outside_temp_june = 15*np.sin(2*math.pi/24*time_values[17472:]) + 80

outside_temp = np.concatenate([outside_temp_april, outside_temp_may, outside_temp_june]) #Putting it all together


furnace_rate = 2     # degF/hr
ac_rate = 1.5        # degF/hr
house_leakage_factor = .02   # (degF/hr)/degF


pre_vac_heating_one_day = np.concatenate([             #40 days = first sector before vacay
    np.repeat(68, int(9 / delta_t)),           
    np.repeat(74, int(6 / delta_t)),
    np.repeat(70, int(9 / delta_t))])
pre_vac_heating = np.tile(pre_vac_heating_one_day,40)

pre_vac_cooling_one_day = np.concatenate([                
    np.repeat(82, int(9 / delta_t)),
    np.repeat(79, int(6 / delta_t)),
    np.repeat(79, int(9 / delta_t))])
pre_vac_cooling = np.tile(pre_vac_cooling_one_day,40)

vac_heating_one_day = np.concatenate([             #ACCOUNTS FOR BREAK Acapulco in May heating, 14 days on vacay
    np.repeat(55, int(24 / delta_t))])          
vac_heating = np.tile(vac_heating_one_day,14)

vac_cooling_one_day = np.concatenate([                #ACCOUNTS FOR BREAK cooling
    np.repeat(88, int(24 / delta_t))])
vac_cooling = np.tile(vac_cooling_one_day,14) 


post_vac_heating = np.tile(pre_vac_heating_one_day,37) #37 days = last sector post vacation
post_vac_cooling = np.tile(pre_vac_cooling_one_day,37)


heating_thermostat = np.concatenate([pre_vac_heating, vac_heating, post_vac_heating]) #all together
cooling_thermostat = np.concatenate([pre_vac_cooling, vac_cooling, post_vac_cooling])
       
#"A standard 3 ton AC will cost you a whopping 50 cents an hour to run assuming the previous numbers." -From Sears website.
#"Standard furnace - 12 cents per hour." Electric Heating -From SF gate website.
heating_cost_rate = .12  #cost/hr
ac_cost_rate = .50 #cost/hr


heater_on = np.empty(len(time_values))
heater_on[0] = False
ac_on = np.empty(len(time_values))
ac_on[0] = False

T = np.empty(len(time_values))
T[0] = 55

HYST = 1.0                  # for hysterisis (degF) #buffer #fluctuation

for i in range(1,len(time_values)):
    
    if heater_on[i-1]:
        if T[i-1] - heating_thermostat[i] > HYST:
            heater_on[i] = False
            furnace_heat = 0            # degF/hr
        else:
            heater_on[i] = True
            furnace_heat = furnace_rate   # degF/hr
    else:
        if T[i-1] - heating_thermostat[i] < -HYST:
            heater_on[i] = True
            furnace_heat = furnace_rate   # degF/hr
        else:
            heater_on[i] = False
            furnace_heat = 0        
            
    if ac_on[i-1]:
        if T[i-1] - cooling_thermostat[i] > -HYST:
            ac_on[i] = True
            ac_cool = ac_rate  # degF/hr
        else:
            ac_on[i] = False
            ac_cool = 0   # degF/hr
    else:
        if T[i-1] - cooling_thermostat[i] < HYST:
            ac_on[i] = False
            ac_cool = 0   # degF/hr
        else:
            ac_on[i] = True
            ac_cool = ac_rate       # degF/hr

    leakage_rate = (house_leakage_factor * 
        (T[i-1] - outside_temp[i-1]))      # degF/hr

    T_prime = furnace_heat - leakage_rate

    T[i] = T[i-1] + T_prime * delta_t


#Plotting

plt.plot(time_values,T,color="brown",label="inside temp", 
    linewidth=2,linestyle="-")
plt.plot(time_values, outside_temp,color="purple",
        label="outside temp",linewidth=2,linestyle="-")
plt.plot(time_values,
	heating_thermostat, 
    #np.repeat(heating_thermostat,len(time_values)),
    color="red",
	label="heating thermostat",linewidth=1,linestyle=":")
plt.plot(time_values,
	cooling_thermostat, 
    #np.repeat(cooling_thermostat,len(time_values)),
    color="blue",
	label="cooling thermostat",linewidth=1,linestyle=":")
plt.plot(time_values,5*heater_on,color="orange",label="heater")
plt.plot(time_values,5*ac_on,color="green",label="cooler")
plt.legend()
plt.show()

#Print Statements
print("The heater was on for around " + str(heater_on.mean() * 100) + "% of the day.")
print("The Air Conditioner was on about " + str(ac_on.mean() * 100) + "% of the day.")
#print("The avg heater cost is " + str(heater_on.mean()*100 + heating_cost_rate * 720))
avg_heater_cost = (heater_on.mean()*100 + heating_cost_rate * 720)
print("The avg heater cost is " + str(avg_heater_cost))
avg_ac_cost = (ac_on.mean()*100 + ac_cost_rate * 720)
print("The avg A.C. cost is " + str(avg_ac_cost))
#print("The A.C. cost is " + str(ac_on.mean()*100 + ac_cost_rate * 720))
avg_monthly_utility = (avg_heater_cost + avg_ac_cost)
print("The estimated average monthly utility cost is " + str(avg_monthly_utility))
#print("The estimated average monthly utility cost is " + (str(heater_on.mean()*100 + heating_cost_rate * 720)) + (str(ac_on.mean()*100 + ac_cost_rate * 720)))