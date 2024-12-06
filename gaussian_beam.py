import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import pandas as pd
from scipy.special import erf 

def modified_erf(x, power_min, power_max, beam_radius, beam_center):
    return power_min + power_max/2 * (1-erf(np.sqrt(2) * (x - beam_center)/beam_radius))

def gaussian_beam_fn(z, beam_radius, beam_waist_loc, m):
    return np.sqrt((beam_radius ** 2) * (1 + ((z - beam_waist_loc) ** 2) * (((m ** 2) * (397e-6) / np.pi / (beam_radius ** 2)) ** 2)))

df = pd.DataFrame(pd.read_excel("C:/Users/Hae Lim/Desktop/241030.xlsx"))
print(df)
df.set_index("x\z-position (mm)", inplace=True)
print(df)

z = np.array(df.columns[1:])
print(z)
x = np.array(df.iloc[:0])
print(x)
powers_list = list()

for i in range(len(location_list)):
    powers_list.append(df.iloc[:, i + 1].values)

beam_radius_list = list()

for location in range(len(location_list)):
    # Initial guesses for parameters
    init_guess = [min(powers_list[location]), max(powers_list[location]), 1, np.mean(positions)]

    # Fit the data
    popt, pcov = opt.curve_fit(modified_erf, positions, powers_list[location], p0=init_guess)

    # Extract the fitted parameters
    power_min, power_max, beam_radius, beam_center = popt

    # Compare the data and the fit
    plt.plot(positions, powers_list[location], 'o', label='Data')
    plt.plot(positions, modified_erf(positions, power_min, power_max , beam_radius, beam_center), '-', label='Fit')
    plt.title("Knife Position vs. Power")
    plt.xlabel("Knife Position (mm)")
    plt.ylabel("Power (mW)")
    plt.legend()
    plt.show()

    displacement = np.arange(np.mean(positions) - 5 * beam_radius, np.mean(positions) + 5 * beam_radius, 0.01) - beam_center
    intensity = 2 * max(powers_list[location]) / np.pi / (beam_radius ** 2) * np.exp(-2 * ((displacement) / beam_radius) ** 2)
    plt.plot(displacement, intensity)
    plt.title("Displacement vs. Intensity")
    plt.xlabel("Displacement (mm)")
    plt.ylabel("Intensity (mW/mm^2)")
    plt.show()

    beam_radius_list.append(beam_radius)

print("Beam Radius List:", beam_radius_list)




# Compare the data and the fit
init_guess = [np.mean(beam_radius_list), np.mean(location_list), 1]
popt, pcov = opt.curve_fit(gaussian_beam_fn, location_list, beam_radius_list, p0=init_guess)

# Extract the fitted parameters
beam_radius, beam_waist_loc, m = popt
print("Beam Radius:", beam_radius, "mm")
print("Beam Waist Location:", beam_waist_loc, "mm")
print("M^2:", m ** 2)
print("Rayleigh Range:", np.pi * (beam_radius ** 2) / (m ** 2) / (397e-6), "mm")
print("Divergence Angle:", (m ** 2) * (397e-6) / np.pi / beam_radius * 360 / (2 * np.pi), "degrees")

# 
plt.plot(location_list, beam_radius_list, 'o', label='Data')
z = np.arange(np.min(location_list), np.max(location_list), 1)
plt.plot(z, gaussian_beam_fn(z, beam_radius, beam_waist_loc, m), '-', label='Fit')
plt.title("z-coordinate vs. Beam Radius")
plt.xlabel("z-coordinate (mm)")
plt.ylabel("Beam Radius (mm)")
plt.show()
