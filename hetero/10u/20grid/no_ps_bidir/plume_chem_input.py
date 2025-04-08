import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
import netCDF4 as nc

import microhh_tools as mht
import helpers as hlp
from constants import *
from lsm_input import LSM_input

"""
Settings.
"""
float_type = "f8"

xsize = 7680
ysize = 5760
zsize = 3840

itot = 384
jtot = 288
ktot = 192

start_hour = 0
end_hour = 24

base_date = "20-09-20"
hour = int(start_hour)
minute = 0
second = 0
datetime_str = f"{base_date} {hour:02d}:{minute:02d}:{second:02d}"

lat=51.97
lon=4.926

# Define grid spacing explicitly
dx = xsize/itot  # Grid spacing in x
dy = ysize/jtot  # Grid spacing in y
print(f"\nGrid spacing:")
print(f"dx = {dx}m")
print(f"dy = {dy}m")

# Enable resolved plume rise:
sw_plume_rise = False

# Enable non-linear KPP chemistry:
sw_chemistry = True

# Enable land-surface model and more detailled deposition.
sw_land_surface = True


"""
Read base .ini file for case settings.
"""
ini = mht.Read_namelist("plume_chem.ini.base")


"""
Create case input.
"""
if sw_chemistry:

    # Read TUV output table.
    # There is a total of 24 hour available, generated for the
    # Jaenschwalde power plant on 23/05/2022.
    columns = [
        "time",
        "sza",
        "jo31d",
        "jh2o2",
        "jno2",
        "jno3",
        "jn2o5",
        "jch2or",
        "jch2om",
        "jch3o2h",
    ]
    tuv = pd.read_table(
        "plume_chem_tuv_output.txt",
        sep="\\s+",
        skiprows=12,
        skipfooter=1,
        engine="python",
        names=columns,
        index_col="time",
    )

    # NOTE: `.loc` is value based, not on index.
    tuv = tuv.loc[start_hour:end_hour]

    # Convert to seconds, and subtract starting time.
    tuv.index *= 3600
    tuv.index -= tuv.index.values[0]

    # Emissions (?)
    emi_no = np.zeros(tuv.index.size)
    emi_isop = np.zeros(tuv.index.size)

    xmnh3 = 17.031;
    xmair = 28.9647;
    xmair_i = 1.0 / xmair;
    rho = 1.2658
    c_ug = (1.0e9) * rho * xmnh3 * xmair_i;

    # Concentrations, for now constant with height.
    #species = {"nh3": (5.0/c_ug)}
    species = {"nh3": 6.735e-9}

    deposition_species = ["nh3"]
else:
    species = {}



#def stretched_vertical_grid(z_bottom, z_top, n_layers, stretch_factor):
#    # Create a normalized grid in [0, 1]
#    eta = np.linspace(0, 1, n_layers + 1)  # n_layers + 1 points (including top and bottom)
#    stretched_eta = (np.exp(stretch_factor * eta) - 1) / (np.exp(stretch_factor) - 1)
#    zh = z_bottom + (z_top - z_bottom) * stretched_eta
#    z = 0.5 * (zh[:-1] + zh[1:])
#    return z, zh
#
#z_bottom = 0
#z_top = zsize
#n_layers = ktot
#stretch_factor = 1.6  # This gives ~2.5m near surface, ~20m aloft for 384 layers
#
#z, zh = stretched_vertical_grid(z_bottom, z_top, n_layers, stretch_factor)
#
## Calculate dz for informational purposes
#dz = np.diff(zh)

dz = zsize / ktot
z = np.arange(0.5 * dz, zsize, dz)


def profile(zi, v_bulk, dv, gamma_v, clip_at_zero=False):
    """
    Create well mixed profile with jump and constant lapse rate above.
    """

    k_zi = np.abs(z - zi).argmin()

    profile = np.zeros(ktot)
    profile[:k_zi] = v_bulk
    profile[k_zi:] = v_bulk + dv + gamma_v * (z[k_zi:] - zi)

    if clip_at_zero:
        profile[profile < 0] = 0.0

    return profile


# Vertical profiles.
thl = profile(zi=600, v_bulk=291,   dv=2,     gamma_v=0.006)
qt  = profile(zi=600, v_bulk=7.5e-3, dv=-0.3e-3, gamma_v=-0.002e-3, clip_at_zero=True)

u = np.ones(ktot) * 10
nudgefac = np.ones(ktot) / 10800

# Surface fluxes.
#t0 = start_hour * 3600
#t1 = end_hour * 3600
#td = 12 * 3600
#time = np.linspace(t0, t1, 32)
time = np.linspace(start_hour*3600, end_hour*3600, 97)
hour_of_day = (time / 3600) % 24
daytime_mask = (hour_of_day >= 6) & (hour_of_day <= 18)


# wthl = 0.15 * np.sin(np.pi * (time-t0) / td)
# wqt  = 8e-5 * np.sin(np.pi * (time-t0) / td)
#wthl = 0.2 * np.sin(np.pi * (time - t0) / td)
#wqt = 8e-5 * np.sin(np.pi * (time - t0) / td)

# Heat fluxes should also use hour_of_day for consistency:
wthl = 0.2 * np.sin(np.pi * hour_of_day / 24)
wqt = 8e-5 * np.sin(np.pi * hour_of_day / 24)

##########################################################
# Surface radiation (only used with land-surface enabled).
##########################################################
max_rad = 650  
#sw_flux_dn = max_rad * np.sin(np.pi * (time-t0) / td)

# Initialize radiation arrays
sw_flux_dn = np.zeros_like(time)
sw_flux_dn[daytime_mask] = max_rad * np.sin(np.pi * (hour_of_day[daytime_mask] - 6) / 12)

sw_flux_dn[sw_flux_dn < 0] = 0
sw_flux_up = 0.2 * sw_flux_dn

lw_flux_dn = np.ones_like(sw_flux_dn) * 334
lw_flux_up = np.ones_like(sw_flux_dn) * 452
##########################################################

##pl.figure(figsize=(10,5))
##
##pl.subplot(131)
##pl.plot(time/3600, wthl)
##pl.xlabel("time (h)")
##pl.ylabel("w`thl` (K m s-1)")
##
##pl.subplot(132)
##pl.plot(time/3600, wqt*1000)
##pl.xlabel("time (h)")
##pl.ylabel("w`qt` (g kg-1 m s-1)")
##
##pl.subplot(133)
##pl.plot(time/3600, sw_flux_dn, label="sw_flux_dn")
##pl.plot(time/3600, sw_flux_up, label="sw_flux_up")
##pl.plot(time/3600, lw_flux_dn, label="lw_flux_dn")
##pl.plot(time/3600, lw_flux_up, label="lw_flux_up")
##pl.xlabel("time (h)")
##pl.ylabel("sw_flux_dn` (W m-2)")
##pl.legend()
##
##pl.tight_layout()

################################
#Write input NetCDF file.
################################

# Defines helper function add_nc_var to add variables to NetCDF file:
def add_nc_var(name, dims, nc, data):
    if dims is None:
        var = nc.createVariable(name, np.float64)
    else:
        var = nc.createVariable(name, np.float64, dims)
    var[:] = data

# Creates "plume_chem_input.nc" file
nc_file = nc.Dataset("plume_chem_input.nc", mode="w", datamodel="NETCDF4", clobber=True)

# Add start_hour as a variable
add_nc_var("start_hour", None, nc_file, start_hour)  # None means it's a scalar variable without dimensions
add_nc_var("max_rad", None, nc_file, max_rad)  # None means scalar variable

###############################
# Sets up dimensions and groups
###############################

# "z" dimension for vertical levels:
nc_file.createDimension("z", ktot)
add_nc_var("z", ("z"), nc_file, z)

# Atmospheric input.
# ("init" group for initial atmospheric conditions)
nc_group_init = nc_file.createGroup("init")

add_nc_var("u", ("z"), nc_group_init, u)
add_nc_var("thl", ("z"), nc_group_init, thl)
add_nc_var("qt", ("z"), nc_group_init, qt)
add_nc_var("nudgefac", ("z"), nc_group_init, nudgefac)
add_nc_var("qt_nudge", ("z"), nc_group_init, qt)
#add_nc_var("co2", ("z"), nc_group_init, co2)
#add_nc_var("co2_inflow", ("z"), nc_group_init, co2)

#("timedep" group for time-dependent surface conditions)
nc_tdep = nc_file.createGroup("timedep");
nc_tdep.createDimension("time_surface", time.size)

add_nc_var("time_surface", ("time_surface"), nc_tdep, time-time[0])
add_nc_var("thl_sbot", ("time_surface"), nc_tdep, wthl)
add_nc_var("qt_sbot", ("time_surface"), nc_tdep, wqt)

if (sw_chemistry):
    # Chemistry input.
    # "timedep_chem" group (if chemistry enabled)
    nc_chem = nc_file.createGroup("timedep_chem");
    nc_chem.createDimension("time_chem", tuv.index.size)

    add_nc_var("time_chem", ("time_chem"), nc_chem, tuv.index)
    add_nc_var("jo31d", ("time_chem"), nc_chem, tuv.jo31d)
    add_nc_var("jh2o2", ("time_chem"), nc_chem, tuv.jh2o2)
    add_nc_var("jno2", ("time_chem"), nc_chem, tuv.jno2)
    add_nc_var("jno3", ("time_chem"), nc_chem, tuv.jno3)
    add_nc_var("jn2o5", ("time_chem"), nc_chem, tuv.jn2o5)
    add_nc_var("jch2or", ("time_chem"), nc_chem, tuv.jch2or)
    add_nc_var("jch2om", ("time_chem"), nc_chem, tuv.jch2om)
    add_nc_var("jch3o2h", ("time_chem"), nc_chem, tuv.jch3o2h)
    add_nc_var("emi_isop", ("time_chem"), nc_chem, emi_isop)
    add_nc_var("emi_no", ("time_chem"), nc_chem, emi_no)

    for name, value in species.items():
        profile = np.ones(ktot, dtype=np.float64)*value
        add_nc_var(name, ("z"), nc_group_init, profile)
        add_nc_var("{}_inflow".format(name), ("z"), nc_group_init, profile)

    # Add flux_nh3 variable
    add_nc_var("flux_nh3", ("z"), nc_group_init, np.zeros(ktot, dtype=np.float64))
    add_nc_var("flux_inst", ("z"), nc_group_init, np.zeros(ktot, dtype=np.float64))

if (sw_land_surface):
    # "soil" group (if land surface enabled)
    nc_soil = nc_file.createGroup("soil")
    nc_soil.createDimension("z", 4)
    add_nc_var("z", ("z"), nc_soil, np.array([-1.945, -0.64, -0.175, -0.035]))

    add_nc_var("theta_soil", ("z"), nc_soil, np.array([0.34, 0.25, 0.21, 0.18]))
    add_nc_var("t_soil", ("z"), nc_soil, np.array([282, 287, 290, 286]))
    add_nc_var("index_soil", ("z"), nc_soil, np.ones(4) * 2)
    #add_nc_var("root_frac", ("z"), nc_soil, np.array([0.05, 0.3, 0.4, 0.25]))

    # Add idealized (prescribed) radiation.
    add_nc_var("sw_flux_dn", ("time_surface"), nc_tdep, sw_flux_dn)
    add_nc_var("sw_flux_up", ("time_surface"), nc_tdep, sw_flux_up)
    add_nc_var("lw_flux_dn", ("time_surface"), nc_tdep, lw_flux_dn)
    add_nc_var("lw_flux_up", ("time_surface"), nc_tdep, lw_flux_up)

nc_file.close()

"""
Define emissions.
"""
# Coordinates of central cooling tower (m):
# Coordinates of central cooling tower (m):
x0 = 400
y0 = ysize/2.
z0 = 0

# # Std-dev of plume widths:
# sigma_x = 25
# sigma_y = 25
sigma_x = (xsize/itot)*0.5
sigma_y = (ysize/jtot)*0.5
sigma_z = (xsize/itot)*0.5

# # Handles plume rise conditions:
# # If enabled: uses tower height (120m)
# # If disabled: uses fitted heights from CSV profiles
# 
# if sw_plume_rise:
#     # Emissions from tower height.
#     z0 = 120
#     sigma_z = 25
# else:
#     # The heights and sigma are from Dominik"s .csv profiles, curve fitted with Python.
#     z0 = 299.68    # of 599.69 for high
#     sigma_z = 122.37

# # x,y spacing towers:
# dx = 290
# dy = 120
# ddx = 40

# # Strength of plumes, from the CoCO2 simulation protocol:
# strength_co2 = 0.0  / 9. / MCO2   # kmol(CO2) s-1
# strength_no2 = 0.0 / 9. / MNO2   # kmol(NO2) s-1
# strength_no  = 0.0  / 9. / MNO    # kmol(NO) s-1
# strength_co  = 0.0  / 9. / MCO    # kmol(CO) s-1
# strength_nh3  = 1.0  / 9. / MNH3    # kmol(NH3) s-1
# strength_nh3  = 1.48950934102587E-06  # kmol(NH3) s-1 (equivalent to one barn of 80 cows)
# strength_nh3  = 1.0e-06  # kmol(NH3) s-1 
# strength_nh3  = (2000.0 / (365 * 24 * 3600)) / 17.031
# strength_nh3  = (0.01e-3) / 17.031  # 0.01 g/s (Livestock farm)
strength_nh3  = 0.0

# Emission of heat and moisture. Numbers are from:
# Effective pollutant emission heights for atmospheric transport modelling based on real-world information
# Pregger and Friedrich, 2009, 10.1016/j.envpol.2008.09.027
## Tp = 50+T0    # Flue gass temperature (K)
## Mp = 790      # Volume-flux (m3 s-1)
## 
## # This is not very accurate...:
## pp = 1e5
## rhop = pp/(Rd*Tp)
## rhp = 1.
## qp = rhp * hlp.calc_qsat(Tp, pp)
## 
## strength_q = np.round(Mp * rhop * qp, decimals=2)
## strength_T = np.round(Mp * rhop * Tp, decimals=2)
## 
# Emission input model.
emi = hlp.Emissions()

##x = x0
##y = y0
##z = z0


y_offsets = [-28, -9, 9, 18]  # Specified y-offsets

for y_offset in y_offsets:
    x = x0
    y = y0 + y_offset
    z = z0

    # Only adding NH3 as the source as requested
    if (sw_chemistry):
        emi.add('nh3', strength_nh3, True, x, y, z, sigma_x, sigma_y, sigma_z)
    
##        # emi.add("co2", strength_co2, True, x, y, z, sigma_x, sigma_y, sigma_z)

##if sw_chemistry:
##    emi.add("nh3", strength_nh3, True, x, y, z, sigma_x, sigma_y, sigma_z)

## if sw_plume_rise:
##     emi.add("thl", strength_T, False, x, y, z, sigma_x, sigma_y, sigma_z)
##     emi.add("qt", strength_q, False, x, y, z, sigma_x, sigma_y, sigma_z)


##############################################################################################
##Create heterogeneous land-surface with three distinct regions (grass_upwind, forest, grass_downwind).
##############################################################################################


if (sw_land_surface):
    # Define block sizes for x and y directions
    blocksize_i = 1  # Size of block in x direction (in grid points)
    blocksize_j = 1  # Size of block in y direction (in grid points)
    
    # Calculate size of each block in total grid points
    block_points = blocksize_i * blocksize_j  # Total grid points in one block
    
    # Calculate total number of possible regions in each direction
    region_sizex = itot // blocksize_i  # Number of possible regions in x direction
    region_sizey = jtot // blocksize_j  # Number of possible regions in y direction
    
    # Print information about block structure
    print(f"\nBlock structure:")
    print(f"Each block is {blocksize_i}×{blocksize_j} grid points")
    print(f"Maximum possible regions: {region_sizex} (x) × {region_sizey} (y)")

    # EDITED: Define boundaries for the three regions (in grid points)
    
    # Adjusting boundaries to align with block size
    grass_upwind_end = int((1400/dx) // blocksize_i) * blocksize_i
    forest_end = int((1400/dx + 5000/dx) // blocksize_i) * blocksize_i
    # grass_downwind continues to end (32 * 100 = 3200m)

    # Print information about domain setup
    print(f"\nDomain setup (aligned to block boundaries):")
    print(f"Grass_in region: 0 to {grass_upwind_end*dx}m (0 to {grass_upwind_end} points)")
    print(f"Forest region: {grass_upwind_end*dx}m to {forest_end*dx}m ({grass_upwind_end} to {forest_end} points)")
    print(f"Grass_out region: {forest_end*dx}m to {itot*dx}m ({forest_end} to {itot} points)")
    
    ##########################################################
    # Create and save mask files according to MicroHH format
    ##########################################################
    
    # Initialize masks for three regions
    mask_grass_upwind = np.zeros((jtot, itot), dtype=float_type)
    mask_forest = np.zeros((jtot, itot), dtype=float_type)
    mask_grass_downwind = np.zeros((jtot, itot), dtype=float_type)
    
    # Set the regions according to specified boundaries using block-wise assignment
    for j in range(0, jtot, blocksize_j):
        j_slice = slice(j, j + blocksize_j)
        
        # Assign grass_upwind region blocks
        for i in range(0, grass_upwind_end, blocksize_i):
            i_slice = slice(i, i + blocksize_i)
            mask_grass_upwind[j_slice, i_slice] = 1
        
        # Assign forest region blocks
        for i in range(grass_upwind_end, forest_end, blocksize_i):
            i_slice = slice(i, i + blocksize_i)
            mask_forest[j_slice, i_slice] = 1
        
        # Assign grass_downwind region blocks
        for i in range(forest_end, itot, blocksize_i):
            i_slice = slice(i, i + blocksize_i)
            mask_grass_downwind[j_slice, i_slice] = 1
    
    # Save binary mask files according to MicroHH format
    mask_grass_upwind.tofile("grass_upwind.0000000")
    mask_forest.tofile("forest.0000000")
    mask_grass_downwind.tofile("grass_downwind.0000000")
    
    ##########################################################
    # Initialize and setup Land Surface Model
    ##########################################################
    
    # Initialize LSM
    # ls = LSM_input(itot, jtot, 4, sw_water=True, TF=float_type, debug=True, exclude_fields=["z0m", "z0h"])
    ls = LSM_input(itot, jtot, 4, sw_water=True, TF=float_type, debug=True)
    
    # Create boolean masks for LSM setup (aligned with block boundaries)
    mask_forest_bool = np.zeros((jtot, itot), dtype=bool)
    mask_forest_bool[:, grass_upwind_end:forest_end] = True

    # Create and set roughness length fields
    z0m = np.zeros((jtot, itot), dtype=float_type)
    z0h = np.zeros((jtot, itot), dtype=float_type)
    
    # Set roughness lengths for forest
    z0m[mask_forest_bool] = 0.75    # Momentum roughness length for forest
    z0h[mask_forest_bool] = 0.75   # Heat roughness length for forest
    
    # Set roughness lengths for grass (both in and out regions)
    z0m[~mask_forest_bool] = 0.03   # Momentum roughness length for grass
    z0h[~mask_forest_bool] = 0.003 # Heat roughness length for grass

    # Set z0m and z0h in LSM
    ls["z0m"][:,:] = z0m
    ls["z0h"][:,:] = z0h
    
    # Root distribution by layer
    root_frac = np.zeros((4, jtot, itot), dtype=float_type)

    # Set root distribution values
    forest_values = [0.24, 0.38, 0.31, 0.07]  # Layers 1-4 for forest
    grass_values = [0.35, 0.38, 0.23, 0.04]   # Layers 1-4 for grass

    # Apply values based on mask
    for layer in range(4):
        for j in range(jtot):
            for i in range(itot):
                if mask_forest_bool[j, i]:  # If forest
                    root_frac[layer, j, i] = forest_values[layer]
                else:  # If grassland
                    root_frac[layer, j, i] = grass_values[layer]

    # Set the root_frac in the land surface model
    ls["root_frac"][:,:,:] = root_frac

    # Save binary files
    z0m.tofile("z0m.0000000")
    z0h.tofile("z0h.0000000")
    
    def set_value(variable, forest, grass):
        ls[variable][mask_forest_bool] = forest      # Forest values
        ls[variable][~mask_forest_bool] = grass      # Grass values (both in and out)

    # Set surface properties for forest and grass regions
    set_value("c_veg", forest=1.0, grass=1.0)      # Vegetation fraction
    set_value("lai", forest=4.0, grass=3.1)        # Leaf Area Index
    set_value("water_mask", forest=0, grass=0)     # No water surfaces

    ##########################################################
    # Set constant properties for all surfaces
    ##########################################################

     # Surface parameters
    #ls["gD"][:,:] = 0.00              # Vegetation water stress parameter(1/hPa)
    set_value("gD", forest=0.03, grass=0.0)
    #ls["rs_veg_min"][:,:] =100        # Minimum canopy surface resistance
    set_value("rs_veg_min", forest=250, grass=100)
    ls["rs_soil_min"][:,:] = 50      # Minimum soil surface resistance
    ls["lambda_stable"][:,:] = 10     # Skin conductivity for stable conditions (W/m²/K)
    ls["lambda_unstable"][:,:] = 10   # Skin conductivity for unstable conditions (W/m²/K)
    ls["cs_veg"][:,:] = 0.0           # Vegetation heat capacity
    ls["t_bot_water"][:,:] = 295     # Bottom water temperature (K)

    # Soil properties (4 layers)
    ls["t_soil"][:,:,:] = 290        # Soil temperature
    ls["theta_soil"][:,:,:] = 0.3    # Volumetric soil moisture content (m³/m³)
    ls["index_soil"][:,:,:] = 2      # Soil type index
    ls["root_frac"][:,:,:] = 0.25    # Root fraction distribution in each soil layer

    # Check if all values are set
    ls.check()
    
    # Save LSM setup
    ls.save_binaries(allow_overwrite=True)
    ls.save_netcdf("lsm_input.nc", allow_overwrite=True)

    ###########################
    #Add settings to .ini file.
    ###########################
    
    # Sets grid parameters:
    ini["grid"]["itot"] = itot
    ini["grid"]["jtot"] = jtot
    ini["grid"]["ktot"] = ktot
    
    ini["grid"]["xsize"] = xsize
    ini["grid"]["ysize"] = ysize
    ini["grid"]["zsize"] = zsize
    
    
    # Add statistics settings for masks
    ini["stats"]["xymasklist"] = "grass_upwind,forest,grass_downwind"
    

##    ###############################################
##
##if (sw_land_surface):
##    # Choose which surface type to use for the entire domain
##    # Set this variable to 'forest' or 'grass' to select which type covers the entire domain
##    surface_type = 'grass'  # Change to 'forest' to use forest properties instead
##    
##    # Set block size to cover the entire domain
##    blocksize_i = xsize # Size of block in x direction (entire domain)
##    blocksize_j = ysize  # Size of block in y direction (entire domain)
##    
##    # Calculate size of each block in total grid points
##    block_points = blocksize_i * blocksize_j  # Total grid points in one block
##    
##    # Print information about block structure
##    print(f"\nBlock structure:")
##    print(f"Using single block of size {blocksize_i}×{blocksize_j} grid points")
##    print(f"Selected surface type for entire domain: {surface_type}")
##    
##    ##########################################################
##    # Create and save mask files according to MicroHH format
##    ##########################################################
##    
##    # Initialize masks for both types (but only one will be used)
##    mask_forest = np.zeros((jtot, itot), dtype=float_type)
##    mask_grass = np.zeros((jtot, itot), dtype=float_type)
##    
##    # Set the selected mask to cover the entire domain
##    if surface_type == 'forest':
##        mask_forest[:,:] = 1.0
##        print(f"\nDomain setup: Entire domain ({itot*dx}m x {jtot*dy}m) set as FOREST")
##    else:  # grass
##        mask_grass[:,:] = 1.0
##        print(f"\nDomain setup: Entire domain ({itot*dx}m x {jtot*dy}m) set as GRASS")
##    
##    # Save binary mask files according to MicroHH format
##    mask_forest.tofile("forest.0000000")
##    mask_grass.tofile("grass.0000000")
##    
##    ##########################################################
##    # Initialize and setup Land Surface Model
##    ##########################################################
##    
##    # Initialize LSM
##    ls = LSM_input(itot, jtot, 4, sw_water=True, TF=float_type, debug=True, exclude_fields=["z0m", "z0h"])
##    
##    # Create boolean mask for LSM setup (entire domain based on selection)
##    mask_forest_bool = np.ones((jtot, itot), dtype=bool) if surface_type == 'forest' else np.zeros((jtot, itot), dtype=bool)
##    mask_grass_bool = ~mask_forest_bool  # Opposite of forest mask
##    
##    # Create roughness length fields
##    z0m = np.zeros((jtot, itot), dtype=float_type)
##    z0h = np.zeros((jtot, itot), dtype=float_type)
##    
##    # Set roughness lengths based on surface type
##    # Forest values
##    z0m[mask_forest_bool] = 0.75    # Momentum roughness length for forest
##    z0h[mask_forest_bool] = 0.75    # Heat roughness length for forest
##    
##    # Grass values 
##    z0m[mask_grass_bool] = 0.03     # Momentum roughness length for grass
##    z0h[mask_grass_bool] = 0.003    # Heat roughness length for grass
##    
##    # Set z0m and z0h in LSM
##    ls["z0m"][:,:] = z0m
##    ls["z0h"][:,:] = z0h
##    
##    # Save binary files
##    z0m.tofile("z0m.0000000")
##    z0h.tofile("z0h.0000000")
##    
##    # Function to set values based on surface type
##    def set_value(variable, forest, grass):
##        ls[variable][mask_forest_bool] = forest  # Forest values
##        ls[variable][mask_grass_bool] = grass    # Grass values
##    
##    # Set surface properties for forest and grass regions
##    set_value("c_veg", forest=1.0, grass=1.0)      # Vegetation fraction
##    set_value("lai", forest=4.0, grass=3.1)        # Leaf Area Index
##    set_value("water_mask", forest=0, grass=0)     # No water surfaces
##    
##    # Set different resistance values for each surface type
##    set_value("rs_veg_min", forest=250, grass=100)  # Minimum vegetation surface resistance
##
##    # Surface parameters (same for both surface types)
##    ls["gD"][:,:] = 0.00              # Vegetation water stress parameter(1/hPa)
##    ls["rs_soil_min"][:,:] = 50       # Minimum soil surface resistance
##    ls["lambda_stable"][:,:] = 10     # Skin conductivity for stable conditions (W/m²/K) 
##    ls["lambda_unstable"][:,:] = 10   # Skin conductivity for unstable conditions (W/m²/K) 
##    ls["cs_veg"][:,:] = 0.0           # Vegetation heat capacity
##    ls["t_bot_water"][:,:] = 295      # Bottom water temperature (K)
##    
##    # Soil properties (4 layers)
##    ls["t_soil"][:,:,:] = 293         # Soil temperature
##    ls["theta_soil"][:,:,:] = 0.3     # Volumetric soil moisture content (m³/m³)
##    ls["index_soil"][:,:,:] = 2       # Soil type index
##    ls["root_frac"][:,:,:] = 0.25     # Root fraction distribution in each soil layer
##
##    # Check if all values are set
##    ls.check()
##    
##    # Save LSM setup
##    ls.save_binaries(allow_overwrite=True)
##    ls.save_netcdf("lsm_input.nc", allow_overwrite=True)
##
##    ###########################
##    #Add settings to .ini file.
##    ###########################
##    
##    # Sets grid parameters:
##    ini["grid"]["itot"] = itot
##    ini["grid"]["jtot"] = jtot
##    ini["grid"]["ktot"] = ktot
##    
##    ini["grid"]["xsize"] = xsize
##    ini["grid"]["ysize"] = ysize
##    ini["grid"]["zsize"] = zsize
##    
##    # Add statistics settings for masks - include only the active one
##    if surface_type == 'forest':
##        ini["stats"]["xymasklist"] = "forest"
##    else:
##        ini["stats"]["xymasklist"] = "grass"
##
##
##    ###############################################


    # Handles scalar variables:
    scalars = list(species.keys())
    ini["advec"]["fluxlimit_list"] = scalars
    ini["limiter"]["limitlist"] = scalars
    ini["fields"]["slist"] = scalars
    ini["boundary"]["scalar_outflow"] = scalars
    
    ini['grid']['lat'] = lat
    ini['grid']['lon'] = lon
    ini['deposition']['start_hour'] = start_hour
    ini['radiation']['max_rad'] =  max_rad
    ini["time"]["datetime_utc"] = datetime_str
    ini["time"]["endtime"] = (end_hour - start_hour) * 3600
    
    # Configures sources and chemistry:
    ini["source"]["sourcelist"] = emi.source_list
    
    ini["chemistry"]["swchemistry"] = sw_chemistry
    
    # crosslist = ["thl", "qt", "u", "v", "w", "thl_fluxbot", "qt_fluxbot","flux_nh3","flux_inst"]
    crosslist = ["u", "thl_fluxbot", "qt_fluxbot", "flux_nh3", "flux_inst"]
    
    if (sw_chemistry):
        # Add chemicial species and their vertical integrals.
        crosslist += list(species.keys())
        crosslist += [f"{x}_path" for x in species.keys()]
        crosslist += [f"vd{x}" for x in deposition_species]
    
    if (sw_land_surface and sw_chemistry):
        # Add deposition for each land-surface tile.
        for s in deposition_species:
            for t in ["soil", "wet", "veg"]:
                crosslist.append(f"vd{s}_{t}")

    crosslist.append("ra")  # Grid-mean aerodynamic resistance
    crosslist.append("rb")  
    crosslist.append("obuk")
    crosslist.append("ustar")
    crosslist.append("ccomp_tot")  # Grid-mean compensation point

    # Add new resistance components
    crosslist.append("cw")        # Grid-mean external leaf resistance
    crosslist.append("cstom")     # Grid-mean stomatal resistance
    crosslist.append("csoil_eff") # Grid-mean soil effective resistance

    # Add new compensation points
    crosslist.append("cw_out")    # Grid-mean external leaf compensation point
    crosslist.append("cstom_out") # Grid-mean stomatal compensation point
    crosslist.append("csoil_out") # Grid-mean soil compensation point
    crosslist.append("rc_tot")
    crosslist.append("rc_eff")

    ### Add surface parameters for each land-surface tile
    for t in ["soil", "wet", "veg"]:
        crosslist.append(f"ra_{t}")
        crosslist.append(f"rb_{t}")
    ##    crosslist.append(f"obuk_{t}")
    ##    crosslist.append(f"ustar_{t}")
    ##    crosslist.append(f"cw_{t}")
    ##    crosslist.append(f"cstom_{t}")
    ##    crosslist.append(f"csoil_eff_{t}")
    ##    crosslist.append(f"cw_out_{t}")
    ##    crosslist.append(f"cstom_out_{t}")
    ##    crosslist.append(f"csoil_out_{t}")
        crosslist.append(f"rc_tot_{t}")
        crosslist.append(f"rc_eff_{t}")


    ## Configures boundary conditions based on land-surface flag:
    # With land-surface:
    if (sw_land_surface):
        ini["boundary"]["swboundary"] = "surface_lsm"
        ini["boundary"]["sbcbot"] = "flux"
        ini["boundary"]["sbot"] = "0"
        ini["boundary"]["thl"] = "dirichlet"
        ini["boundary"]["qt"] = "dirichlet"
        ini["boundary"]["swtimedep"] = False
        ini["boundary"]["timedeplist"] = "empty"
    
        ini["radiation"]["swradiation"] = "prescribed"
    
    # Without land-surface (Radiation source is off):
    else:
        ini["boundary"]["swboundary"] = "surface"
        ini["boundary"]["sbcbot"] = "flux"
        ini["boundary"]["swtimedep"] = True
        ini["boundary"]["timedeplist"] = ["thl_sbot", "qt_sbot"]
    
        ini["radiation"]["swradiation"] = False
    
    
    ## Sets deposition settings:
    #  if BOTH chemistry AND land surface are enabled, turns ON deposition in the model!
    if (sw_chemistry and sw_land_surface):
        ini["deposition"]["swdeposition"] = True
    
    # If either one or both are disabled, turns OFF deposition in the model!
    else:
        ini["deposition"]["swdeposition"] = False
    
    # Configures cross-section output and source parameters:
    ini["cross"]["crosslist"] = crosslist
    ini["cross"]["xz"] = ysize/2
    
    # Adds emission source locations and parameters:
    ini["source"]["source_x0"] = emi.x0
    ini["source"]["source_y0"] = emi.y0
    ini["source"]["source_z0"] = emi.z0
    
    ini["source"]["sigma_x"] = emi.sigma_x
    ini["source"]["sigma_y"] = emi.sigma_y
    ini["source"]["sigma_z"] = emi.sigma_z
    
    ini["source"]["strength"] = emi.strength
    ini["source"]["swvmr"] = emi.sw_vmr
    
    ini["source"]["line_x"] = emi.line_x
    ini["source"]["line_y"] = emi.line_y
    ini["source"]["line_z"] = emi.line_z
    
    #  saves the configuration:
    ini.save("plume_chem.ini", allow_overwrite=True)

