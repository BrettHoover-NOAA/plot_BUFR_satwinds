# Example code for reading a Himawari AMV debufr (ascii dump) file, retrieving
# ob data, and plotting a score-card.
#
# PyGSI Compliant
#
# Arrays
import numpy as np
# Plotting
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import colors 
import copy
# Map Transform
import cartopy
import cartopy.feature as cfeat
import cartopy.crs as ccrs
# Read From Debufr
from plot_BUFR_satwinds_dependencies import can_float
from plot_BUFR_satwinds_dependencies import compute_base_bufrdict
from plot_BUFR_satwinds_dependencies import compute_quality_info
from plot_BUFR_satwinds_dependencies import spddir_to_uwdvwd
# Plot Scorecard
from plot_BUFR_satwinds_dependencies import ob_density_plot
from plot_BUFR_satwinds_dependencies import ob_hist_latlonpre
from plot_BUFR_satwinds_dependencies import ob_hist_spddirqi
from plot_BUFR_satwinds_dependencies import stage_scorecard
# Read from Himawari AMV debufr ascii file
subtype_filename='/Users/bhoover/Desktop/IMSG/PROJECTS/himawari9/NC005044.debufr.out'
tags=[ '005002  CLAT' ,   # Latitude
       '006002  CLON' ,   # Longitude
       '001007  SAID' ,   # Satellite ID
       '001033  OGCE' ,   # Originating Centre
       '001032  GNAP' ,   # Generating Application
       '033007  PCCF' ,   # Per Cent Confidence (Quality Indicator)
       '002023  SWCM' ,   # Satellite Wind Computation Method
       '011001  WDIR' ,   # Wind Direction
       '011002  WSPD' ,   # Wind Speed
       '007004  PRLC'     # Pressure
     ]
basedict=compute_base_bufrdict(subtype_filename,tags)
fulldict=compute_quality_info(basedict)
# Process debufr data:
# qifn used for Himawari-9 AMVs
# all WSPD replications are identical
# all WDIR replications are identical
# only first PRLC replication contains non-nan data
qi=fulldict['QIFN'].squeeze()
wdir=fulldict['WDIR'][:,0].squeeze()
wspd=fulldict['WSPD'][:,0].squeeze()
pres=fulldict['PRLC'][:,0].squeeze()
lat=fulldict['CLAT'].squeeze()
lon=fulldict['CLON'].squeeze()
fix=np.where(lon<0.)
lon[fix]=lon[fix]+360.
# Convert (wspd,wdir) to (u,v)
uwd,vwd=spddir_to_uwdvwd(wspd,wdir)
# Plot scorecard
font = { #................................................................................. Font definition library
        'family' : 'DejaVu Sans',
        'weight' : 'bold',
        'size'   : 12
       }
matplotlib.rc('font', **font)
fig=stage_scorecard(lat,lon,pres,wspd,wdir,qi,85.)
fig.savefig('NC005044_scorecard.png',bbox_inches='tight',facecolor='w')
# FINISHED
