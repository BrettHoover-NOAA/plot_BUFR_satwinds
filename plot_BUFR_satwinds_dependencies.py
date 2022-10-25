# Set of functions for *.debufr.out text parsing for extracting satwinds data
#
# PyGSI compliant
#
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import colors 
import copy
import cartopy
import cartopy.feature as cfeat
import cartopy.crs as ccrs
#
# General Purpose Functions
#   can_float                Returns True if input can be converted to a float, otherwise returns False
#   spddir_to_uwdvwd         Converts (spd,dir) to (u,v)
#   truncate_colorbar        Returns colorbar only covering subspace within [0. 1.] of input colorbar
#   key_substr_search        Returns dictionary key that contains substring, or None if no key contains substring
#
# Debufr Parsing Functions
#
#   compute_base_bufrdict    Searches all instances of each tag and constructs vectors of values. Converts to float if possible,
#                            and reshapes to (num obs, num replications) as necessary. Returns dictionary of keys for each tag.
#
#   compute_quality_info     Extracts QIFY, QIFN, and EE values from GNAP, OGCE, and PCCF, and constructs arrays. Returns new
#                            dictionary with QIFY, QIFN, and EE keys. Does not currently work for all message types and satIDs.
# Wind Scorecard Functions
#
#   ob_density_plot          Fills input axis with a density plot of ob-counts in lower/upper troposphere (layered semi-transparent
#                            pcolormesh) on input projection (e.g. ccrs.PlateCaree())
#
#   ob_hist_latlonpre        Fills input axes with 3 histograms of ob-count by: pressure, latitude, and longitude
#
#   ob_hist_spddirqi         Fills input axes with histogram of ob-count by speed, wind-rose histogram of ob-count by direction, and
#                            a pie-chart of ob-counts above/below a given QI threshold
#
#   stage_scorecard          Sets a single figure with appropriate figure/subfigure/axes settings and runs scorecard functions to
#                            generate a scorecard for input wind data
#
def can_float(element: any) -> bool:
    # Determins if an input element can be converted to a float, returns if True or False
    try:
        float(element)
        return True
    except ValueError:
        return False

def spddir_to_uwdvwd(spd,ang):
    uwd=-spd*np.sin(ang*(np.pi/180.))
    vwd=-spd*np.cos(ang*(np.pi/180.))
    return uwd, vwd

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def key_substr_search(keys,substr):
    selected_key=None
    for key in keys:
        if substr in key:
            selected_key=key
    return selected_key

def compute_base_bufrdict(filename,tags):
    # Given a debufr file and a list of tags, this function will retrieve a vector of
    # values containing the value provided for a tag on each line of the debufr file
    # containing the tag, and then reshape arrays into (nobs,nreps) replications for
    # any vector that has some multiple of nobs entries (based on comparison to TYPE,
    # which should have only one entry per observation).
    #
    # INPUTS
    #   filename: name of debufr file
    #   tags: list of tags, ideally in '######  ABCD' format, with each key of the
    #         output dictionary being 'ABCD'
    #
    # OUTPUTS
    #   basedict: dictionary containing a key for each tag as well as a TYPE key
    #             containing the BUFR tank type in 'NC005###' format
    #
    # Initialize an empty dictionary for output
    basedict={}
    # Open file-handle for filename (read-only)
    hdl=open(filename,'r')
    # Dump entire text of debufr into a string
    ftxt=hdl.read()
    # Explicit search for BUFR message TYPE for TYPE key
    tag='MESSAGE TYPE'
    tagKey='TYPE'
    # Initialize empty list of tag values and set default tag_beg, tag_end
    tag_vals=[]
    tag_beg=9999
    tag_end=0
    # Perform while-loop searching for each instance of tag
    while tag_beg > 0:
        # Search for tag between the end of the last tag and the end of the file
        tag_beg = ftxt.find(tag,tag_end,len(ftxt))
        tag_end = tag_beg + len(tag)+9 # set tag_end to 9 characters after tag (' NC005###')
        # If the tag is found (tag_beg is not -1), append the beginning location to tag_begs
        if (tag_beg > 0):
            strt=tag_beg
            fnsh=ftxt.find('\n',strt)
            val=ftxt[strt:fnsh].split()[2] # 3rd element in whitespace-delimited split string
            tag_vals.append(val) # Append in native (string) format, we know this is 'NC005###'
    
    basedict[tagKey]=np.asarray(tag_vals).squeeze()
    # Loop through all tags
    for tag in tags:
        # Define tagKey as ending 'ABCD' of '######  ABCD' tag
        tagKey=tag.split()[1] # Name that is used to define dictionary key
        # Initialize empty list of tag values and set default tag_beg, tag_end
        tag_vals=[]
        tag_beg=9999
        tag_end=0
        # Perform while-loop searching for each instance of tag
        while tag_beg > 0:
            # Search for tag between the end of the last tag and the end of the file
            tag_beg = ftxt.find(tag,tag_end,len(ftxt))
            tag_end = tag_beg + len(tag)
            # If the tag is found (tag_beg is not -1), append the beginning location to tag_begs
            if (tag_beg > 0):
                strt=tag_beg
                fnsh=ftxt.find('\n',strt)
                val=ftxt[strt:fnsh].split()[2] # 3rd element in whitespace-delimited split string
                if val=='MISSING':
                    tag_vals.append(np.nan)
                elif can_float(val):
                    tag_vals.append(float(val))
                else:
                    tag_vals.append(val)
        # Load vector of float values in dictionary under tagKey
        basedict[tagKey]=np.asarray(tag_vals).squeeze()
    # Realign arrays that are a x-multiple of the length of TYPE as replications, [len(TYPE),x]
    for key in basedict.keys():
        typ=np.copy(basedict['TYPE'])
        x=np.copy(basedict[key])
        if np.size(x)>np.size(typ):
            nreps=int(np.size(x)/np.size(typ)) # number of replications
            print(key,'larger than TYPE by factor of',nreps)
            y=np.reshape(x,(np.size(typ),nreps))
            basedict[key]=y
    # Return
    return basedict

def compute_quality_info(bdict):
    # Replicates code-blocks in read_satwnd.f90 to extract appropriate
    # quality information (QI or EE) based on values of GNAP, OGCE, and
    # PCCF replications based on SAID and (tank) TYPE
    #
    # Currently does not work on types that use AMVQI instead of PCCF for
    # quality info, will maybe work on this in the future.
    #
    # INPUTS
    #   bdict: dictionary containing TYPE, SAID, OGCE, GNAP, and PCCF values/replications
    # OUTPUTS
    #   bdict: modified dictionary containing new QIFN, QIFY, and EE values
    #
    # NOTES:
    #        Some ob-types use 'GNAPS' instead of GNAP, so we will use whatever key *includes*
    #        'GNAP' rather than search for a GNAP key specifically.
    key=key_substr_search(bdict.keys(),'SAID')
    nobs=np.shape(bdict[key])[0]
    # Initialize output vectors
    qify_vec=np.nan*np.ones((nobs,))
    qifn_vec=np.nan*np.ones((nobs,))
    ee_vec=np.nan*np.ones((nobs,))
    # Loop through obs, find quality info based on SAID and (BUFR tank) TYPE
    for i in range(nobs):
        key=key_substr_search(bdict.keys(),'SAID')
        said=bdict[key][i]
        key=key_substr_search(bdict.keys(),'OGCE')
        ogce=bdict[key][i]
        key=key_substr_search(bdict.keys(),'GNAP')
        gnap=bdict[key][i]
        key=key_substr_search(bdict.keys(),'PCCF')
        pccf=bdict[key][i]
        key=key_substr_search(bdict.keys(),'TYPE')
        tank=bdict[key][i]
        qify=1.0e+10
        qifn=1.0e+10
        ee=1.0e+10
        if any((np.isin(tank,['NC005064','NC005065','NC005066']))&((said<80.)&(said>=50.))): # EUMETSAT range
            for j in range(3,9):
                if (qify<105.)&(qifn<105.)&(ee<105.): break
                if (~np.isnan(gnap[j]))&(~np.isnan(pccf[j])):
                    if (gnap[j]==1.)&(qify>105.):
                        qify=pccf[j]
                    elif (gnap[j]==2.)&(qifn>105.):
                        qifn=pccf[j]
                    elif (gnap[j]==3.)&(ee>105.):
                        ee=pccf[j]
        if any((np.isin(tank,['NC005044','NC005045','NC005046']))&((said<199.)&(said>=100.))): # JMA range
            for j in range(3,9):
                if (qify<105.)&(qifn<105.)&(ee<105.): break
                if (~np.isnan(gnap[j]))&(~np.isnan(pccf[j])):
                    if (gnap[j]==101.)&(qify>105.):
                        qify=pccf[j]
                    elif (gnap[j]==102.)&(qifn>105.):
                        qifn=pccf[j]
                    elif (gnap[j]==103.)&(ee>105.):
                        ee=pccf[j]
        if any((np.isin(tank,['NC005010','NC005011','NC005012']))&((said<299.)&(said>=250.))): # NESDIS range
            for j in range(0,8):
                if (qify<105.)&(qifn<105.)&(ee<105.): break
                if (~np.isnan(gnap[j]))&(~np.isnan(pccf[j])):
                    if (gnap[j]==1.)&(qify>105.):
                        qify=pccf[j]
                    elif (gnap[j]==3.)&(qifn>105.):
                        qifn=pccf[j]
                    elif (gnap[j]==4.)&(ee>105.):
                        ee=pccf[j]
        if any((np.isin(tank,['NC005070','NC005071']))&((said<799.)&(said>=700.))): # MODIS range
            for j in range(0,8):
                if (qify<105.)&(qifn<105.)&(ee<105.): break
                if (~np.isnan(gnap[j]))&(~np.isnan(pccf[j])):
                    if (gnap[j]==1.)&(qify>105.):
                        qify=pccf[j]
                    elif (gnap[j]==3.)&(qifn>105.):
                        qifn=pccf[j]
                    elif (gnap[j]==4.)&(ee>105.):
                        ee=pccf[j]
        if any((np.isin(tank,['NC005080']))&((said==10.)|((said<=223.)&(said>=200.)))): # AVHRR range
            for j in range(0,8):
                if (qify<105.)&(qifn<105.)&(ee<105.): break
                if (~np.isnan(gnap[j]))&(~np.isnan(pccf[j])):
                    if (gnap[j]==1.)&(qify>105.):
                        qify=pccf[j]
                    elif (gnap[j]==3.)&(qifn>105.):
                        qifn=pccf[j]
                    elif (gnap[j]==4.)&(ee>105.):
                        ee=pccf[j]
        if any((np.isin(tank,['NC005019']))&((said<=299.)&(said>=250.))): # NESDIS (swIR) range
            for j in range(0,7):
                if (qify<105.)&(qifn<105.)&(ee<105.): break
                if (~np.isnan(gnap[j]))&(~np.isnan(pccf[j])):
                    if (gnap[j]==1.)&(qify>105.):
                        qify=pccf[j]
                    elif (gnap[j]==3.)&(qifn>105.):
                        qifn=pccf[j]
                    elif (gnap[j]==4.)&(ee>105.):
                        ee=pccf[j]
        if any((np.isin(tank,['NC005072']))&((said==854.))): # LEOGEO range
            qify=pccf[0]
            qifn=pccf[1]
            ee=pccf[2]
        if any((np.isin(tank,['NC005090']))&((said<=250.)&(said>=200.))): # VIIRS range
            for j in range(0,7):
                if (qify<105.)&(qifn<105.)&(ee<105.): break
                if (~np.isnan(gnap[j]))&(~np.isnan(pccf[j])):
                    if (gnap[j]==1.)&(qify>105.):
                        qify=pccf[j]
                    elif (gnap[j]==3.)&(qifn>105.):
                        qifn=pccf[j]
                    elif (gnap[j]==4.)&(ee>105.):
                        ee=pccf[j]
        if any((np.isin(tank,['NC005067','NC005068','NC005069']))&((said<80.)&(said>=50.))): # (NEW) EUMETSAT range
            print('cannot currently process (NEW) EUMETSAT winds')
        if any((np.isin(tank,['NC005081']))&((said<10.))): # NESDIS METOP-B/C range
            print('cannot currently process NESDIS METOP-B/C winds')
        if any((np.isin(tank,['NC005091']))&((said<=250.)&(said>=200.))): # VIIRS NPP/N20 range
            print('cannot currently process VIIRS NPP/N20 winds')
        if any((np.isin(tank,['NC005030','NC005031','NC005032','NC005034','NC005039']))&((said<=299.)&(said>=250.))): # (NEW) NESDIS GOES-R range
            print('cannot currently process (NEW) NESDIS GOES-R winds')
        # Fill vectors with data, or leave as nan if no data is available
        if (qify<105.): qify_vec[i]=qify
        if (qifn<105.): qifn_vec[i]=qifn
        if (ee<105.): ee_vec[i]=ee
    # Generate new dictionary entries and fill with quality information
    bdict['QIFY']=qify_vec
    bdict['QIFN']=qifn_vec
    bdict['EE']=ee_vec
    # Return modified dictionary
    return bdict

def ob_density_plot(ob_lat,ob_lon,ob_pre,lat_rng,lon_rng,figax):
    # Generates a heatmap of observation density by latitude and longitude
    # Splits into 2 heatmaps for upper- vs lower-tropospheric ob density based on pressure
    # Designed to fit into the following axis space:
    #    fig.add_axes([0.075,0.075,0.92,0.92],projection=ccrs.PlateCarree()) <- or some other projection type
    # INPUTS
    #   ob_lat: vector of observation latitudes
    #   ob_lon: vector of observation longitudes
    #   lat_rng: vector of latitude bin-edges
    #   lon_rng: vector of longitude bin-edges
    #   figax: figure axis (MUST include projection)
    # OUTPUTS
    #   figreturn: returned axis
    # DEPENDENCIES: matplotlib, numpy, cartopy
    #
    # Compute total ob-count
    totobs=np.size(ob_lat)
    # Compute 2d histogram for upper-troposphere
    idx=np.where(ob_pre<50000.) # Pa
    uH,xe,ye=np.histogram2d(ob_lon[idx],ob_lat[idx],bins=(lon_rng,lat_rng))
    # Compute 2d histogram for lower-troposphere
    idx=np.where(ob_pre>=50000.) # Pa
    lH,xe,ye=np.histogram2d(ob_lon[idx],ob_lat[idx],bins=(lon_rng,lat_rng))
    # Compute bin centers
    xc=0.5*(xe[0:-1]+xe[1:])
    yc=0.5*(ye[0:-1]+ye[1:])
    # Define plot projection transform
    transform=figax.projection
    figreturn=figax
    # Generate plot
    colmap=cm.get_cmap('gist_ncar').copy()
    colmap=truncate_colormap(colmap,0.15,0.35,n=256)
    lpfill=figax.pcolormesh(xc, yc, lH.T, transform=transform, cmap=colmap,alpha=0.67,vmin=0.05*np.max(uH+lH),vmax=np.max(uH+lH))
    lpfill.cmap.set_under('w')
    colmap=cm.get_cmap('gist_ncar').copy()
    colmap=truncate_colormap(colmap,0.60,0.80,n=256)
    upfill=figax.pcolormesh(xc, yc, uH.T, transform=transform, cmap=colmap,alpha=0.67,vmin=0.05*np.max(uH+lH),vmax=np.max(uH+lH))
    upfill.cmap.set_under('w')
    pmap=figax.coastlines(resolution='50m',linewidth=2)
    # Add colorbar
    plt.colorbar(upfill,label='Upper Troposphere')
    plt.colorbar(lpfill,label='Lower Troposphere')
    # Add title
    figax.set_title('ob density ({:d} observations)'.format(totobs))
    # Return 
    return figreturn

def ob_hist_latlonpre(ob_lat,ob_lon,ob_pre,lat_rng,lon_rng,pre_rng,figax1,figax2,figax3):
    # Generates histograms of ob-count by latitude, longitude, and pressure.
    # The pressure-histogram is oriented along the y-axis for ease of plot
    # interpretation. Designed to fit the 3 plots into the following axis
    # spaces:
    #   (1) pressure-histogram: fig.add_axes([0.,0.,0.3,0.90])
    #   (2) latitude-histogram: fig.add_axes([0.4,0.,0.5,0.38])
    #   (3) longitude-histogram: fig.add_axes([0.4,0.52,0.5,0.38])
    #
    #   --------   ------------------
    #   |      |   |       3        |
    #   |      |   |                |
    #   |   1  |   ------------------
    #   |      |   |       2        |
    #   |      |   |                |
    #   --------   ------------------
    #
    # INPUTS
    #   ob_lat: vector of observation latitudes
    #   ob_lon: vector of observation longitudes
    #   ob_pre: vector of observation pressures
    #   lat_rng: vector of latitude bin-edges
    #   lon_rng: vector of longitude bin-edges
    #   pre_rng: vector of pressure bin-edges
    #   figax(1,2,3): figure axes for plots 1, 2, 3
    # OUTPUTS
    #   figreturn: returned axes
    # DEPENDENCIES: matplotlib, numpy
    #
    # Generate pressure-histogram
    ax=figax1
    dy=pre_rng[1]-pre_rng[0]
    ax.hist(ob_pre,pre_rng,orientation="horizontal",facecolor='#0C00FF',edgecolor='w')
    ax.set_ylim((np.min(ob_pre)-dy,np.max(ob_pre)+dy))
    ax.invert_yaxis()
    ax.set_title('count by pressure')
    # Generate latitude-histogram
    ax=figax2
    dx=lat_rng[1]-lat_rng[0]
    ax.hist(ob_lat,lat_rng,facecolor='#E86D00',edgecolor='w')
    ax.set_xlim((np.min(ob_lat)-dx,np.max(ob_lat)+dx))
    ax.set_title('count by latitude')
    # Generate longitude-histogram
    ax=figax3
    ax.hist(ob_lon,lon_rng,facecolor='#D7D700',edgecolor='w')
    ax.set_xlim((np.min(ob_lon)-dx,np.max(ob_lon)+dx))
    ax.set_title('count by longitude')
    # Return figure axes
    return figax1, figax2, figax3

def ob_hist_spddirqi(ob_spd,ob_dir,ob_qi,spd_rng,dir_rng,qi_thresh,figax1,figax2,figax3):
    # Generates histograms of ob-count by wind speed and direction, and a pie-chart of obs >= or < qi_thresh
    # The direction-histogram is polar-oriented wth 0-deg facing south, as per wind convention
    # Designed to fit the 3 plots into the following axis
    # spaces:
    #    (1) windrose plot:    fig.add_axes([0.,0.,0.45,0.4],projection='polar')
    #    (2) speed histogram:  fig.add_axes([0.,0.52,0.9,0.33])
    #    (3) qi pi-chart:      fig.add_axes([0.5,0.,0.45,0.4])
    #
    #   -----------------------------
    #   |            2              |
    #   |                           |
    #   |----------------------------
    #   |     1      |      3       |
    #   |            |              |
    #   ----------------------------
    #
    # INPUTS
    #   ob_spd: vector of observation speeds
    #   ob_dir: vector of observation directions
    #   ob_qi: vector of observation quality-indicator values (typically QIFN or EE)
    #   spd_rng: vector of speed bin-edges
    #   dir_rng: vector of direction bin-edges
    #   qi_thresh: threshold of quality-indicator for good/bad obs
    #   figax(1,2,3): figure axes for plots 1, 2, 3
    # OUTPUTS
    #   figreturn: returned axes
    # DEPENDENCIES: matplotlib, numpy
    #
    # NOTES:
    #       Sometimes QI is all np.nan values, in which case we will present a blank pie-chart
    #
    # Generate windrose (polar histogram)
    ax=figax1
    ax.set_theta_zero_location('S')
    ax.set_theta_direction(-1)
    h,x=np.histogram(ob_dir,dir_rng)
    dx=dir_rng[1]-dir_rng[0]
    xc=0.5*(x[0:-1]+x[1:])
    ax.bar(x=xc, height=h, width=0.75*dx*np.pi/180.,facecolor='#00B143',edgecolor='w')
    ax.set_title('wind direction')
    # Generate speed histogram
    ax=figax2
    ax.hist(ob_spd,spd_rng,facecolor='#CC00E8',edgecolor='w')
    dx=spd_rng[1]-spd_rng[0]
    ax.set_xlim((0.,np.max(ob_spd)+dx))
    ax.set_title('wind speed')
    # Generate pi-chart
    ax=figax3
    labels=['qi>={:.0f}'.format(qi_thresh),'qi<{:.0f}'.format(qi_thresh)]
    y=np.where(np.isnan(ob_qi)==False)
    if (np.size(y)>0):
        sizes=[np.size(np.where(ob_qi[y]>=qi_thresh)),np.size(np.where(ob_qi[y]<qi_thresh))]
        ax.pie(sizes, labels=labels, autopct='%1.1f%%',shadow=False, startangle=90,colors=['#00C5FF','#FF2D2D'])
        ax.axis('equal')
        ax.set_title('quality indicator')
    # Return
    return figax1,figax2,figax3

def stage_scorecard(ob_lat,ob_lon,ob_pre,ob_spd,ob_dir,ob_qi,qi_thresh=85.):
    # Stages sub-figures for score-card layout
    # INPUTS
    #   ob_lat: vector of ob latitudes
    #   ob_lon: vector of ob longitudes
    #   ob_pre: vector of ob pressures (Pa)
    #   ob_spd: vector of ob speeds
    #   ob_dir: vector of ob directions
    #   ob_qi: vector of ob quality-indicator values (usually QIFN, can also be EE)
    #   qi_threshold: threshold quality-indicator value for good/bad ob (often 85., as default)
    # OUTPUTS
    #   fighdl: figure handle containing plot
    # DEPENDENCIES: matplotlib, ob_density_plot, ob_hist_latlonpre, ob_hist_spddir
    #
    # Outer figure domain
    figax=plt.figure(figsize=(21,14))
    # Split figure into a 2-col, 1-row set of subfigures
    subfigs = figax.subfigures(2, 1).flat
    # Top subfigure: ob_density_plot
    sfig=subfigs[0]
    ax=sfig.add_axes([0.075,0.075,0.92,0.92],projection=ccrs.PlateCarree())
    lat_rng=np.arange(-90.,90.1,2.5)
    lon_rng=np.arange(0.,360.1,2.5)
    ax=ob_density_plot(ob_lat,ob_lon,ob_pre,lat_rng,lon_rng,figax=ax)
    # Bottom subfigure:
    sfig=subfigs[1]
    # Split bottom subfigure into a 1-row, 2-col set of sub-subfigures
    ssfigs=sfig.subfigures(1,2).flat
    # Left sub-subfigure: ob_hist_latlonpre
    ssfig=ssfigs[0]
    ax1=ssfig.add_axes([0.,0.,0.3,0.90])
    ax2=ssfig.add_axes([0.4,0.,0.5,0.38])
    ax3=ssfig.add_axes([0.4,0.52,0.5,0.38])
    lat_rng=np.arange(-90.,90.1,5.)
    lon_rng=np.arange(0.,360.1,10.)
    pre_rng=np.arange(10000.,110000.1,5000.)
    ax1,ax2,ax3=ob_hist_latlonpre(ob_lat,ob_lon,ob_pre,lat_rng,lon_rng,pre_rng,ax1,ax2,ax3)
    # Right sub-subfigure: ob_hist_spddirqi
    ssfig=ssfigs[1]
    ax1=ssfig.add_axes([0.,0.,0.45,0.4],projection='polar')
    ax2=ssfig.add_axes([0.,0.52,0.9,0.33])
    ax3=ssfig.add_axes([0.5,0.,0.45,0.4])
    spd_rng=np.arange(0.,100.1,2.)
    dir_rng=np.arange(0.,360.1,15.)
    ax1,ax2,ax3=ob_hist_spddirqi(ob_spd,ob_dir,ob_qi,spd_rng,dir_rng,qi_thresh,ax1,ax2,ax3)
    # Return figax
    return figax

