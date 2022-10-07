# Set of functions for *.debufr.out text parsing for extracting satwinds data
#
# PyGSI compliant
#
import numpy as np
#
# General Purpose Functions
#   can_float                Returns True if input can be converted to a float, otherwise returns False
#   spddir_to_uwdvwd         Converts (spd,dir) to (u,v)
#
# Debufr Parsing Functions
#
#   compute_base_bufrdict    Searches all instances of each tag and constructs vectors of values. Converts to float if possible,
#                            and reshapes to (num obs, num replications) as necessary. Returns dictionary of keys for each tag.
#
#   compute_quality_info     Extracts QIFY, QIFN, and EE values from GNAP, OGCE, and PCCF, and constructs arrays. Returns new
#                            dictionary with QIFY, QIFN, and EE keys. Does not currently work for all message types and satIDs.
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
    nobs=np.size(bdict['SAID'])
    # Initialize output vectors
    qify_vec=np.nan*np.ones((nobs,))
    qifn_vec=np.nan*np.ones((nobs,))
    ee_vec=np.nan*np.ones((nobs,))
    # Loop through obs, find quality info based on SAID and (BUFR tank) TYPE
    for i in range(nobs):
        said=bdict['SAID'][i]
        ogce=bdict['OGCE'][i]
        gnap=bdict['GNAP'][i]
        pccf=bdict['PCCF'][i]
        tank=bdict['TYPE'][i]
        qify=1.0e+10
        qifn=1.0e+10
        ee=1.0e+10
        if (np.isin(tank,['NC005064','NC005065','NC005066']))&((said<80.)&(said>=50.)): # EUMETSAT range
            for j in range(3,9):
                if (qify<105.)&(qifn<105.)&(ee<105.): break
                if (~np.isnan(gnap[j]))&(~np.isnan(pccf[j])):
                    if (gnap[j]==1.)&(qify>105.):
                        qify=pccf[j]
                    elif (gnap[j]==2.)&(qifn>105.):
                        qifn=pccf[j]
                    elif (gnap[j]==3.)&(ee>105.):
                        ee=pccf[j]
        if (np.isin(tank,['NC005044','NC005045','NC005046']))&((said<199.)&(said>=100.)): # JMA range
            for j in range(3,9):
                if (qify<105.)&(qifn<105.)&(ee<105.): break
                if (~np.isnan(gnap[j]))&(~np.isnan(pccf[j])):
                    if (gnap[j]==101.)&(qify>105.):
                        qify=pccf[j]
                    elif (gnap[j]==102.)&(qifn>105.):
                        qifn=pccf[j]
                    elif (gnap[j]==103.)&(ee>105.):
                        ee=pccf[j]
        if (np.isin(tank,['NC005010','NC005011','NC005012']))&((said<299.)&(said>=250.)): # NESDIS range
            for j in range(0,8):
                if (qify<105.)&(qifn<105.)&(ee<105.): break
                if (~np.isnan(gnap[j]))&(~np.isnan(pccf[j])):
                    if (gnap[j]==1.)&(qify>105.):
                        qify=pccf[j]
                    elif (gnap[j]==3.)&(qifn>105.):
                        qifn=pccf[j]
                    elif (gnap[j]==4.)&(ee>105.):
                        ee=pccf[j]
        if (np.isin(tank,['NC005070','NC005071']))&((said<799.)&(said>=700.)): # MODIS range
            for j in range(0,8):
                if (qify<105.)&(qifn<105.)&(ee<105.): break
                if (~np.isnan(gnap[j]))&(~np.isnan(pccf[j])):
                    if (gnap[j]==1.)&(qify>105.):
                        qify=pccf[j]
                    elif (gnap[j]==3.)&(qifn>105.):
                        qifn=pccf[j]
                    elif (gnap[j]==4.)&(ee>105.):
                        ee=pccf[j]
        if (np.isin(tank,['NC005080']))&((said==10.)|((said<=223.)&(said>=200.))): # AVHRR range
            for j in range(0,8):
                if (qify<105.)&(qifn<105.)&(ee<105.): break
                if (~np.isnan(gnap[j]))&(~np.isnan(pccf[j])):
                    if (gnap[j]==1.)&(qify>105.):
                        qify=pccf[j]
                    elif (gnap[j]==3.)&(qifn>105.):
                        qifn=pccf[j]
                    elif (gnap[j]==4.)&(ee>105.):
                        ee=pccf[j]
        if (np.isin(tank,['NC005019']))&((said<=299.)&(said>=250.)): # NESDIS (swIR) range
            for j in range(0,7):
                if (qify<105.)&(qifn<105.)&(ee<105.): break
                if (~np.isnan(gnap[j]))&(~np.isnan(pccf[j])):
                    if (gnap[j]==1.)&(qify>105.):
                        qify=pccf[j]
                    elif (gnap[j]==3.)&(qifn>105.):
                        qifn=pccf[j]
                    elif (gnap[j]==4.)&(ee>105.):
                        ee=pccf[j]
        if (np.isin(tank,['NC005072']))&((said==854.)): # LEOGEO range
            qify=pccf[0]
            qifn=pccf[1]
            ee=pccf[2]
        if (np.isin(tank,['NC005090']))&((said<=250.)&(said>=200.)): # VIIRS range
            for j in range(0,7):
                if (qify<105.)&(qifn<105.)&(ee<105.): break
                if (~np.isnan(gnap[j]))&(~np.isnan(pccf[j])):
                    if (gnap[j]==1.)&(qify>105.):
                        qify=pccf[j]
                    elif (gnap[j]==3.)&(qifn>105.):
                        qifn=pccf[j]
                    elif (gnap[j]==4.)&(ee>105.):
                        ee=pccf[j]
        if (np.isin(tank,['NC005067','NC005068','NC005069']))&((said<80.)&(said>=50.)): # (NEW) EUMETSAT range
            print('cannot currently process (NEW) EUMETSAT winds')
        if (np.isin(tank,['NC005081']))&((said<10.)): # NESDIS METOP-B/C range
            print('cannot currently process NESDIS METOP-B/C winds')
        if (np.isin(tank,['NC005091']))&((said<=250.)&(said>=200.)): # VIIRS NPP/N20 range
            print('cannot currently process VIIRS NPP/N20 winds')
        if (np.isin(tank,['NC005030','NC005031','NC005032','NC005034','NC005039']))&((said<=299.)&(said>=250.)): # (NEW) NESDIS GOES-R range
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


