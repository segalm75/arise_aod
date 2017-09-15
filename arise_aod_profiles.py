
# coding: utf-8

# # Plotting some ARISE AOD profiles for Polar WRF paper
# Sep-7 and Sep-17 are "reference" days, and Sep-9 and Sep-24 are case study days
# for comparing Polar-WRF with various cloud parameterization schemes

# In[27]:

# upload moduls

import pandas as pd
import numpy as np
import scipy as sio
import scipy.stats as stats

get_ipython().magic(u'matplotlib inline')
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import scipy as sio
import os, math, sys
import datetime as dt
import time as tm
import seaborn as sns

# to check moduls versions: pd.__version__


# read csv files

# In[64]:

# read ARISE AOD files

# set data dir

data_dir ='..//..//py_data//arise_aod//'

## loop over arise dates:

days2analyze = ['20140907','20140909','20140917','20140924']

#days2analyze = ['20140907']

i = 0
df = pd.DataFrame([])

for s in days2analyze:
    i=+ 1
    print 'days2analyze= ' + s
    fname = data_dir + '{}aod.csv'.format(s)
    print 'filename = ' + fname
    
    # read csv using the pandas library
    
    data            = pd.read_csv(fname, index_col=False, na_values=[-9999.0])
    data['Date']    = pd.Timestamp(s)
    data['TimeUTC'] = (data.Start_UTC/86400.)*24.
    
    df = df.append(data)


# assign new index
#df.set_index(['Date','TimeUTC'], inplace=True)
df.head()
        
        
    


# In[3]:

len(df)


# In[4]:

df.columns.tolist()


# ## add profile number to data frame
# ##-----------------------------------

# In[65]:

prof_data    = '..//..//py_data//arise_aod//air_atmProfiles_4plotting_created_on_2017-08-19_MERRA2.csv'
profile_data = pd.read_csv(prof_data, index_col=False)
profile_data.head()
profile_data.ProfileNum = profile_data.ProfileNum.astype(int)
profile_data.Day        = profile_data.Day.astype(int)
profile_data.head()
#profile = profile_data.ProfileNum[profile_data.Day==7].unique()
#profile
#for ii in profile:
#    print ii


# In[63]:

# read ARISE AIR files - this is just test

# set data dir

# data_dir ='..//..//py_data//arise_aod//'

## loop over arise dates:

#days2analyze = ['20140907','20140909','20140917','20140924']

#days2analyze = ['20140907']

#i = 0
#df = pd.DataFrame([])

#for s in days2analyze:
#    i=+ 1
#    print 'days2analyze= ' + s
#    fname = data_dir + '{}air.csv'.format(s)
#    print 'filename = ' + fname
    
    # read csv using the pandas library
    
#    data_air            = pd.read_csv(fname, index_col=False, na_values=[-9999.0])
#    data_air['Date']    = pd.Timestamp(s)
#    data_air['TimeUTC'] = (data_air.Start_UTC/86400.)*24.
    
#    df = df.append(data_air)


# assign new index
#df.set_index(['Date','TimeUTC'], inplace=True)
#df.head()


# In[61]:

## plot flight altitude versus time from profile dataset:
## in arise_aod_stats the aod data starts after 00:00UTC
#fig = plt.figure(figsize=(7.195, 7.195), dpi=100)
#r-c-m-k
#plt.plot(profile_data['TimeUTC'][(dfp['Date']=='2014-09-07')],dfp['AOD0501'][(dfp['Date']=='2014-09-07')],
#         'ob', label='2014-09-07')
#plt.plot(dfp['TimeUTC'][(dfp['Date']=='2014-09-09')],dfp['AOD0501'][(dfp['Date']=='2014-09-09')],
#         'og', label='2014-09-09')
#plt.plot(dfp['TimeUTC'][(dfp['Date']=='2014-09-17')],dfp['AOD0501'][(dfp['Date']=='2014-09-17')],
#         'or', label='2014-09-17')
#plt.plot(dfp['TimeUTC'][(dfp['Date']=='2014-09-24')],dfp['AOD0501'][(dfp['Date']=='2014-09-24')],
#         'om', label='2014-09-24')
#plt.legend(loc='best') 
#plt.xlabel('TimeUTC')
#plt.ylabel('AOD 500 nm')
#for axis in ['top','bottom','left','right']:
#  ax.spines[axis].set_linewidth(0.5)
#plt.show() 


# In[92]:

# test - go over each of the profiles for the selected days:
days = [7, 9, 17, 24]

data_reduced = profile_data[profile_data.Day==days[2]]
aod_data_reduced = df[df.Date==days2analyze[2]]
print df.shape
print aod_data_reduced.shape

for k in range(len(days)):
    print days2analyze[k]

#print aod_data_reduced.head()
#alt_data_reduced = df.GPS_alt[df.Date==days2analyze[ii-1]]
#profile = profile_data.ProfileNum[profile_data.Day==i].unique()
profile = data_reduced.ProfileNum.unique()
print "profile_len"
print len(profile)

min_UTC = [0]*len(profile)
max_UTC = [0]*len(profile)

min_UTC[0] = data_reduced.prof_UTC[data_reduced.ProfileNum==1].min()
max_UTC[0] = data_reduced.prof_UTC[data_reduced.ProfileNum==1].max()
if min_UTC[0]>=24:
    min_UTC[0] = min_UTC[0] - 24
if max_UTC[0]>=24:
    max_UTC[0] = max_UTC[0] - 24
print min_UTC[0]
print max_UTC[0]
print "aod_reduced_min"
print aod_data_reduced.TimeUTC.min()
print "aod_reduced_max"
print aod_data_reduced.TimeUTC.max()
# see if we have AOD profiles:
aod_profile = aod_data_reduced.AOD0501[(aod_data_reduced.TimeUTC<=max_UTC[0])&(aod_data_reduced.TimeUTC>=min_UTC[0])]
print "length aod profile"
print len(aod_profile)



# In[112]:

# go over each of the profiles for the selected days:
days = [7, 9, 17, 24]
days2analyze = ['20140907','20140909','20140917','20140924']


#days = [7]
#days2analyze = ['20140907']

# setting up dummy plot
# Using contourf to provide my colorbar info, then clearing the figure
colormap = plt.cm.gist_ncar #nipy_spectral, Set1,Paired
sns.reset_orig()
Z = [[0,0],[0,0]]
levels = range(1,11,1)# this is profile number (max 10 profiles) np.arange(0.0, 1.0, 0.1)#range(0.,1.,0.1)# this is AOD
fig2 = plt.figure()
CS3 = plt.contourf(Z, levels, cmap=colormap)
plt.clf()

#k = [0,1,2,3]
#for m,j in enumerate(ax1.lines):
#        j.set_color(colors[m])
        
for k,i in enumerate(days):
    
    #k=+ 1
    print "i"
    print i
    print "k"
    print k
    print days2analyze[k]
    # go over each profile
    data_reduced = profile_data[profile_data.Day==i]
    aod_data_reduced = df[(df.Date==days2analyze[k])&(df.qual_flag==0)]
    print aod_data_reduced.head()
    #alt_data_reduced = df.GPS_alt[df.Date==days2analyze[ii-1]]
    #profile = profile_data.ProfileNum[profile_data.Day==i].unique()
    profile = data_reduced.ProfileNum.unique()
    min_UTC = [0]*len(profile)
    max_UTC = [0]*len(profile)
    
    # set figure panel for each date
    #----------------------------------
    #subdata = df.loc[size_data['Flight_Date']==f]
    #UTC     = subdata.iloc[:,0]
    #UTC     = np.around(UTC, decimals=2)
    #subdata = subdata.iloc[:,1:93]
    
    # set figure to plot
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    
    
    # for each profile get min max time values
    for ii in profile:
        print "ii"
        print ii
        min_UTC[ii-1] = data_reduced.prof_UTC[data_reduced.ProfileNum==ii].min()
        max_UTC[ii-1] = data_reduced.prof_UTC[data_reduced.ProfileNum==ii].max()
        if min_UTC[ii-1]>=24:
            min_UTC[ii-1] = min_UTC[ii-1] - 24
        if max_UTC[ii-1]>=24:
            max_UTC[ii-1] = max_UTC[ii-1] - 24
    #print min_UTC
    #print max_UTC
    
        # go over the AOD data and add profile number
        # days2analyze = ['20140907','20140909','20140917','20140924']
        # get AOD profile data for each of the profiles
        aod_profile = aod_data_reduced.AOD0501[(aod_data_reduced.TimeUTC<=max_UTC[ii-1])&(aod_data_reduced.TimeUTC>=min_UTC[ii-1])]
        print "length aod profile"
        print len(aod_profile)
        alt_profile = aod_data_reduced.GPS_alt[(aod_data_reduced.TimeUTC<=max_UTC[ii-1])&(aod_data_reduced.TimeUTC>=min_UTC[ii-1])]
        
        
        ## plot each profile:
        #---------------------
        
        #ax1.plot(aod_profile,alt_profile,'-',linewidth=2.0)
        ax1.plot(aod_profile,alt_profile,'.')
        
    ## set colors for each profile    
        
    colors = [colormap(i) for i in np.linspace(0, 1,len(ax1.lines))]
    
    for m,j in enumerate(ax1.lines):
        j.set_color(colors[m])
        
    ## finalize plot
    #ax1.set_xscale('log')
    plt.xlabel('AOD [-]')
    plt.ylabel('Altitude [m]')
    plt.title(days2analyze[k])
    #plt.ylim([0, 5000])
    
    cb = plt.colorbar(CS3) # using the colorbar info I got from contourf
    #fig1.colorbar
    #cb = plt.colorbar()
    #cb.ax.set_title('This is a title')
    cb.set_label('Profile #')
    # save figure per site
    fi = '../../py_figs/arise_aod/' + days2analyze[k] + '_AOD_vs_Alt_perProfile.png'
    plt.savefig(fi, bbox_inches='tight',dpi=1000)
        


# In[ ]:

## plot multiple AOD profiles for each of the flight day:

## plot distributions per Site

# setting up dummy plot
# Using contourf to provide my colorbar info, then clearing the figure
colormap = plt.cm.gist_ncar #nipy_spectral, Set1,Paired
sns.reset_orig()
Z = [[0,0],[0,0]]
levels = range(0,1,0.1)# this is AOD
fig2 = plt.figure()
CS3 = plt.contourf(Z, levels, cmap=colormap)
plt.clf()

for f in days2analyze:
    
    subdata = df.loc[size_data['Flight_Date']==f]
    UTC     = subdata.iloc[:,0]
    UTC     = np.around(UTC, decimals=2)
    subdata = subdata.iloc[:,1:93]
     
    #print "diameter shape"
    #print diameter.shape
    #print "y"
    #print subdata.iloc[0,:].shape
    # set figure to plot
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    
    for i in range(len(subdata)):
        #ax1.plot(radius,subdata.iloc[i,:].values,'-',linewidth=2.0,label=Jday[i])
        ax1.plot(diameter,subdata.iloc[i,:].values,'-',linewidth=2.0)

    
       
    colors = [colormap(i) for i in np.linspace(0, 1,len(ax1.lines))]
    
    for i,j in enumerate(ax1.lines):
        j.set_color(colors[i])
    

    #ax1.legend(loc=2)
    #ax1.set_xscale('log')
    plt.xlabel('Diameter [micron]')
    plt.ylabel('dN(D)/dlog(D)')
    plt.title(f)
    plt.ylim([0, 60000])
    #sm = plt.cm.ScalarMappable(cmap=colormap, norm=fig1.normalize(min=0, max=1))
    cb = plt.colorbar(CS3) # using the colorbar info I got from contourf
    #fig1.colorbar
    #cb = plt.colorbar()
    #cb.ax.set_title('This is a title')
    cb.set_label('UTC [hr]')
    # save figure per site
    fi = '../../py_figs/oracles_aerosol/' + f + '_size_dist_byTime_matplolibStyle.png'
    plt.savefig(fi, bbox_inches='tight',dpi=1000)


# In[5]:

# grouping
df_part = df[df['Date']=='2014-09-07']['AOD0501']
df_part.head()
len(df_part)


# In[6]:

# plots
matplotlib.style.use('ggplot')


# In[159]:

# timeseries
df_part.plot()


# In[19]:

# filtered time series
df_part = df['AOD0501'][(df['Date']=='2014-09-07')&(df['qual_flag']==0)]
df_part.plot()


# Plot histogram of AOD

# In[9]:

#output = plt.hist(df[df['Date']=='2014-09-07']['AOD0501'], bins=10, histtype='step', label=['2014-09-07'])
#plt.legend(loc='upper right')

# parse by qual_flag==0
ax1=df_part.plot.hist(alpha=0.8,bins=50,histtype='step')
fig = ax1.get_figure()
fig.savefig('../../py_figs/arise_aod/20140907AOD0501hist.png', bbox_inches='tight')


# In[166]:

# overlay hist plots
# filter df for qual_flag=0
df_qual = df[df['qual_flag']==0]
ax=df_qual.groupby('Date').AOD0501.value_counts().unstack(0).fillna(0).plot()
fig = ax.get_figure()
ax.set_ylabel('frequency')
fig.savefig('../../py_figs/arise_aod/AOD0501hist_resize.png', bbox_inches='tight',dpi=1000)


# In[11]:

# check amount of data not filtered for clouds
aod_group = df.groupby('Date')
aod_group.size()


# In[12]:

# check amount of data  filtered for clouds
aod_group_filt = df_qual.groupby('Date')
aod_group_filt.size()


# In[13]:

len(df_qual)


# In[167]:

## create subdata frame 
#df_part = df['AOD0501'][(df['Date']=='2014-09-07')&(df['qual_flag']==0)]
#df_part = df['Date','TimeUTC','GPS_alt','AOD0501'][(df['qual_flag']==0)]
cols =['Date','TimeUTC','GPS_alt','AOD0501','qual_flag']
dfp = df[cols]
# filter out the state level data using boolean masking on the SUMLEV column
dfp=dfp[dfp['qual_flag']==0]
dfp.head()

#import matplotlib as mpl
#mpl.rcParams['axes.linewidth'] = 1 #set the value globally

fig = plt.figure(figsize=(7.195, 7.195), dpi=100)
#r-c-m-k
plt.plot(dfp['TimeUTC'][(dfp['Date']=='2014-09-07')],dfp['AOD0501'][(dfp['Date']=='2014-09-07')],
         'ob', label='2014-09-07')
plt.plot(dfp['TimeUTC'][(dfp['Date']=='2014-09-09')],dfp['AOD0501'][(dfp['Date']=='2014-09-09')],
         'og', label='2014-09-09')
plt.plot(dfp['TimeUTC'][(dfp['Date']=='2014-09-17')],dfp['AOD0501'][(dfp['Date']=='2014-09-17')],
         'or', label='2014-09-17')
plt.plot(dfp['TimeUTC'][(dfp['Date']=='2014-09-24')],dfp['AOD0501'][(dfp['Date']=='2014-09-24')],
         'om', label='2014-09-24')
plt.legend(loc='best') 
plt.xlabel('TimeUTC')
plt.ylabel('AOD 500 nm')
#for axis in ['top','bottom','left','right']:
#  ax.spines[axis].set_linewidth(0.5)
plt.show() 
fig.savefig('../../py_figs/arise_aod/AOD0501timeseries_resized_recolor.png', bbox_inches='tight',dpi=1000)


# # Bin AOD data by altitude for each of the days

# In[ ]:




# In[ ]:




# In[61]:


#bins = np.linspace(df_qual.GPS_alt.min(), df_qual.GPS_alt.max(), 20)
alt_bins = np.arange(0.,8000.,500.)
alt_bins

#alts.groupby(['Date']).AOD0501.value_counts().plot(kind='bar')

#orientation='horizontal'


# In[62]:

alt_names = alt_bins.astype('str')[0:-1]
alt_names


# In[160]:

# create alt categories
#df_qual['alt_categories'] = pd.cut(df_qual['GPS_alt'], alt_bins, labels=alt_names)
#df_qual.head()


# In[157]:

df_qual.groupby('Date').plot(kind='bar',y='AOD0501',x='alt_categories');


# # Improved binning code

# In[63]:

# make sure no NA
df_qual = df_qual.dropna() #The mapper won't be happy with NA values
len(df_qual)


# In[64]:

df_qual.head()


# # a warpper to bin data

# In[65]:


#Write a short mapper that bins data
def map_bin(x, bins):
    kwargs = {}
    if x == max(bins):
        kwargs['right'] = True
    bin = bins[np.digitize([x], bins, **kwargs)[0]]
    bin_lower = bins[np.digitize([x], bins, **kwargs)[0]-1]
    return '[{0}-{1}]'.format(bin_lower, bin)


# In[66]:

# bin data
freq_bins = np.arange(0., 8000., 500.)


# In[67]:

# check mapper
map_bin(500, freq_bins)


# In[68]:

# map our dataframe
df_qual['Binned'] = df_qual['GPS_alt'].apply(map_bin, bins=freq_bins)
df_qual[['GPS_alt','Binned']][:10]


# Group data by bin and average per each bin

# In[69]:

grouped = df_qual.groupby(['Date','Binned'])
grouped_data = grouped.mean()
grouped_data.head()
#len(grouped_data)


# In[140]:

gd = pd.DataFrame(grouped_data['AOD0501'])
gd.head()
gd=gd.reset_index()
gd.head()
gd.rename(columns={'Binned':'Altitude bins [m]'},inplace=True)
gd.columns.tolist()
#?df.rename()


# In[141]:

# list unique altitude bins values
list(gd['Altitude bins [m]'].unique())


# In[142]:

# order the altitude bins:
gd['Altitude bins [m]']=gd['Altitude bins [m]'].astype('category',
                          categories=['[7000.0-7500.0]','[6500.0-7000.0]','[6000.0-6500.0]','[5500.0-6000.0]',
                                      '[5000.0-5500.0]','[4500.0-5000.0]','[3500.0-4000.0]','[3000.0-3500.0]',
                                      '[2500.0-3000.0]','[2000.0-2500.0]','[1500.0-2000.0]','[1000.0-1500.0]',
                                      '[500.0-1000.0]','[0.0-500.0]'],
                          ordered=True)
gd.head()


# In[146]:

import seaborn as sns

sns.set(style="whitegrid", color_codes=True)

ax=sns.swarmplot(x="AOD0501", y="Altitude bins [m]", hue="Date", data=gd,size=8,linewidth=1);
fig = ax.get_figure()
fig.savefig('../../py_figs/arise_aod/AOD0501_byAltBins.png', bbox_inches='tight')


# In[168]:

# add lines
sns_plot = sns.factorplot(x="AOD0501", y="Altitude bins [m]", hue="Date", data=gd, size=8,linewidth=1,legend_out=False);
sns_plot.savefig('../../py_figs/arise_aod/AOD0501_byAltBins_wLines_resize.png',dpi=1000)


# # Combine AOD data with ARISE profiles

# In[ ]:

# load profile summary .csv file

