# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 11:26:19 2017

@author: Ewan Pinnington

utility functions for plotting jules output
"""
import matplotlib
import netCDF4 as nc
import netCDF4_utils as nc_utils
import numpy as np
import datetime as dt
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
#  plt.rcParams['animation.ffmpeg_path'] = '/opt/tools/bin/ffmpeg'
from mpl_toolkits.basemap import Basemap, cm
import os
import seaborn as sns


def draw_map(low_lat=4.5, high_lat=12, low_lon=-3.5, high_lon=1.5, ax='None'):
    """
    Creates a cylindrical Basemap instance.
    :param low_lat: lower left lat
    :param high_lat: upper right lat
    :param low_lon: lower left lon
    :param high_lon: upper right lon
    :param ax: axis to create instance for
    :return: Basemap instance
    """
    if ax == 'None':
        m = Basemap(projection='cyl',resolution='i',
                    llcrnrlat=low_lat, urcrnrlat=high_lat,
                    llcrnrlon=low_lon, urcrnrlon=high_lon)
    else:
        m = Basemap(projection='cyl',resolution='i',
                    llcrnrlat=low_lat, urcrnrlat=high_lat,
                    llcrnrlon=low_lon, urcrnrlon=high_lon, ax=ax)
    # draw coastlines, state and country boundaries, edge of map.
    m.drawcoastlines()
    m.drawstates()
    m.drawcountries()
    parallels = np.arange(0., 81, 1.)
    # labels = [left,right,top,bottom]
    m.drawparallels(parallels, labels=[False, True, True, False])
    meridians = np.arange(0., 361., 1.)
    m.drawmeridians(meridians, labels=[True, False, False, True])
    return m


def weighted_rmse(obs, mod, err):
    """
    Calculates the RMSE weight by observation std
    :param obs: observations
    :param mod: modelled observations
    :param err: observation error
    :return: weighted RMSE
    """
    obs2 = obs[np.logical_not(np.isnan(obs))]
    mod2 = mod[np.logical_not(np.isnan(obs))]
    err2 = err[np.logical_not(np.isnan(obs))]
    innov = [(obs2[i]-mod2[i])**2/err2[i] for i in xrange(len(obs2))]
    rmse = np.sqrt(np.sum(innov)/len(obs2))
    return rmse


def fourier_trans(dat, modes, cci=0):
    """
    Calculates fft of data, then removes modes above given point and returns inverse ft
    :param dat: data to ft
    :param modes: amount of modes to remove
    :param cci: is the data from esa cci with gaps
    :return: inverse ft
    """
    if cci is True:
        cv_i = np.arange(len(dat))
        mask_cv = np.isfinite(dat)
        dat = np.interp(cv_i, cv_i[mask_cv], dat[mask_cv])
    dat_ft = np.fft.fft(dat)
    dat_ft[modes:] = 0
    dat_ift = np.fft.ifft(dat_ft)
    return dat_ift