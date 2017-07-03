import numpy as np
import netCDF4 as nc
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plot_utils as plt_ut
import os
import shutil as sh
import glob
# Rewrite this as a class?


class PlotJules:
    def __init__(self, prior_nc, posterior_nc, esa_nc='/export/cloud/nceo/users/if910917/esa_cci_v03'
                                                      '/ghana/esacci_sm_1989_2014_regrid.nc'):
        # extract data from JULES runs
        self.prior_dat = nc.Dataset(prior_nc, 'r')
        self.posterior_dat = nc.Dataset(posterior_nc, 'r')
        self.lats = self.prior_dat.variables['latitude'][:,0]
        self.lons = self.prior_dat.variables['longitude'][0]
        self.time_var = self.prior_dat.variables['time']
        self.times = nc.num2date(self.time_var[:], self.time_var.units)
        self.prior_sm = self.prior_dat.variables['smcl'][:, 0, :, :] / 100.
        self.posterior_sm = self.posterior_dat.variables['smcl'][:, 0, :, :] / 100.
        # extract cci obs
        self.esa_cci_dat = nc.Dataset(esa_nc, 'r')  # CCI sm observations
        self.strt_idx = nc.date2index(self.times[0], self.esa_cci_dat.variables['time'])
        self.end_idx = nc.date2index(self.times[-1], self.esa_cci_dat.variables['time'])
        self.lat_idx1 = np.where(self.esa_cci_dat.variables['lat'][:] == self.lats[0])[0][0]
        self.lat_idx2 = np.where(self.esa_cci_dat.variables['lat'][:] == self.lats[-1])[0][0]
        self.lon_idx1 = np.where(self.esa_cci_dat.variables['lon'][:] == self.lons[0])[0][0]
        self.lon_idx2 = np.where(self.esa_cci_dat.variables['lon'][:] == self.lons[-1])[0][0]
        self.cci_sm = np.array(self.esa_cci_dat.variables['sm'][self.strt_idx:self.end_idx+1,
                               self.lat_idx1:self.lat_idx2+1, self.lon_idx1:self.lon_idx2+1])  # get soil moisture obs
        self.cci_sm[self.cci_sm < 0.] = np.nan
        self.cci_sm_err = np.array(self.esa_cci_dat.variables['sm_uncertainty'][self.strt_idx:self.end_idx+1,
                               self.lat_idx1:self.lat_idx2+1, self.lon_idx1:self.lon_idx2+1])  # get soil mositure uncertainty
        self.cci_sm_err[self.cci_sm_err < 0.] = np.nan
        # Find soil parameters for lat lon grid
        self.soil_dat = nc.Dataset('soil.regional.nc', 'r')
        self.b = self.soil_dat.variables['field1381'][self.lat_idx1:self.lat_idx2+1, self.lon_idx1:self.lon_idx2+1]
        self.sathh = self.soil_dat.variables['field342'][self.lat_idx1:self.lat_idx2+1, self.lon_idx1:self.lon_idx2+1]
        self.satcon = self.soil_dat.variables['field333'][self.lat_idx1:self.lat_idx2+1, self.lon_idx1:self.lon_idx2+1]
        self.sm_sat = self.soil_dat.variables['field332'][self.lat_idx1:self.lat_idx2+1, self.lon_idx1:self.lon_idx2+1]
        self.sm_crit = self.soil_dat.variables['field330'][self.lat_idx1:self.lat_idx2+1, self.lon_idx1:self.lon_idx2+1]
        self.sm_wilt = self.soil_dat.variables['field329'][self.lat_idx1:self.lat_idx2+1, self.lon_idx1:self.lon_idx2+1]
        self.hcap = self.soil_dat.variables['field335'][self.lat_idx1:self.lat_idx2+1, self.lon_idx1:self.lon_idx2+1]
        self.hcon = self.soil_dat.variables['field336'][self.lat_idx1:self.lat_idx2+1, self.lon_idx1:self.lon_idx2+1]
        self.albsoil = self.soil_dat.variables['field1395'][self.lat_idx1:self.lat_idx2+1,
                                                            self.lon_idx1:self.lon_idx2+1]
        # Plotting setup
        sns.set_context('poster', font_scale=1., rc={'lines.linewidth': 2., 'lines.markersize': 3})
        sns.set_style('whitegrid')
        self.palette = sns.color_palette("colorblind", 11)

    def da_point(self, lat_idx, lon_idx):
        """
        Plots a da run for selected point
        :param lat_idx: lat idx on map
        :param lon_idx: lon idx on map
        :return: fig and axis for plot
        """
        fig, ax = plt.subplots(nrows=1, ncols=1)

        ax.plot(self.times, self.prior_sm[:, lat_idx, lon_idx], label='prior', color=self.palette[3])
        ax.plot(self.times, self.posterior_sm[:, lat_idx, lon_idx], label='posterior', color=self.palette[0])
        ax.plot(self.times, self.cci_sm[:, lat_idx, lon_idx], label='CCI', color=self.palette[2])
        # ax[0].set_title('Soil moisture comparison for JULES and ESA CCI at (' + lat + ', ' + lon + ')')
        ax.legend(loc=2)

        ax.set_xticks(self.times, minor=True)
        ax.set_ylabel(r'(m$^3$ m$^{-3}$)')
        myFmt = mdates.DateFormatter('%Y')
        ax.xaxis.set_major_formatter(myFmt)
        fig.subplots_adjust(hspace=0.3)
        return fig


if __name__ == "__main__":
    pj = PlotJules('output_da/prior.outvars.nc', 'output_da/posterior.outvars.nc')
    fig = pj.da_point(2, 7)
    fig.savefig('test.png')