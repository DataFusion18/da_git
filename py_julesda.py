import numpy as np
import netCDF4 as nc
import datetime as dt
import scipy.optimize as spop
import multiprocessing as mp
import itertools as itt
import py_jules as pyj
import os
import shutil as sh
import glob
# Rewrite this as a class?


class Jules_DA:
    def __init__(self, strt_yr=2009, end_yr=2009, lat=10.75, lon=0.25):
        self.lat = lat
        self.lon = lon
        self.n = 0  # iteration
        self.jules_class = pyj.jules()  # python jules wrapper class
        # extract cci obs
        self.esa_cci_dat = nc.Dataset('/export/cloud/nceo/users/if910917/esa_cci_v03/ghana/esacci_sm_1989_2014_'
                                      'regrid.nc', 'r')  # CCI sm observations
        self.strt_d = dt.datetime(strt_yr, 1, 2, 0, 0)
        self.strt_idx = nc.date2index(self.strt_d, self.esa_cci_dat.variables['time'])
        self.end_d = dt.datetime(end_yr, 12, 31, 0, 0)
        self.end_idx = nc.date2index(self.end_d, self.esa_cci_dat.variables['time'])
        self.lat_idx = np.where(self.esa_cci_dat.variables['lat'][:] == lat)[0][0]
        self.lon_idx = np.where(self.esa_cci_dat.variables['lon'][:] == lon)[0][0]
        self.cci_sm = np.array(self.esa_cci_dat.variables['sm'][self.strt_idx:self.end_idx+1, self.lat_idx,
                               self.lon_idx])  # get soil moisture obs
        self.cci_sm[self.cci_sm < 0.] = np.nan
        self.cci_sm_err = np.array(self.esa_cci_dat.variables['sm_uncertainty'][self.strt_idx:self.end_idx+1,
                                   self.lat_idx, self.lon_idx])  # get soil mositure uncertainty
        self.cci_sm_err[self.cci_sm_err < 0.] = np.nan
        # Find soil parameters for lat lon and set NML model grid file
        self.latlon_dat = nc.Dataset('../lonlat.regional.nc', 'r')
        self.soil_dat = nc.Dataset('../soil.regional_orig.nc', 'r')
        self.jules_class.model_grid_nml.mapping["jules_model_grid_1_lat_bounds"] = str(lat-0.25)+','+str(lat+0.25)+','
        self.jules_class.model_grid_nml.mapping["jules_model_grid_1_lon_bounds"] = str(lon-0.25)+','+str(lon+0.25)+','
        self.lat_idx = np.where(self.latlon_dat.variables['latitude'][:,0] == lat)[0][0]
        self.lon_idx = np.where(self.latlon_dat.variables['longitude'][0] == lon)[0][0]
        self.b = self.soil_dat.variables['field1381'][self.lat_idx, self.lon_idx]
        self.sathh = self.soil_dat.variables['field342'][self.lat_idx, self.lon_idx]
        self.satcon = self.soil_dat.variables['field333'][self.lat_idx, self.lon_idx]
        self.sm_sat = self.soil_dat.variables['field332'][self.lat_idx, self.lon_idx]
        self.sm_crit = self.soil_dat.variables['field330'][self.lat_idx, self.lon_idx]
        self.sm_wilt = self.soil_dat.variables['field329'][self.lat_idx, self.lon_idx]
        self.hcap = self.soil_dat.variables['field335'][self.lat_idx, self.lon_idx]
        self.hcon = self.soil_dat.variables['field336'][self.lat_idx, self.lon_idx]
        self.albsoil = self.soil_dat.variables['field1395'][self.lat_idx, self.lon_idx]
        self.soil_update = nc.Dataset('../soil.regional.nc', 'a')
        self.xb = np.array([self.b, self.sm_crit, self.sm_wilt])  # inital guess to 3 optimised parameters
        self.prior_err = 0.4*self.xb
        # Set output dirctory
        self.output_dir = "output2/"
        self.steps = []

    def run_jules(self, run_id='gh'):
        """
        Runs JULES changing soil parameters
        :param run_id: id of run as a string
        :return: location of JULES output as string
        """
        self.jules_class.output_nml.mapping["JULES_OUTPUT_1_run_id"] = "'" + run_id + "',"
        # print j.output_nml.mapping["JULES_OUTPUT_1_run_id"]
        self.jules_class.ancillaries_nml.mapping["jules_soil_props_1_const_val"] = \
            str(self.b) + ", " + str(self.sathh) + ", " + str(self.satcon) + ", " + str(self.sm_sat) + \
            ", " + str(self.sm_crit) + ", " + str(self.sm_wilt) + ", " + str(self.hcap) + \
            ", " + str(self.hcon) + ", " + str(self.albsoil) + ","
        # print self.jules_class.ancillaries_nml.mapping["jules_soil_props_1_const_val"]
        self.jules_class.runJules()
        print self.n
        return self.output_dir + "/" + run_id + '.outvars.nc'

    def obs_cost(self, jules_nc):
        """
        Calculates observation cost function between jules and cci obs
        :param jules_nc: files location of JULES netcdf output as string
        :return: cost function value for supplied model output
        """
        jules_dat = nc.Dataset(jules_nc, 'r')
        jules_sm = jules_dat.variables['smcl'][:, 0, 0, 0] / 100.
        obs = self.cci_sm[np.logical_not(np.isnan(self.cci_sm))]
        obs_err = self.cci_sm_err[np.logical_not(np.isnan(self.cci_sm_err))]
        mod = jules_sm[np.logical_not(np.isnan(self.cci_sm))]
        innov = [(obs[i] - mod[i]) ** 2 / obs_err[i] for i in xrange(len(obs))]
        obs_cost = np.sum(innov)
        return obs_cost

    def cost_b(self, b):
        """
        Calculates the whole cost function when varying the b soil parameter
        :param b: soil parameter b as a list
        :return: Cost function value
        """
        self.b = b[0]
        self.n += 1
        mod_cost = (b - 6.631272)**2 / (0.6 * 6.6313)
        jules_nc = self.run_jules(run_id="iter")
        obs_cost = self.obs_cost(jules_nc)
        ret_val = mod_cost + obs_cost
        self.steps.append([self.n, b[0], ret_val[0]])
        return ret_val

    def minimize_b(self, b):
        res = spop.minimize(self.cost_b, b, method='nelder-mead', options={'xtol': 1e-1, 'disp': True})
        return res

    def cost_b_smcrit_smwilt(self, x0):
        """
        Calculates the whole cost function when varying the b soil parameter
        :param x0: vector of soil parameter values as a list
        :return: Cost function value
        """
        self.b = x0[0]
        self.sm_crit = x0[1]
        self.sm_wilt = x0[2]
        self.n += 1
        innov = [(self.xb[i] - x0[i]) ** 2 / self.prior_err[i] for i in xrange(len(x0))]
        mod_cost = np.sum(innov)
        jules_nc = self.run_jules(run_id="iter")
        obs_cost = self.obs_cost(jules_nc)
        ret_val = mod_cost + obs_cost
        self.steps.append([self.n, x0, ret_val])
        return ret_val

    def minimize(self, x0):
        res = spop.minimize(self.cost_b_smcrit_smwilt, x0, method='nelder-mead', options={'xtol': 1e-1, 'disp': True})
        return res

    def da_run(self):
        self.jules_class.output_nml.mapping["JULES_OUTPUT_1_output_dir"] = "'"+self.output_dir+"',"
        #self.output_dir = out_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        res = self.minimize(self.xb)
        output = open('da_out_'+str(self.lat)+'_'+str(self.lon)+'.csv', 'w')
        for item in self.steps:
            output.write(str(item).strip("[]") + "\n")
        output.close()
        print res
        self.soil_update.variables['field1381'][self.lat_idx, self.lon_idx] = res.x[0]
        self.soil_update.variables['field330'][self.lat_idx, self.lon_idx] = res.x[1]
        self.soil_update.variables['field329'][self.lat_idx, self.lon_idx] = res.x[2]
        self.soil_update.close()


def spatial_run(lat_lon):
    """
    Runs JULES for specified lat lon
    :param lat_lon: tuple containing latitude and longitude coordinate
    :return: na
    """
    out_dir = 'output_' + str(lat_lon[0]) + '_' + str(lat_lon[1]) + '/'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for file in glob.glob(r'*.nml'):
        sh.copy(file, out_dir)
    os.chdir(out_dir)
    print lat_lon
    jcda = Jules_DA(lat=lat_lon[0], lon=lat_lon[1])
    jcda.output_dir = out_dir
    jcda.da_run()
    os.chdir('../')


if __name__ == "__main__":
    # spatial_run((10.75, 0.25))
    mp.freeze_support()
    lats = np.array([9.75, 10.25, 10.75, 11.25, 11.75])
    lons = np.array([-3.25, -2.75, -2.25, -1.75, -1.25, -0.75, -0.25, 0.25, 0.75, 1.25])
    pool = mp.Pool(processes=10)
    res = pool.map(spatial_run, itt.product(lats,lons))
    pool.close()
    pool.join()
    # Figure out how to run forestcast after this!






    """
    jcda = Jules_DA()
    jcda.jules_class.output_nml.mapping["JULES_OUTPUT_1_output_dir"] = "'min_b_smcrit_smwilt',"
    jcda.output_dir = "min_b_smcrit_smwilt/"
    if not os.path.exists("min_b_smcrit_smwilt"):
        os.makedirs("min_b_smcrit_smwilt")
    res = jcda.minimize(jcda.xb)
    output = open('da_out_b_smcrit_smwilt.csv', 'w')
    for item in jcda.steps:
        output.write(str(item).strip("[]")+"\n")
    output.close()
    print res
    jcda.jules_class.timesteps_nml.mapping["jules_time_1_main_run_end"] = "'2014-12-31 21:00:00',"
    jcda.run_jules(run_id='prior')
    jcda.run_jules(b=res.x[0], sm_crit=res.x[1], sm_wilt=res.x[2], run_id='posterior')
    """