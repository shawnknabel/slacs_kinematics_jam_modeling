import numpy as np
import os
from tqdm import tqdm_notebook, tnrange
import joblib
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from scipy.interpolate import interp1d

from lenstronomy.Sampling.parameters import Param
from lenstronomy.Analysis.kinematics_api import KinematicsAPI
import lenstronomy.Util.class_creator as class_creator
from lenstronomy.Analysis.td_cosmography import TDCosmography
from lenstronomy.GalKin.galkin import Galkin
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from lenstronomy.Util.param_util import phi_q2_ellipticity
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.Analysis.light_profile import LightProfileAnalysis
from lenstronomy.Util.param_util import ellipticity2phi_q
import lenstronomy.Util.multi_gauss_expansion as mge
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.Analysis.lens_profile import LensProfileAnalysis
from mgefit.mge_fit_1d import mge_fit_1d
from jampy.jam_axi_proj import jam_axi_proj
from astropy.convolution import Gaussian2DKernel
from astropy.convolution import convolve
from copy import deepcopy

cwd = os.getcwd()
base_path, _ = os.path.split(cwd)


class ModelOutput(object):
    """
    Class to compute velocity dispersion in spherical symmetry for RXJ 1131.
    """
    # measured velocity dispersion from Buckley-Geer et al. (2020)
    # VEL_DIS = np.array([296.])
    # SIG_VEL_DIS = np.array([19.])
    PSF_FWHM = 0.7
    # MOFFAT_BETA = np.array([1.74])
    X_GRID, Y_GRID = np.meshgrid(
        np.arange(-3.0597, 3.1597, 0.1457), # x-axis points to negative RA
        np.arange(-3.0597, 3.1597, 0.1457),
    )
    PIXEL_SIZE = 0.1457
    X_CENTER = 21.5 #0.35490234894050443 # fitted from the KCWI cube # 21.5
    Y_CENTER = 23.5 #0.16776792706671506 # fitted from the KCWI cube # 23.5

    Z_L = 0.295 # deflector redshift from Agnello et al. (2018)
    Z_S = 0.657 # source redshift

    R_sersic_1 = lambda _: np.random.normal(2.49, 0.01) * 0.878 #np.sqrt(
                                        #np.random.normal(0.912, 0.004))
    n_sersic_1 = lambda _: np.random.normal(0.93, 0.03)
    I_light_1 = lambda _: np.random.normal(0.091, 0.001)
    q_light_1 = lambda _: 0.878 #np.random.normal(0.921, 0.004)
    phi_light_1 = lambda _: 90 - 121.6 # np.random.normal(90-121.6, 0.5)

    R_sersic_2 = lambda _: np.random.normal(0.362, 0.009) * 0.849 #np.sqrt(
                                        #np.random.normal(0.867, 0.002))
    n_sersic_2 = lambda _: np.random.normal(1.59, 0.03)
    I_light_2 = lambda _: np.random.normal(0.89, 0.03)
    q_light_2 = lambda _: 0.849 #np.random.normal(0.849, 0.004)

    R_EFF = 1.85

    def __init__(self, mcmc_chain_dir,
                 cosmo=None, cgd=False):
        """
        Load the model output file and load the posterior chain and other model
        speification objects.
        """
        self.samples_mcmc = np.loadtxt(mcmc_chain_dir)

        # random.shuffle()
        self.num_param_mcmc = len(self.samples_mcmc)

        self.r_eff = [] #self.get_r_eff()
        self.a_ani = []
        self.inclination = []

        # declare following variables to populate later
        self.model_velocity_dispersion = None

        self._cgd = cgd
        if self._cgd:
            self.kwargs_model = {
                'lens_model_list': ['PEMD'],
                'lens_light_model_list': ['SERSIC', 'SERSIC'],
            }
        else:
            self.kwargs_model ={
                'lens_model_list': ['PEMD'],
                'lens_light_model_list': ['HERNQUIST'],
            }

        # numerical options to perform the numerical integrals
        self.kwargs_galkin_numerics = {#'sampling_number': 1000,
                                       'interpol_grid_num': 1000,
                                       'log_integration': True,
                                       'max_integrate': 100,
                                       'min_integrate': 0.001}

        self.lens_cosmo = LensCosmo(self.Z_L, self.Z_S, cosmo=cosmo)

        self._kwargs_cosmo = {'d_d': self.lens_cosmo.dd,
                              'd_s': self.lens_cosmo.ds,
                              'd_ds': self.lens_cosmo.dds}

        self.td_cosmography = TDCosmography(z_lens=self.Z_L, z_source=self.Z_S,
                                            kwargs_model=self.kwargs_model)

        self._intrinsic_axis_ratio_distribution = self.get_intrinsic_axis_ratio_distribution()

    def get_num_samples(self):
        """
        Get the number of samples.
        :return:
        :rtype:
        """
        return len(self.samples_mcmc)

    def get_e_parameters(self):
        """

        """
        q1, phi = self.q_light_1(), self.phi_light_1()
        e11, e12 = phi_q2_ellipticity(phi, q1)

        q2 = self.q_light_2()
        e21, e22 = phi_q2_ellipticity(phi, q2)

        return e11, e12, e21, e22

    def get_double_sersic_kwargs(self, is_shperical=True):
        """

        """
        kwargs_lens_light = [
            {'amp': self.I_light_1(), 'R_sersic': self.R_sersic_1(),
             'n_sersic': self.n_sersic_1(),
             'center_x': self.X_CENTER, 'center_y': self.Y_CENTER},
            {'amp': self.I_light_2(), 'R_sersic': self.R_sersic_2(),
             'n_sersic': self.n_sersic_2(),
             'center_x': self.X_CENTER, 'center_y': self.Y_CENTER}
        ]

        if is_shperical:
            return kwargs_lens_light
        else:
            e11, e12, e21, e22 = self.get_e_parameters()

            kwargs_lens_light[0]['e1'] = e11
            kwargs_lens_light[0]['e2'] = e12

            kwargs_lens_light[1]['e1'] = e21
            kwargs_lens_light[1]['e2'] = e22

            return kwargs_lens_light

    def get_light_profile_mge(self, kwargs_light, e11, e12, e21, e22):
        """

        """
        light_model = LightModel(['SERSIC', 'SERSIC'])
        # x, y = np.meshgrid(np.arange(-5, 5, 0.01), np.arange(-5, 5, 0.01))
        # kwargs_light = self.get_double_sersic_kwargs(is_shperical=True)
        # model_image = light_model.surface_brightness(x, y, kwargs_light, )

        for i in range(2):
            kwargs_light[i]['center_x'] = 0
            kwargs_light[i]['center_y'] = 0

        rs_1 = np.geomspace(1e-2, 10 * kwargs_light[0]['R_sersic'], 200)
        rs_2 = np.geomspace(1e-2, 10 * kwargs_light[1]['R_sersic'], 200)

        flux_r_1 = light_model.surface_brightness(rs_1, 0 * rs_1, kwargs_light,
                                                  k=0)
        flux_r_2 = light_model.surface_brightness(rs_2, 0 * rs_2, kwargs_light,
                                                  k=1)

        mge_fit_1 = mge_fit_1d(rs_1, flux_r_1, ngauss=20, quiet=True)
        mge_fit_2 = mge_fit_1d(rs_2, flux_r_2, ngauss=20, quiet=True)

        mge_1 = (mge_fit_1.sol[0], mge_fit_1.sol[1])
        mge_2 = (mge_fit_2.sol[0], mge_fit_2.sol[1])

        sigma_lum = np.append(mge_1[1], mge_2[1])
        surf_lum = np.append(mge_1[0], mge_2[0])  # / 2*np.pi / sigma_lum**2

        _, q_1 = ellipticity2phi_q(e11,
                                   e12)  # kwargs_light[0]['e1'], kwargs_light[0]['e2'])
        _, q_2 = ellipticity2phi_q(e21,
                                   e22)  # kwargs_light[1]['e1'], kwargs_light[1]['e2'])

        qobs_lum = np.append(np.ones_like(mge_1[1]) * q_1,
                             np.ones_like(mge_2[1]) * q_2)

        return surf_lum, sigma_lum, qobs_lum

    def get_mass_mge(self, sample_index=0):
        """

        """
        lens_model = LensModel(['PEMD'])

        # 'q','$\theta_{E}$','$\gamma$','$\theta_{E,satellite}$','$\gamma_{ext}$','$\theta_{ext}$'

        q, theta_e, gamma = self.samples_mcmc[sample_index][:3]
        r_array = np.geomspace(1e-4, 1e2, 200) * theta_e

        kwargs_lens = [{'theta_E': theta_e, 'gamma': gamma, 'e1': 0., 'e2': 0.,
                        'center_x': 0., 'center_y': 0.}]

        mass_r = lens_model.kappa(r_array, r_array * 0, kwargs_lens)

        # amps, sigmas, _ = mge.mge_1d(r_array, mass_r, N=20)
        mass_mge = mge_fit_1d(r_array, mass_r, ngauss=20, quiet=True)
        amps, sigmas = mass_mge.sol[0], mass_mge.sol[1]

        # mge_fit = mge_fit_1d(r_array, mass_r, ngauss=20)
        # print(mge_fit)

        lens_cosmo = LensCosmo(z_lens=self.Z_L, z_source=self.Z_S)

        surf_pot = lens_cosmo.kappa2proj_mass(
            amps) / 1e12
        sigma_pot = sigmas
        qobs_pot = np.ones_like(sigmas) * q

        return surf_pot, sigma_pot, qobs_pot

    def get_bs(self, params, surf_lum, sigma_lum, model='Osipkov-Merritt'):
        """

        """
        if model == 'Osipkov-Merritt':
            betas = 1 / (1 + (params / sigma_lum) ** 2)
        elif model == 'generalized-OM':
            betas = params[1] / (1 + (params[0] / sigma_lum) ** 2)
        elif model == 'constant':
            betas = (1 - params**2) * np.ones_like(sigma_lum)
        elif model == 'step':
            divider = 1. # arcsec
            betas = (1 - params[0]**2) * np.ones_like(sigma_lum)
            # betas[sigma_lum <= divider] = 1 - params[0]**2
            betas[sigma_lum > divider] = 1 - params[1]**2
        else:
            betas = sigma_lum * 0. # isotropic
        # vs = np.zeros((len(sigma_lum), len(sigma_lum)))  # r, k

        # for i in range(len(sigma_lum)):
        #     vs[i] = surf_lum[i] / np.sqrt(2 * np.pi) / sigma_lum[i] * np.exp(
        #         -sigma_lum ** 2 / sigma_lum[i] ** 2 / 2.)
        #
        # ys = np.sum(vs, axis=0) / (1 - betas)
        # bs = np.linalg.solve(vs.T, ys)

        # import matplotlib.pyplot as plt
        #
        # plt.loglog(sigma_lum, betas, 'o', ls='none', )
        # plt.loglog(sigma_lum, 1 - (np.sum(vs, axis=0) / (vs.T @ bs)), 'o',
        #            ls='none', )

        return betas

    @staticmethod
    def transform_pix_coords(xs, ys, x_center, y_center, angle):
        """
        """
        xs_ = xs - x_center
        ys_ = ys - y_center
        # angle *= np.pi / 180.

        xs_rotated = xs_ * np.cos(angle) + ys_ * np.sin(angle)
        ys_rotated = -xs_ * np.sin(angle) + ys_ * np.cos(angle)

        return xs_rotated, ys_rotated

    def get_jam_grid(self, phi=0., supersampling_factor=1):
        """

        """
        # n_pix = self.X_GRID.shape[0] * oversampling_factor
        # pix_size = self.PIXEL_SIZE / oversampling_factor
        #
        # pix_coordinates = np.arange(n_pix) * pix_size + pix_size / 2.
        # x_grid, y_grid = np.meshgrid(pix_coordinates, pix_coordinates)
        #
        # x_center_pix, y_center_pix = (n_pix - 1) / 2, (n_pix - 1) / 2
        # x_center_coord = x_center_pix * pix_size + pix_size / 2.
        # y_center_coord = y_center_pix * pix_size + pix_size / 2.

        delta_x = (self.X_GRID[0, 1] - self.X_GRID[0, 0])
        delta_y = (self.Y_GRID[1, 0] - self.Y_GRID[0, 0])
        assert np.abs(delta_x) == np.abs(delta_y)

        x_start = self.X_GRID[0, 0] - delta_x / 2. * (1 -
                                                      1 / supersampling_factor)
        x_end = self.X_GRID[0, -1] + delta_x / 2. * (1 -
                                                     1 / supersampling_factor)
        y_start = self.Y_GRID[0, 0] - delta_y / 2. * (1 -
                                                      1 / supersampling_factor)
        y_end = self.Y_GRID[-1, 0] + delta_y / 2. * (1 -
                                                     1 / supersampling_factor)

        xs = np.arange(x_start, x_end + delta_x / (10 + supersampling_factor),
                       delta_x / supersampling_factor)
        ys = np.arange(y_start, y_end + delta_y / (10 + supersampling_factor),
                       delta_y / supersampling_factor)

        x_grid_supersampled, y_grid_supersmapled = np.meshgrid(xs, ys)

        # x_grid = -(x_grid - x_center_coord)
        # y_grid = (y_grid - y_center_coord)

        x_grid, y_grid = self.transform_pix_coords(x_grid_supersampled,
                                            y_grid_supersmapled,
                                            self.X_CENTER, self.Y_CENTER, phi
                                          )

        return x_grid.flatten(), y_grid.flatten(), \
               x_grid_supersampled.flatten(), y_grid_supersmapled.flatten()

    def get_light_model(self, kwargs_light, e11, e12, e21, e22, x_grid,
                        y_grid):
        """
        """
        light_model = LightModel(['SERSIC_ELLIPSE', 'SERSIC_ELLIPSE'])

        kwargs_light[0]['e1'] = e11
        kwargs_light[0]['e2'] = e12
        kwargs_light[1]['e1'] = e21
        kwargs_light[1]['e2'] = e22

        model_image = light_model.surface_brightness(x_grid, y_grid,
                                                     kwargs_light)

        return model_image

    def compute_jam_velocity_dispersion(self,
                                        a_ani_min=0.5,
                                        a_ani_max=5,
                                        start_index=0,
                                        num_compute=None,
                                        print_step=None,
                                        r_eff_uncertainty=0.02,
                                        analytic_kinematics=False,
                                        supersampling_factor=1,
                                        voronoi_bins=None,
                                        single_slit=False,
                                        do_convolve=True,
                                        anisotropy_model='Osipkov-Merritt',
                                        is_spherical=False
                                        ):
        """
        """
        num_samples = self.get_num_samples()

        self.model_velocity_dispersion = []

        sigma = self.PSF_FWHM / 2.355 / self.PIXEL_SIZE * supersampling_factor
        kernel = Gaussian2DKernel(x_stddev=sigma, x_size=4*int(sigma)+1,
                                  y_size=4*int(sigma)+1)

        for i in range(start_index, start_index+num_compute):
            if print_step is not None:
                if (i-start_index)%print_step == 0:
                    print('Computing step: {}'.format(i-start_index))

            r_eff_uncertainty_factor = np.random.normal(loc=1., scale=0.05)

            r_eff = 1.85 * r_eff_uncertainty_factor #self.get_r_eff(i)
            # Jeffrey's prior for a_ani
            if anisotropy_model == 'Osipkov-Merritt':
                ani_param = 10**np.random.uniform(np.log10(a_ani_min),
                                              np.log10(a_ani_max)) * r_eff
            elif anisotropy_model == 'generalized-OM':
                a_ani = 10 ** np.random.uniform(np.log10(a_ani_min),
                                                np.log10(a_ani_max)) * r_eff
                amp = np.random.uniform(0., 1.)
                ani_param = np.array([a_ani, amp])
            elif anisotropy_model == 'constant':
                ani_param = np.random.uniform(0.5, 1.)
                toss = np.random.uniform(0., 1.)
                if toss > 0.5:
                    ani_param = 1. / ani_param
            elif anisotropy_model == 'step':
                ani_param = np.random.uniform([0.5, 0.5], [1., 1.], size=2)
                for k in range(2):
                    toss = np.random.uniform(0, 1.)
                    if toss > 0.5:
                        ani_param[k] = 1. / ani_param[k]

            self.a_ani.append(ani_param)
            self.r_eff.append(r_eff)

            # theta_E = self.samples_mcmc[i][1]
            # gamma = self.samples_mcmc[i][2]

            kwargs_light = self.get_double_sersic_kwargs(is_shperical=True)
            e11, e12, e21, e22 = self.get_e_parameters()
            surf_lum, sigma_lum, qobs_lum = self.get_light_profile_mge(
                deepcopy(kwargs_light), e11, e12, e21, e22)
            surf_pot, sigma_pot, qobs_pot = self.get_mass_mge(sample_index=i)
            if is_spherical:
                qobs_lum = np.ones_like(qobs_lum)
                qobs_pot = np.ones_like(qobs_pot)
            bs = self.get_bs(ani_param, surf_lum, sigma_lum,
                             model=anisotropy_model
                             )

            #inc = 90
            mbh = 0
            distance = self.lens_cosmo.dd

            # phi = (113.4 + 90) / 180 * np.pi
            phi, _ = ellipticity2phi_q(e11, e12)
            q = np.mean(qobs_pot)

            #min_inclination = np.arccos(np.sqrt(q**2 - 1e-2**2)) * 180 / np.pi
            #inclination = np.random.uniform(min_inclination, 90.)
            random_axis_ratio = np.random.choice(
                self._intrinsic_axis_ratio_distribution[
                    self._intrinsic_axis_ratio_distribution <= q]
            )
            inclination = np.arccos(np.sqrt(q ** 2 - random_axis_ratio ** 2)) * 180 / np.pi
            if is_spherical:
                inclination = 90
            self.inclination.append(inclination)

            x_grid_spaxel, y_grid_spaxel, _, _ = self.get_jam_grid(phi,
                                                    supersampling_factor=1)

            x_grid, y_grid, x_grid_original, y_grid_original = self.get_jam_grid(
                                    phi,
                                    supersampling_factor=supersampling_factor)

            # print(x_grid.shape, y_grid.shape)
            if do_convolve:
                sigma_psf = 0.7 / 2.355
            else:
                sigma_psf = 0.
            norm_psf = 1.

            jam = jam_axi_proj(
                surf_lum / np.sqrt(2 * np.pi) / sigma_lum,
                sigma_lum, qobs_lum,
                surf_pot / np.sqrt(2 * np.pi) / sigma_pot,
                sigma_pot, qobs_pot,
                inclination, mbh, distance, x_grid_spaxel, y_grid_spaxel,
                plot=False, pixsize=0.1457, #self.PIXEL_SIZE/supersampling_factor,
                pixang=phi, quiet=1,
                sigmapsf=sigma_psf, normpsf=norm_psf,
                moment='zz',
                #goodbins=goodbins,
                align='sph',
                beta=bs,
                #data=rms, errors=erms,
                ml=1)

            num_pix = int(np.sqrt(len(x_grid)))
            num_pix_spaxel = int(np.sqrt(len(x_grid_spaxel)))
            vel_dis_model = jam.model.reshape((num_pix_spaxel, num_pix_spaxel))

            flux = self.get_light_model(kwargs_light, e11, e12, e21, e22,
                        x_grid_original, y_grid_original).reshape((num_pix,
                                                                   num_pix))

            if do_convolve:
                convolved_flux = convolve(flux, kernel)
                # convolved_map = convolve(flux * vel_dis_map ** 2, kernel)
                # convolved_flux_spaxels = convolved_flux.reshape(
                #         len(self.X_GRID), supersampling_factor,
                #         len(self.Y_GRID), supersampling_factor
                #     ).sum(3).sum(1)
                # convolved_map = convolved_flux_spaxels * vel_dis_model ** 2
            else:
                convolved_flux = flux

            convolved_flux_spaxel = convolved_flux.reshape(
                len(self.X_GRID), supersampling_factor,
                len(self.Y_GRID), supersampling_factor
            ).sum(3).sum(1)
            convolved_map = convolved_flux_spaxel * vel_dis_model**2


            if voronoi_bins is not None:
                # supersampled_voronoi_bins = voronoi_bins.repeat(
                #     supersampling_factor, axis=0).repeat(supersampling_factor,
                #                                          axis=1)

                n_bins = int(np.max(voronoi_bins)) + 1

                binned_map = np.zeros(n_bins)
                binned_IR = np.zeros(n_bins)
                for n in range(n_bins):
                    binned_map[n] = np.sum(
                        convolved_map[voronoi_bins == n]
                    )
                    binned_IR[n] = np.sum(
                        convolved_flux_spaxel[voronoi_bins == n]
                    )
                vel_dis_map = np.sqrt(binned_map / binned_IR)
            else:
                # binned_map = convolved_map.reshape(
                #     len(self.X_GRID), supersampling_factor,
                #     len(self.Y_GRID), supersampling_factor
                # ).sum(3).sum(1)
                #
                # IR_integrated = convolved_flux.reshape(
                #     len(self.X_GRID), supersampling_factor,
                #     len(self.Y_GRID), supersampling_factor
                # ).sum(3).sum(1)
                vel_dis_map = vel_dis_model

            self.model_velocity_dispersion.append(vel_dis_map)

        self.model_velocity_dispersion = np.array(
            self.model_velocity_dispersion)

        return self.model_velocity_dispersion

    def get_intrinsic_axis_ratio_distribution(self):
        """
        Return a distribution of b/a values sampled from the SDSS
        ellipticals of Padilla & Strauss (2008).
        """
        scrapped_points = np.array([
            0, 0,
            0.05, 0,
            0.116, 0,
            0.17, 0.049,
            0.223, 0.223,
            0.272, 0.467,
            0.322, 0.652,
            0.376, 0.745,
            0.426, 0.842,
            0.475, 0.995,
            0.525, 1.109,
            0.577, 1.217,
            0.626, 1.337,
            0.676, 1.484,
            0.725, 1.516,
            0.776, 1.576,
            0.826, 1.489,
            0.876, 1.342,
            0.928, 1.076,
            0.976, 0.755,
        ])

        x = scrapped_points[::2]
        y = scrapped_points[1::2]

        interp_func = interp1d(x, y, bounds_error=False, fill_value='extrapolate')

        sample = np.random.uniform(0, 1, 200000)
        sample_weighted = np.random.choice(sample, size=20000,
                                           p=interp_func(sample) / np.sum(
                                               interp_func(sample)))
        return sample_weighted

    def compute_model_velocity_dispersion(self,
                                          a_ani_min=0.5,
                                          a_ani_max=5,
                                          start_index=0,
                                          num_compute=None,
                                          print_step=None,
                                          r_eff_uncertainty=0.02,
                                          analytic_kinematics=False,
                                          supersampling_factor=1,
                                          voronoi_bins=None,
                                          single_slit=False,
                                          ):
        """
        Compute velocity dispersion from the lens model for different measurement setups.
        :param num_samples: default `None` to compute for all models in the
        chain, use lower number only for testing and keep it same between
        `compute_model_time_delays` and this method.
        :param start_index: compute velocity dispersion from this index
        :param num_compute: compute for this many samples
        :param print_step: print a notification after this many step
        """
        num_samples = self.get_num_samples()

        self.model_velocity_dispersion = []

        anisotropy_model = 'OM'  # anisotropy model applied
        aperture_type = 'IFU_grid'  # type of aperture used

        if num_compute is None:
            num_compute = num_samples - start_index

        if single_slit:
            kwargs_aperture = {'aperture_type': 'slit',
                               'length': 1.,
                               'width': 0.81,
                               'center_ra': 0.,
                               # lens_light_result[0]['center_x'],
                               'center_dec': 0.,
                               # lens_light_result[0]['center_y'],
                               'angle': 0
                               }

        else:
            kwargs_aperture = {'aperture_type': aperture_type,
                               'x_grid': self.X_GRID,
                               'y_grid': self.Y_GRID,
                               'center_ra': 0., #lens_light_result[0]['center_x'],
                               'center_dec': 0., #lens_light_result[0]['center_y'],
                               #'angle': 0
                               }

        if single_slit:
            kwargs_seeing = {'psf_type': 'GAUSSIAN',
                             'fwhm': 0.7,
                             # 'moffat_beta': self.MOFFAT_BETA[n]
                             }
        else:
            kwargs_seeing = {'psf_type': 'GAUSSIAN',
                             'fwhm': self.PSF_FWHM,
                             #'moffat_beta': self.MOFFAT_BETA[n]
                            }
        if self._cgd:
            light_model_bool = [True, True]
        else:
            light_model_bool = [True]
        lens_model_bool = [True]

        # galkin = Galkin(kwargs_model=self.kwargs_model,
        #                 kwargs_aperture=kwargs_aperture,
        #                 kwargs_psf=kwargs_seeing,
        #                 kwargs_cosmo=self._kwargs_cosmo,
        #                 kwargs_numerics=self.kwargs_galkin_numerics,
        #                 analytic_kinematics=analytic_kinematics)

        kinematics_api = KinematicsAPI(z_lens=self.Z_L, z_source=self.Z_S,
                                       kwargs_model=self.kwargs_model,
                                       kwargs_aperture=kwargs_aperture,
                                       kwargs_seeing=kwargs_seeing,
                                       anisotropy_model=anisotropy_model,
                                       cosmo=None,
                                       lens_model_kinematics_bool=lens_model_bool,
                                       light_model_kinematics_bool=light_model_bool,
                                       multi_observations=False,
                                       kwargs_numerics_galkin=self.kwargs_galkin_numerics,
                                       analytic_kinematics=(not self._cgd),
                                       Hernquist_approx=False,
                                       MGE_light=self._cgd,
                                       MGE_mass=False, #self._cgd,
                                       kwargs_mge_light=None,
                                       kwargs_mge_mass=None,
                                       sampling_number=1000,
                                       num_kin_sampling=2000,
                                       num_psf_sampling=500,
                                       )

        for i in range(start_index, start_index+num_compute):
            if print_step is not None:
                if (i-start_index)%print_step == 0:
                    print('Computing step: {}'.format(i-start_index))

            sample = self.samples_mcmc[i]

            #vel_dis_array = []

            r_eff_uncertainty_factor = np.random.normal(loc=1., scale=0.05)

            r_eff = self.R_EFF * r_eff_uncertainty_factor #self.get_r_eff(i)
            # Jeffrey's prior for a_ani
            a_ani = 10**np.random.uniform(np.log10(a_ani_min),
                                          np.log10(a_ani_max))

            self.a_ani.append(a_ani)
            self.r_eff.append(r_eff)
            theta_E = self.samples_mcmc[i][1]
            gamma = self.samples_mcmc[i][2]

            kwargs_lens = [{'theta_E': theta_E,
                            'gamma': gamma,
                            'e1': 0., 'e2': 0.,
                            'center_x': self.X_CENTER,
                            'center_y': self.Y_CENTER
                            }]

            if self._cgd:
                kwargs_lens_light = self.get_double_sersic_kwargs(
                    is_shperical=True)
            else:
                kwargs_lens_light = [{'amp': 1., 'Rs': r_eff / (1+np.sqrt(2)),
                                      'center_x': self.X_CENTER,
                                      'center_y': self.Y_CENTER
                                      }]

            # set the anisotropy radius. r_eff is pre-computed half-light
            # radius of the lens light
            kwargs_anisotropy = {'r_ani': a_ani * r_eff}

            # compute the velocity disperson in a pre-specified cosmology
            # (see lenstronomy function)
            if single_slit:
                vel_dis = kinematics_api.velocity_dispersion(
                    kwargs_lens,
                    kwargs_lens_light,
                    # kwargs_result['kwargs_lens_light'],
                    kwargs_anisotropy,
                    r_eff=(1 + np.sqrt(2)) * r_eff, theta_E=theta_E,
                    gamma=gamma,
                    kappa_ext=0,
                )
            else:
                vel_dis = kinematics_api.velocity_dispersion_map(
                    kwargs_lens,
                    kwargs_lens_light,
                    #kwargs_result['kwargs_lens_light'],
                    kwargs_anisotropy,
                    r_eff=(1+np.sqrt(2)) * r_eff, theta_E=theta_E,
                    gamma=gamma,
                    kappa_ext=0,
                    direct_convolve=True,
                    supersampling_factor=supersampling_factor,
                    voronoi_bins=voronoi_bins
                )

            self.model_velocity_dispersion.append(vel_dis)

        self.model_velocity_dispersion = np.array(
            self.model_velocity_dispersion)

        return self.model_velocity_dispersion

    def load_velocity_dispersion(self, dir_prefix, dir_suffix,
                                 compute_chunk, total_samples=None):
        """
        Load saved model velocity dispersions.

        :param dir_prefix:
        :type dir_prefix: should include the slash at the end
        :param dir_suffix: example '_mod_out.txt'
        :type dir_suffix:
        :param compute_chunk: number of samples in a computed chunk to
        combine output files with computed velocity dispersions
        :type compute_chunk: int
        :return:
        :rtype:
        """
        loaded_vel_dis = []
        if total_samples is None:
            total_samples = self.get_num_samples()

        for i in range(int(total_samples / compute_chunk)):
            start_index = i * compute_chunk
            file_path = dir_prefix + '{}_'.format(start_index) + dir_suffix

            if loaded_vel_dis == []:
                loaded_vel_dis = np.loadtxt(file_path)
            else:
                loaded_vel_dis = np.append(loaded_vel_dis,
                                           np.loadtxt(file_path),
                                           axis=0)
        print(len(loaded_vel_dis))
        assert len(loaded_vel_dis) == total_samples

        self.model_velocity_dispersion = loaded_vel_dis

    def load_a_ani(self, dir_prefix, dir_suffix,
                   compute_chunk, total_samples=None):
        """
        Load saved r_ani from files.

        :param dir_prefix:
        :type dir_prefix: should include the slash at the end
        :param dir_suffix: example '_mod_out.txt'
        :type dir_suffix:
        :param compute_chunk: number of samples in a computed chunk to
        combine output files with computed a_ani
        :type compute_chunk: int
        :return:
        :rtype:
        """
        loaded_a_ani = []

        if total_samples is None:
            total_samples = self.get_num_samples()

        for i in range(int(total_samples / compute_chunk)):
            start_index = i * compute_chunk
            file_path = dir_prefix + '{}_'.format(start_index) + dir_suffix

            if loaded_a_ani == []:
                loaded_a_ani = np.loadtxt(file_path)
            else:
                loaded_a_ani = np.append(loaded_a_ani, np.loadtxt(file_path),
                                         axis=0)

        assert len(loaded_a_ani) == total_samples

        self.a_ani = loaded_a_ani

    def load_inclination(self, dir_prefix, dir_suffix,
                   compute_chunk, total_samples=None):
        """
        Load saved r_ani from files.

        :param dir_prefix:
        :type dir_prefix: should include the slash at the end
        :param dir_suffix: example '_mod_out.txt'
        :type dir_suffix:
        :param compute_chunk: number of samples in a computed chunk to
        combine output files with computed a_ani
        :type compute_chunk: int
        :return:
        :rtype:
        """
        loaded_inclination = []

        if total_samples is None:
            total_samples = self.get_num_samples()

        for i in range(int(total_samples / compute_chunk)):
            start_index = i * compute_chunk
            file_path = dir_prefix + '{}_'.format(start_index) + dir_suffix

            if loaded_inclination == []:
                loaded_inclination = np.loadtxt(file_path)
            else:
                loaded_inclination = np.append(loaded_inclination,
                                               np.loadtxt(file_path),
                                               axis=0)

        assert len(loaded_inclination) == total_samples

        self.inclination = loaded_inclination

    def load_r_eff(self, dir_prefix, dir_suffix,
                   compute_chunk, total_samples=None):
        """
        Load saved r_eff from files.

        :param dir_prefix:
        :type dir_prefix: should include the slash at the end
        :param dir_suffix: example '_mod_out.txt'
        :type dir_suffix:
        :param compute_chunk: number of samples in a computed chunk to
        combine output files with R_eff
        :type compute_chunk: int
        :return:
        :rtype:
        """
        loaded_r_eff = []

        if total_samples is None:
            total_samples = self.get_num_samples()

        for i in range(int(total_samples / compute_chunk)):
            start_index = i * compute_chunk
            file_path = dir_prefix + '{}_'.format(start_index) + dir_suffix

            if loaded_r_eff == []:
                loaded_r_eff = np.loadtxt(file_path)
            else:
                loaded_r_eff = np.append(loaded_r_eff, np.loadtxt(file_path),
                                         axis=0)

        assert len(loaded_r_eff) == total_samples

        self.r_eff = loaded_r_eff

    def save_time_delays(self, model_name, dir_prefix, dir_suffix):
        """
        Save computed time delays.
        :param model_name:
        :type model_name:
        :param dir_prefix:
        :type dir_prefix:
        :param dir_suffix:
        :type dir_suffix:
        :return:
        :rtype:
        """
        file_path = dir_prefix + model_name + dir_suffix
        np.savetxt(file_path, self.model_time_delays)

    def load_time_delays(self, model_name, dir_prefix, dir_suffix):
        """
        Load saved time delays.
        :param model_name:
        :type model_name:
        :param dir_prefix:
        :type dir_prefix:
        :param dir_suffix:
        :type dir_suffix:
        :return:
        :rtype:
        """
        file_path = dir_prefix + model_name + dir_suffix

        loaded_time_delays = np.loadtxt(file_path)

        assert len(loaded_time_delays) == self.get_num_samples()

        self.model_time_delays = loaded_time_delays