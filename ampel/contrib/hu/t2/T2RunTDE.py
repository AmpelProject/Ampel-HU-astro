#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File:                ampel/contrib/hu/t2/T2RunTDE.py
# License:             BSD-3-Clause
# Author:              simeon.reusch@desy.de
# Date:                17.05.2022
# Last Modified Date:  18.05.2022
# Last Modified By:    simeon.reusch@desy.de


import numpy as np
import sncosmo  # type: ignore[import]
from sfdmap import SFDMap  # type: ignore[import]
from typing import Union
import errno, os, backoff, copy

# from astropy.time import Time
import astropy.cosmology as cospy
from typing import Literal, Sequence

# from urllib.request import urlopen
# from urllib.parse import urljoin

from ampel.types import UBson
from ampel.struct.UnitResult import UnitResult
from ampel.contrib.hu.t2.T2RunSncosmo import T2RunSncosmo

# from ampel.model.StateT2Dependency import StateT2Dependency
from ampel.view.T2DocView import T2DocView

# from ampel.enum.DocumentCode import DocumentCode
from ampel.view.LightCurve import LightCurve


class TDESource(sncosmo.Source):

    _param_names: list = ["risetime", "decaytime", "temperature", "amplitude"]
    param_names_latex: list = [
        "Rise Time",
        "Decay Time",
        "Temperature~[log10~K]",
        "Amplitude",
    ]

    def __init__(
        self,
        phase: np.ndarray,
        wave: np.ndarray,
        # add redshift as parameter
        name: str = "TDE",
        version: str = "1.0",
    ) -> None:

        self.name: str = name  # mandatory for SNCosmo
        self.version: str = version  # mandatory for
        self._phase: np.ndarray = phase
        self._wave: np.ndarray = wave

        self.nu_kc: float = 3e8 / 4770e-10  # Reference wavelength used for K-correction
        self.z: float = 0.1

        # Some constants
        self.sigma_SB: float = 5.6704e-5  # Stefan-Boltzmann constant (erg/s/cm^2/K^4)
        self.c: float = 2.9979e10  # Speed of light (cm/s)
        self.h: float = 6.6261e-27  # Planck's constant (erg s)
        self.k: float = 1.3806e-16  # Boltzman's constant (erg/K)
        self.h_over_k: float = self.h / self.k
        self.lumdis: float = float(
            cospy.FlatLambdaCDM(H0=70, Om0=0.3).luminosity_distance(self.z).value
            * 1e6
            * 3.08568e18
        )

        # Fit parameters
        # defaults - peaktime = 0, rise=0.85 / decay=1.5 / T=4.5 / peakflux = 32
        self._parameters: np.ndarray = np.array([0.85, 1.5, 4.5, 32])

    @staticmethod
    def _wl_to_nu(self, wl: np.ndarray) -> np.ndarray:
        """
        Convert wavelength in Angstrom to frequency in Hz

        """
        nu = 2.9979e18 / wl

        return nu

    @staticmethod
    def _Planck(self, nu: np.ndarray, T: float) -> Union[np.float64, np.ndarray]:
        """
        Calculate the spectral radiance of a blackbody
        :nu: np.ndarray, Array containing frequencies in Hz
        :T: np.float, temperature in K
        """
        bb = (
            2
            * self.h
            / self.c**2
            * nu**3
            / (np.exp(self.h * nu / (self.k * T)) - 1)
        )
        return bb

    @staticmethod
    def _cc_bol(self, nu: Union[float, np.ndarray], T: float):
        return self._Planck(self, nu, T) * nu / (self.sigma_SB * T**4 / np.pi)

    @staticmethod
    def _get_cc(self, nu: np.ndarray, T: Union[float, np.float64] = None) -> np.ndarray:
        """
        Calculate a k-correction for each wavelength
        """
        if T is None:
            T = self._parameters[2]
        T = 10**T
        kc = (
            (np.exp(self.h_over_k * self.nu_kc / T) - 1)
            / (np.exp(self.h_over_k * nu / T) - 1)
            * (nu / self.nu_kc) ** 4
        )
        return kc

    @staticmethod
    def _gauss(x: int, sigma: Union[float, np.float64]) -> Union[float, np.float64]:
        """
        Calculate a Gaussian
        """
        gauss = np.exp(-0.5 * x**2 / (sigma**2))
        return gauss

    @staticmethod
    def _gauss_exp(self, phase: np.float64, nu: np.ndarray) -> np.ndarray:
        """
        Calculate a gaussian rise and decay for the lightcurve
        """
        risetime = self._parameters[0]
        decaytime = self._parameters[1]
        temp = self._parameters[2]
        peakflux = self._parameters[3]

        a1 = (
            10**peakflux
        )  # luminosity/flux at peak (this can be bolometric or L(nu_kc) depending on the setting)
        b1 = 10**risetime  # rise rate
        b2 = 10**decaytime  # decay rate
        a2 = a1 * self._gauss(0, b1)

        # are we in the rise or decay phase?
        # risephase
        if phase <= 0:
            val = a1 * self._gauss(phase, b1)

        # decayphase
        else:
            val = a2 * np.exp(-(phase) / b2)

        cc = self._get_cc(self, nu)  # conversion from model curve to nuLnu of the data

        return val * cc

    @staticmethod
    def _lum2flux(
        L: np.ndarray,
        cm: float,
        nu: np.ndarray,
    ):
        """
        erg/s to Jansky
        >> flux = lum2flux(L, z, nu=1.4e9) # in Jy
        input:
         - L: luminosity in erg/s
         - cm: luminosity distance in cm

        note, no K-correction
        """
        return L / (nu * 4 * np.pi * cm**2) * 1e23

    def _flux(self, phase: np.ndarray, wave: np.ndarray) -> np.ndarray:
        """
        Calculate the model flux for given wavelengths and phases
        """
        temp = self._parameters[2]

        if np.ndim(phase) == 0:
            # phase_iter = [phase]
            phase_iter = np.asarray(phase)
        else:
            phase_iter = phase

        model_flux = np.empty((len(phase_iter), len(wave)))

        for i, ph in enumerate(phase_iter):

            nu = self._wl_to_nu(self, wave)
            model = self._gauss_exp(self, phase=ph, nu=nu)
            model_bol = model / self._cc_bol(self, T=10**temp, nu=self.nu_kc)
            cc = self._cc_bol(self, T=10**temp, nu=nu * (1 + self.z))

            model_flux_at_phase = self._lum2flux(
                L=(model_bol * cc), cm=self.lumdis, nu=nu * (1 + self.z)
            ) * (1 + self.z)

            model_flux[i] = model_flux_at_phase

        return model_flux


class T2RunTDE(T2RunSncosmo):
    """
    Create a TDE model and fit to a LightCurve object as process is called.

    Create a TDE sncosmo_model for fit by T2RunSncosmo.
    """

    sncosmo_model_name: str = "tde"

    def post_init(self) -> None:
        """
        Create TDE model.
        """

        phase = np.linspace(-50, 100, 20)
        wave = np.linspace(1000, 10000, 5)

        # initialize the TDE source
        tde_source = TDESource(phase, wave, name=self.sncosmo_model_name)

        # Setup model, with or without MW correction
        if self.apply_mwcorrection:
            dust = sncosmo.models.CCM89Dust()
            self.sncosmo_model = sncosmo.Model(
                source=tde_source,
                effects=[dust],
                effect_names=["mw"],
                effect_frames=["obs"],
            )
            self.dustmap = SFDMap()
            self.fit_params = copy.deepcopy(self.sncosmo_model.param_names)
            self.fit_params.remove("mwebv")
        else:
            self.sncosmo_model = sncosmo.Model(source=source)
            self.fit_params = copy.deepcopy(self.sncosmo_model.param_names)

        # If redshift _should_ be provided we remove this from fit parameters
        if self.redshift_kind is not None or self.backup_z is not None:
            self.fit_params.remove("z")

        self.default_param_vals = self.sncosmo_model.parameters

        # retry on with exponential backoff on "too many open files"
        self.process = backoff.on_exception(  # type: ignore[assignment]
            backoff.expo,
            OSError,
            giveup=lambda exc: exc.errno != errno.EMFILE,
            logger=self.logger,
            max_time=300,
        )(self.process)

    def process(
        self, light_curve: LightCurve, t2_views: Sequence[T2DocView]
    ) -> Union[UBson, UnitResult]:
        """
        Fit the loaded model to the data provided as a LightCurve.
        """

        # Restart sncosmo processing
        return super().process(light_curve, t2_views)
