#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File:                ampel/contrib/hu/t2/T2RunPossis.py
# License:             BSD-3-Clause
# Author:              jnordin@physik.hu-berlin.de
# Date:                11.12.2021
# Last Modified Date:  04.01.2022
# Last Modified By:    jnordin@physik.hu-berlin.de


import numpy as np
import sncosmo # type: ignore[import]
from sfdmap import SFDMap  # type: ignore[import]
import errno, os, backoff, copy
from astropy.time import Time
from typing import Literal, Sequence

from ampel.types import UBson
from ampel.struct.UnitResult import UnitResult
from ampel.contrib.hu.t2.T2RunSncosmo import T2RunSncosmo
from ampel.model.StateT2Dependency import StateT2Dependency
from ampel.view.T2DocView import T2DocView

from ampel.enum.DocumentCode import DocumentCode
from ampel.view.LightCurve import LightCurve


class T2RunPossis(T2RunSncosmo):
    """

    Load one of the POSSIS models and create an sncosmo_model
    model for fit by T2RunSncosmo.

    """

    # Parameters determining which POSSIS model will be read

    # Currently references to sample model included in Ampel-HU-astro
    possis_dir: str = 'data/kilonova_models'
    model_gen: str = 'bns_m3_3comp'
    mej_dyn: float = 0.01
    mej_wind: float = 0.09
    phi: int = 45
    cos_theta: float = 0.3    # Typically 0., 0.1, ...1.0

    sncosmo_model_name: str = '_'.join(map(str, [model_gen, mej_dyn,
        mej_wind, phi, cos_theta]) )

    # Fix time to specific explosion timestamp
    # StockTriggerTime assumes the value is updated during runtime
    explosion_time_jd: None | float | Literal['StockTriggerTime']

    # Which units should this be changed to
    t2_dependency: Sequence[StateT2Dependency[Literal[ # type: ignore[assignment]
                "T2DigestRedshifts",
                "T2MatchBTS",
                "T2PhaseLimit",
                "T2PropagateStockInfo"
            ]]]

    def post_init(self)-> None:
        """
        Retrieve POSSIS model.

        Note that this could be done once at instance init.
        """

        # Find file
        fname = os.path.join(self.possis_dir, self.model_gen,
                     "nph1.0e+06_mejdyn{:05.3f}_mejwind{:05.3f}_phi{}.txt".format(self.mej_dyn, self.mej_wind, self.phi))
        if not os.path.exists(fname):
            raise FileNotFoundError(fname)

        # Read model Parameters from first three lines
        with open(fname, 'r') as fh:
            lines = fh.readlines()
            nobs = int(lines[0])
            nwave = int(lines[1])
            line3 = lines[2].split(' ')
            ntime = int(line3[0])
            t_i = float(line3[1])
            t_f = float(line3[2])
            model_cos_theta = np.linspace(0, 1, nobs)  # 11 viewing angles
            phase = np.linspace(t_i, t_f, ntime)  # epochs

        # Limit to one angle
        # Note: U. Feindt developed model where angle was fit, left out for know
        theta_mask = np.isclose(self.cos_theta, model_cos_theta)
        if not sum(theta_mask)==1:
            raise ValueError("Model cos_theta {model_cos_theta} not defined")

        # Read model data
        mdata = np.genfromtxt(fname, skip_header=3)
        wave = mdata[0 : int(nwave), 0] # noqa
        flux = np.array([mdata[i * int(nwave) : i * int(nwave) + int(nwave), 1:] for i in range(int(nobs))]).T

        # Reduce to one angle
        flux_1angle = flux[:,:,theta_mask].squeeze()
        # Create model
        source = sncosmo.TimeSeriesSource(phase, wave, flux_1angle, name=self.sncosmo_model_name)

        # Setup model, with or without MW correction
        if self.apply_mwcorrection:
            dust = sncosmo.models.CCM89Dust()
            self.sncosmo_model = sncosmo.Model(
                source=source, effects=[dust], effect_names=["mw"], effect_frames=["obs"]
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

        # If explosion time should be fixed, do so
        # If explosion time should be fixed, do so
        if isinstance(self.explosion_time_jd, float):
            self.sncosmo_model.set(t0=self.explosion_time_jd)
            self.fit_params.remove("t0")

        self.default_param_vals = self.sncosmo_model.parameters

        # retry on with exponential backoff on "too many open files"
        self.process = backoff.on_exception( # type: ignore[assignment]
            backoff.expo,
            OSError,
            giveup=lambda exc: exc.errno != errno.EMFILE,
            logger=self.logger,
            max_time=300,
        )(self.process)

    def process(self,
        light_curve: LightCurve, t2_views: Sequence[T2DocView]
    ) -> UBson |UnitResult:
        """
        Adding the option to dynamically grap explosion time from T2PropagateStockInfo

        After setting this it will return to normal T2RunSncosmo.
        """

        if self.explosion_time_jd=='StockTriggerTime':
            for t2_view in t2_views:
                if not t2_view.unit == "T2PropagateStockInfo":
                    continue
                self.logger.debug('Parsing t2 results from {}'.format(t2_view.unit))
                t2_res = res[-1] if isinstance(res := t2_view.get_payload(), list) else res
                if not 'explosion_time' in t2_res.keys():
                    self.logger.info('No explosion time',extra={'t2res':t2_res})
                    return UnitResult(code=DocumentCode.T2_MISSING_INFO)

                if isinstance(t2_res['explosion_time'], float):
                    self.explosion_time_jd = t2_res['explosion_time']
                elif isinstance(t2_res['explosion_time'], str):
                    # Datetime format
                    self.explosion_time_jd = Time(t2_res['explosion_time'], scale="utc").jd
                # Reset model
                self.logger.debug('reset explosion time', extra={'explosion_time': self.explosion_time_jd})
                self.sncosmo_model.set(t0=self.explosion_time_jd)
                self.fit_params.remove("t0")

        # Restart sncosmo processing
        return super().process(light_curve, t2_views)
