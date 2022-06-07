#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File:                ampel/contrib/hu/t2/T2RunSnoopy.py
# License:             BSD-3-Clause
# Author:              jnordin@physik.hu-berlin.de
# Date:                01.06.2022
# Last Modified Date:  01.06.2021
# Last Modified By:    jnordin@physik.hu-berlin.de


from typing import Any, Optional, Union, Literal
from collections.abc import Sequence
import errno, backoff, copy, os


import numpy as np
import snpy # type: ignore[import]
from astropy.table import Table
from sfdmap import SFDMap  # type: ignore[import]

from ampel.types import UBson
from ampel.struct.UnitResult import UnitResult
from ampel.abstract.AbsTiedLightCurveT2Unit import AbsTiedLightCurveT2Unit
from ampel.view.T2DocView import T2DocView
from ampel.view.LightCurve import LightCurve
from ampel.ztf.util.ZTFIdMapper import ZTFIdMapper
from ampel.model.StateT2Dependency import StateT2Dependency
from ampel.plot.utils import mplfig_to_svg_dict1
from ampel.model.PlotProperties import PlotProperties


class T2RunSnoopy(AbsTiedLightCurveT2Unit):
    """
    Gathers information and runs snoopy. Steps include:
    - Parse previous (chained) T2results for redshift or phase limits.
    - Converts lightcurve content to Snoopy lc.
    - Plot output if requested

    Fits lightcurves using Snoopy
    which is assumed to be chained to other T2units that provide redshift and
    fit limits.

    TODO:
    - Add filters during post_init such that pip version can be used.

    """

    # Which snoopy model to use.
    # Default include EBV_model, EBV_model2, color_model, max_model
    snoopy_model_name: str

    # Redshift usage options. Current options
    # T2MatchBTS : Use the redshift published by BTS and  synced by that T2.
    # T2DigestRedshifts : Use the best redshift as parsed by DigestRedshift.
    # None : use backup_z as fixed redshift
    redshift_kind: Optional[str]
    # If loading redshifts from DigestRedshifts, provide the max ampel z group to make use of.
    # (note that filtering based on this can also be done for a potential t3)
    max_ampelz_group: int = 3
    # It is also possible to use fiPITY3LqGTmGWjt0-yviUGhIgUF1Xdr1O4TvYk1OSEqgxed redshift whenever a dynamic redshift kind is not available
    backup_z: Optional[float]
    # Finally, the provided lens redshift might be multiplied with a scale
    # Useful for lensing studies, or when trying multiple values
    scale_z: Optional[float]


    # TODO: Additional snoopy parameters and Bounds
    # Snoopy can also be run in emcee fit mode, where priors can be included.

    # Remove MW dust absorption using SFD maps.
    # If set, needs to have access to local maps through SFDMap
    # If not, will be requested through URSA (slow, requires connection)
    # The default value of Rv will be used.
    apply_mwcorrection: bool = False

    # Phase range usage. Current option:
    # T2PhaseLimit : use the jdmin jdmax provided in this unit output
    # None : use full datapoint range
    # (T2BayesianBlocks should be added)
    phaseselect_kind: Optional[str]

    # Plot behaviour. Default is to not produce any plots.
    # If set, will store png to this directory.
    plot_dir: Optional[str]
    # If true, will "draw" matplotlib figure to current env.
    plot_draw: bool = False

    # The snoopy filter naming scheme is not consistent with the observatories.
    # This is used to map from ampel fid to snoopy filter name
    filter_name_map: dict = {"1":'ztf_g', "2":'ztf_r', "3":'ztf_i'}

    # Which units should this be changed to
    t2_dependency: Sequence[StateT2Dependency[Literal[
                            "T2DigestRedshifts", "T2MatchBTS", "T2PhaseLimit"]]]


    def post_init(self) -> None:
        """
        Retrieve models.
        """


        # Setup model, with or without MW correction
        if self.apply_mwcorrection:
            self.dustmap = SFDMap()


        # retry on with exponential backoff on "too many open files"
        self.process = backoff.on_exception( # type: ignore[assignment]
            backoff.expo,
            OSError,
            giveup=lambda exc: exc.errno != errno.EMFILE,
            logger=self.logger,
            max_time=300,
        )(self.process)


    def _get_redshift(self, t2_views) -> tuple[Optional[float],Optional[str]]:
        """
        Can potentially also be replaced with some sort of T2DigestRershift tabulator?

        Assuming that only one instance of redshift sources exist
        """

        # Examine T2s for eventual information
        z: Optional[float] = None
        z_source: Optional[str] = None


        if self.redshift_kind in ['T2MatchBTS', 'T2DigestRedshifts']:
            for t2_view in t2_views:
                if not t2_view.unit == self.redshift_kind:
                    continue
                self.logger.debug('Parsing t2 results from {}'.format(t2_view.unit))
                t2_res = res[-1] if isinstance(res := t2_view.get_payload(), list) else res
                # Parse this
                if self.redshift_kind == 'T2MatchBTS':
                    if 'bts_redshift' in t2_res.keys() and not t2_res['bts_redshift'] == '-':
                        z = float(t2_res['bts_redshift'])
                        z_source = "BTS"
                elif self.redshift_kind == 'T2DigestRedshifts':
                    if ('ampel_z' in t2_res.keys() and t2_res['ampel_z'] is not None
                            and t2_res['group_z_nbr'] <= self.max_ampelz_group):
                        z = float(t2_res['ampel_z'])
                        z_source = "AMPELz_group" + str(t2_res['group_z_nbr'])
        else:
            # Check if there is a fixed z set for this run, otherwise keep as free parameter
            if self.backup_z:
                z = self.backup_z
                z_source = "Fixed"
            else:
                z = None
                z_source = "Fitted"

        if z and self.scale_z:
            z *= self.scale_z
            z_source += " + scaled {}".format(self.scale_z)

        return z, z_source


    def _get_phaselimit(self, t2_views) -> tuple[Optional[float],Optional[float]]:
        """
        Can potentially also be replaced with some sort of tabulator?

        """

        # Examine T2s for eventual information
        jdstart: Optional[float] = None
        jdend: Optional[float] = None

        if self.phaseselect_kind is None:
            jdstart = -np.inf
            jdend = np.inf
        else:

            for t2_view in t2_views:
                # So far only knows how to parse phases from T2PhaseLimit
                if not t2_view.unit == 'T2PhaseLimit':
                    continue
                self.logger.debug('Parsing t2 results from {}'.format(t2_view.unit))
                t2_res = res[-1] if isinstance(res := t2_view.get_payload(), list) else res
                jdstart = t2_res['t_start']
                jdend = t2_res['t_end']

        return jdstart, jdend


    def _get_snoopy_lcs(self, snpy_sn, light_curve, jdstart, jdend) -> dict:
        """
        Convert the light_curve instance to a dict with snoopy lc's:
        return {'filter_name': lc}

        The filter map defined above will be used to conver between
        ampel "fid" and snoopy filter name.

        This step is to be replaced by a tabulator.
        """

        # Limit phase range
        jd_filter = [
            {'attribute': 'jd', 'operator': '>=', 'value': jdstart},
            {'attribute': 'jd', 'operator': '<=', 'value': jdend}
        ]

        # Check each filter for match
        lcs = {}
        for f_ampel, f_snoopy in self.filter_name_map.items():
            dp_filter = [{'attribute': 'fid', 'operator':'==', 'value': int(f_ampel)}]
            dp_filter.extend(jd_filter)
            dps = light_curve.get_ntuples(('jd', 'magpsf', 'sigmapsf'),
                        filters=dp_filter)
            if dps is None or len(dps)==0:
                continue
            # Sort tuples based on jd
            dps = sorted(dps)
            df = np.asarray(dps)
            jd, mag, sigmamag = df[:,0], df[:,1], df[:,2]
            if len(jd)>0:
                lcs[f_snoopy] = snpy.lc(snpy_sn, f_snoopy, jd, mag, sigmamag)

        return lcs



    # ==================== #
    # AMPEL T2 MANDATORY   #
    # ==================== #
    def process(self,
                light_curve: LightCurve, t2_views: Sequence[T2DocView]
                ) -> Union[UBson, UnitResult]:
        """

        Fit the parameters of the specified snoopy model to the light_curve
        provided. Depending on the configuration, the provided T2DovViews
        are used to look for redshift information and any phase (time)
        limits for the fit.


        Parameters
        -----------
        light_curve: "ampel.view.LightCurve" instance.
        See the LightCurve docstring for more info.

        t2_records: List of T2Records from the following units (if available)
        T2DigestRedshifts (redshift parsed from catalogs)
        T2MatchBTS (redshifts synced from BTS page)
        T2PhaseLimit (fit time-limits as determined from lightcurve)

        Returns
        -------
        dict
        """

        # Initialize output dict
        t2_output: dict[str, UBson] = {"model_name": self.snoopy_model_name}

        # Obtain redshift
        z, z_source = self._get_redshift(t2_views)
        t2_output['z'] = z
        t2_output['z_source'] = z_source
        # A source class of None indicates that a redshift source was required, but not found.
        if t2_output['z_source'] is None:
            return t2_output

        # Check for phase limits
        (jdstart, jdend) = self._get_phaselimit(t2_views)
        t2_output['jdstart'] = jdstart
        t2_output['jdend'] = jdend
        if t2_output['jdstart'] is None:
            return t2_output

        # Initialize SN
        (ra, dec) = light_curve.get_pos()
        if self.apply_mwcorrection:
            transient_mwebv = self.dustmap.ebv(ra, dec)
            snoopy_sn = snpy.sn(name=ZTFIdMapper.to_ext_id(light_curve.stock_id),
                        z=z, ra=ra, dec=dec, EBVgal=transient_mwebv)
        else:
            snoopy_sn = snpy.sn(name=ZTFIdMapper.to_ext_id(light_curve.stock_id),
                        z=z, ra=ra, dec=dec)

        snoopy_sn.choose_model(self.snoopy_model_name)
        snoopy_sn.replot = 0 # default plot is off

        # Obtain and add photometric table
        lc_dict = self._get_snoopy_lcs(snoopy_sn, light_curve, t2_output['jdstart'], t2_output['jdend'])
        for fname, lc in lc_dict.items():
            snoopy_sn.data[fname] = lc
        snoopy_sn.get_restbands()

        # Do the fit
        try:
            snoopy_sn.fit()
        except RuntimeError as err:
            t2_output['RuntimeError']: err
            return t2_output

        # Collect output
        t2_output['parameters'] = snoopy_sn.parameters
        t2_output['errors'] = snoopy_sn.errors



        # Save plot
        if self.plot_dir:
            # Construct name
            fname = os.path.join(self.plot_dir,'{}_snoopy_{}.png'.format(
                    ZTFIdMapper.to_ext_id(light_curve.stock_id), self.snoopy_model_name))
            p = snoopy_sn.plot(outfile=fname)
        if self.plot_draw:
            p = snoopy_sn.plot()



        return t2_output
