#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File              : ampel/contrib/hu/t0/DecentFilter.py
# License           : BSD-3-Clause
# Author            : m. giomi <matteo.giomi@desy.de>
# Date              : 06.06.2018
# Last Modified Date: 31.10.2019
# Last Modified By  : vb <vbrinnel@physik.hu-berlin.de>

import logging
from typing import Any, Dict, Optional, Set
from numpy import exp, array, asarray
from astropy.coordinates import SkyCoord
from astropy.table import Table

from ampel.contrib.hu import catshtm_server
from ampel.base.AmpelAlert import AmpelAlert
from ampel.abstract.AbsT0AlertFilter import AbsT0AlertFilter


class DecentFilter(AbsT0AlertFilter):
	"""
	General-purpose filter with ~ 0.6% acceptance. It selects alerts based on:
	* numper of previous detections
	* positive subtraction flag
	* loose cuts on image quality (fwhm, elongation, number of bad pixels, and the
	  difference between PSF and aperture magnitude)
	* distance to known SS objects
	* real-bogus
	* detection of proper-motion and paralax for coincidence sources in GAIA DR2

	The filter has a very weak dependence on the real-bogus score and it is independent
	on the provided PS1 star-galaxy classification.
	"""

	# Static version info
	version = 1.0
	resources = ('catsHTM.default',)
	
	
	class InitConfig(AbsT0AlertFilter.InitConfig):
		"""
 		Necessary class to validate configuration.
		"""
		
		MIN_NDET					: int	# number of previous detections
		MIN_TSPAN 					: float	# minimum duration of alert detection history [days]
		MAX_TSPAN 					: float # maximum duration of alert detection history [days]
		MIN_RB						: float # real bogus score
		MIN_DRB						: float = 0.  # deep learning real bogus score
		MAX_FWHM					: float # sexctrator FWHM (assume Gaussian) [pix]
		MAX_ELONG					: float	# Axis ratio of image: aimage / bimage 
		MAX_MAGDIFF					: float	# Difference: magap - magpsf [mag]
		MAX_NBAD					: int	# number of bad pixels in a 5 x 5 pixel stamp
		MIN_DIST_TO_SSO				: float	#distance to nearest solar system object [arcsec]
		MIN_GAL_LAT 				: float	#minium distance from galactic plane. Set to negative to disable cut.
		GAIA_RS						: float	#search radius for GAIA DR2 matching [arcsec]
		GAIA_PM_SIGNIF				: float	# significance of proper motion detection of GAIA counterpart [sigma]
		GAIA_PLX_SIGNIF				: float	# significance of parallax detection of GAIA counterpart [sigma]
		GAIA_VETO_GMAG_MIN			: float	# min gmag for normalized distance cut of GAIA counterparts [mag]
		GAIA_VETO_GMAG_MAX			: float	# max gmag for normalized distance cut of GAIA counterparts [mag]
		GAIA_EXCESSNOISE_SIG_MAX	: float	# maximum allowed noise (expressed as significance) for Gaia match to be trusted.
		PS1_SGVETO_RAD				: float	# maximum distance to closest PS1 source for SG score veto [arcsec]
		PS1_SGVETO_SGTH				: float	# maximum allowed SG score for PS1 source within PS1_SGVETO_RAD
		PS1_CONFUSION_RAD			: float	# reject alerts if the three PS1 sources are all within this radius [arcsec]
		PS1_CONFUSION_SG_TOL		: float	# and if the SG score of all of these 3 sources is within this tolerance to 0.5


	def __init__(
		self, logger: logging.Logger, init_config: Dict[str, Any] = None, 
		resources: Dict[str, Any] = None
	):
		"""
		"""
		if not init_config:
			raise ValueError("Please check you init configuration")

		self.on_match_t2_units = [t2.unit_id for t2 in init_config.t2Compute]
		self.logger = logger if logger is not None else logging.getLogger()
		
		# parse the run config
		for k, val in init_config.dict().items():
			self.logger.info(f"Using {k}={val}")
		
		# ----- set filter properties ----- #
		
		# history
		self.min_ndet 					= init_config.MIN_NDET 
		self.min_tspan					= init_config.MIN_TSPAN
		self.max_tspan					= init_config.MAX_TSPAN
		
		# Image quality
		self.max_fwhm					= init_config.MAX_FWHM
		self.max_elong					= init_config.MAX_ELONG
		self.max_magdiff				= init_config.MAX_MAGDIFF
		self.max_nbad					= init_config.MAX_NBAD
		self.min_rb						= init_config.MIN_RB
		self.min_drb					= init_config.MIN_DRB

		# astro
		self.min_ssdistnr	 			= init_config.MIN_DIST_TO_SSO
		self.min_gal_lat				= init_config.MIN_GAL_LAT
		self.gaia_rs					= init_config.GAIA_RS
		self.gaia_pm_signif				= init_config.GAIA_PM_SIGNIF
		self.gaia_plx_signif			= init_config.GAIA_PLX_SIGNIF
		self.gaia_veto_gmag_min			= init_config.GAIA_VETO_GMAG_MIN
		self.gaia_veto_gmag_max			= init_config.GAIA_VETO_GMAG_MAX
		self.gaia_excessnoise_sig_max	= init_config.GAIA_EXCESSNOISE_SIG_MAX
		self.ps1_sgveto_rad				= init_config.PS1_SGVETO_RAD
		self.ps1_sgveto_th				= init_config.PS1_SGVETO_SGTH
		self.ps1_confusion_rad			= init_config.PS1_CONFUSION_RAD
		self.ps1_confusion_sg_tol		= init_config.PS1_CONFUSION_SG_TOL

		self.catshtm = catshtm_server.get_client(
			resources['catsHTM.default']
		)

		# To make this tenable we should create this list dynamically depending on what entries are required
		# by the filter. Now deciding not to include drb in this list, eg.
		self.keys_to_check = (
			'fwhm', 'elong', 'magdiff', 'nbad', 'distpsnr1', 'sgscore1', 'distpsnr2', 
			'sgscore2', 'distpsnr3', 'sgscore3', 'isdiffpos', 'ra', 'dec', 'rb', 'ssdistnr'
		)


	def _alert_has_keys(self, photop) -> bool:
		"""
		check that given photopoint contains all the keys needed to filter
		"""
		for el in self.keys_to_check:
			if el not in photop:
				self.logger.info(None, extra={"missing": el})
				return False
			if photop[el] is None:
				self.logger.info(None, extra={"isNone": el})
				return False
		return True


	def get_galactic_latitude(self, transient):
		"""
		compute galactic latitude of the transient
		"""
		coordinates = SkyCoord(transient['ra'], transient['dec'], unit='deg')
		return coordinates.galactic.b.deg


	def is_star_in_PS1(self, transient) -> bool:
		"""
		apply combined cut on sgscore1 and distpsnr1 to reject the transient if
		there is a PS1 star-like object in it's immediate vicinity
		"""
		
		#TODO: consider the case of alert moving wrt to the position of a star
		# maybe cut on the minimum of the distance!
		return transient['distpsnr1'] < self.ps1_sgveto_rad and \
			transient['sgscore1'] > self.ps1_sgveto_th


	def is_confused_in_PS1(self, transient) -> bool:
		"""
		check in PS1 for source confusion, which can induce subtraction artifatcs. 
		These cases are selected requiring that all three PS1 cps are in the imediate
		vicinity of the transient and their sgscore to be close to 0.5 within given tolerance.
		"""
		very_close = max(
			transient['distpsnr1'], 
			transient['distpsnr2'], 
			transient['distpsnr3']
		) < self.ps1_confusion_rad

		# Update 31.10.19: avoid costly numpy cast 
		# Old:
		# In: %timeit abs(array([sg1, sg2, sg3]) - 0.5 ).max()
		# Out: 5.79 µs ± 80.5 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
		# New:
		# In: %timeit max(abs(sg1-0.5), abs(sg2-0.5), abs(sg3-0.5))
		# Out: 449 ns ± 7.01 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)

		sg_confused = max(
			abs(transient['sgscore1']-0.5), 
			abs(transient['sgscore2']-0.5), 
			abs(transient['sgscore3']-0.5)
		) < self.ps1_confusion_sg_tol

		return sg_confused and very_close


	def is_star_in_gaia(self, transient) -> bool:
		"""
		match tranient position with GAIA DR2 and uses parallax 
		and proper motion to evaluate star-likeliness
		returns: True (is a star) or False otehrwise.
		"""
		
		transient_coords = SkyCoord(
			transient['ra'], transient['dec'], unit='deg'
		)

		srcs, colnames, colunits = self.catshtm.cone_search(
			'GAIADR2', transient_coords.ra.rad, 
			transient_coords.dec.rad, self.gaia_rs
		)

		my_keys = [
			'RA', 'Dec', 'Mag_G', 'PMRA', 'ErrPMRA', 'PMDec', 
			'ErrPMDec', 'Plx', 'ErrPlx', 'ExcessNoiseSig'
		]

		if len(srcs) > 0:

			gaia_tab = Table(asarray(srcs), names=colnames)
			gaia_tab = gaia_tab[my_keys]
			gaia_coords	= SkyCoord(
				gaia_tab['RA'], gaia_tab['Dec'], unit='rad'
			)
			
			# compute distance
			gaia_tab['DISTANCE'] = transient_coords.separation(gaia_coords).arcsec
			gaia_tab['DISTANCE_NORM'] = (
				1.8 + 0.6 * exp( (20 - gaia_tab['Mag_G']) / 2.05) > gaia_tab['DISTANCE']
			)
			gaia_tab['FLAG_PROX'] = [
				x['DISTANCE_NORM'] and \
				self.gaia_veto_gmag_min <= x['Mag_G'] <= self.gaia_veto_gmag_max
				for x in gaia_tab
			]

			# check for proper motion and parallax conditioned to distance
			gaia_tab['FLAG_PMRA'] = abs(gaia_tab['PMRA'] / gaia_tab['ErrPMRA']) > self.gaia_pm_signif
			gaia_tab['FLAG_PMDec'] = abs(gaia_tab['PMDec'] / gaia_tab['ErrPMDec']) > self.gaia_pm_signif
			gaia_tab['FLAG_Plx'] = abs(gaia_tab['Plx'] / gaia_tab['ErrPlx']) > self.gaia_plx_signif
			
			# take into account precison of the astrometric solution via the ExcessNoise key
			gaia_tab['FLAG_Clean'] = gaia_tab['ExcessNoiseSig']<self.gaia_excessnoise_sig_max
			
			# select just the sources that are close enough and that are not noisy
			gaia_tab = gaia_tab[gaia_tab['FLAG_PROX']]
			gaia_tab = gaia_tab[gaia_tab['FLAG_Clean']]
			
			# among the remaining sources there is anything with
			# significant proper motion or parallax measurement
			if (any(gaia_tab['FLAG_PMRA'] == True) or 
				any(gaia_tab['FLAG_PMDec'] == True) or
				any(gaia_tab['FLAG_Plx'] == True)):
				return True
		return False


	# Override
	def apply(self, ampel_alert: AmpelAlert) -> Optional[Set[str]]:
		"""
		Mandatory implementation.
		To exclude the alert, return *None*
		To accept it, either return
		* self.on_match_t2_units
		* or a custom combination of T2 unit names
		"""
		
		# --------------------------------------------------------------------- #
		#					CUT ON THE HISTORY OF THE ALERT						#
		# --------------------------------------------------------------------- #
		
		npp = len(ampel_alert.pps)
		if npp < self.min_ndet:
			#self.logger.debug("rejected: %d photopoints in alert (minimum required %d)"% (npp, self.min_ndet))
			self.logger.info(None, extra={'nDet': npp})
			return None
		
		# cut on length of detection history
		detections_jds = ampel_alert.get_values('jd', upper_limits=False)
		det_tspan = max(detections_jds) - min(detections_jds)
		if not (self.min_tspan < det_tspan < self.max_tspan):
			#self.logger.debug("rejected: detection history is %.3f d long, \
			# requested between %.3f and %.3f d"% (det_tspan, self.min_tspan, self.max_tspan))
			self.logger.info(None, extra={'tSpan': det_tspan})
			return None
		
		# --------------------------------------------------------------------- #
		#							IMAGE QUALITY CUTS							#
		# --------------------------------------------------------------------- #
		
		latest = ampel_alert.pps[0]
		if not self._alert_has_keys(latest):
			return None
		
		if (latest['isdiffpos'] == 'f' or latest['isdiffpos'] == '0'):
			#self.logger.debug("rejected: 'isdiffpos' is %s", latest['isdiffpos'])
			self.logger.info(None, extra={'isdiffpos': latest['isdiffpos']})
			return None
		
		if latest['rb'] < self.min_rb:
			#self.logger.debug("rejected: RB score %.2f below threshod (%.2f)"% (latest['rb'], self.min_rb))
			self.logger.info(None, extra={'rb': latest['rb']})
			return None

		if self.min_drb > 0. and latest['drb'] < self.min_drb:
			#self.logger.debug("rejected: RB score %.2f below threshod (%.2f)"% (latest['rb'], self.min_rb))
			self.logger.info(None, extra={'drb': latest['drb']})
			return None

		
		if latest['fwhm'] > self.max_fwhm:
			#self.logger.debug("rejected: fwhm %.2f above threshod (%.2f)"% (latest['fwhm'], self.max_fwhm))
			self.logger.info(None, extra={'fwhm': latest['fwhm']})
			return None
		
		if latest['elong'] > self.max_elong:
			#self.logger.debug("rejected: elongation %.2f above threshod (%.2f)"% (latest['elong'], self.max_elong))
			self.logger.info(None, extra={'elong': latest['elong']})
			return None
		
		if abs(latest['magdiff']) > self.max_magdiff:
			#self.logger.debug("rejected: magdiff (AP-PSF) %.2f above threshod (%.2f)"% (latest['magdiff'], self.max_magdiff))
			self.logger.info(None, extra={'magdiff': latest['magdiff']})
			return None
		
		# --------------------------------------------------------------------- #
		#								ASTRONOMY								#
		# --------------------------------------------------------------------- #
		
		# check for closeby ss objects
		if (0 <= latest['ssdistnr'] < self.min_ssdistnr):
			#self.logger.debug("rejected: solar-system object close to transient (max allowed: %d)."% (self.min_ssdistnr))
			self.logger.info(None, extra={'ssdistnr': latest['ssdistnr']})
			return None
		
		# cut on galactic latitude
		b = self.get_galactic_latitude(latest)
		if abs(b) < self.min_gal_lat:
			#self.logger.debug("rejected: b=%.4f, too close to Galactic plane (max allowed: %f)."% (b, self.min_gal_lat))
			self.logger.info(None, extra={'galPlane': abs(b)})
			return None
		
		# check ps1 star-galaxy score
		if self.is_star_in_PS1(latest):
			#self.logger.debug("rejected: closest PS1 source %.2f arcsec away with sgscore of %.2f"% (latest['distpsnr1'], latest['sgscore1']))
			self.logger.info(None, extra={'distpsnr1': latest['distpsnr1']})
			return None
		
		if self.is_confused_in_PS1(latest):
			#self.logger.debug("rejected: three confused PS1 sources within %.2f arcsec from ampel_alert."% (self.ps1_confusion_rad))
			self.logger.info(None, extra={'ps1Confusion': True})
			return None
		
		# check with gaia
		if self.is_star_in_gaia(latest):
			#self.logger.debug("rejected: within %.2f arcsec from a GAIA start (PM of PLX)" % (self.gaia_rs))
			self.logger.info(None, extra={'gaiaIsStar': True})
			return None
		
		# congratulation alert! you made it!
		#self.logger.debug("Alert %s accepted. Latest pp ID: %d"%(ampel_alert.tran_id, latest['candid']))
		self.logger.debug("Alert accepted", extra={'latestPpId': latest['candid']})

		#for key in self.keys_to_check:
		#	self.logger.debug("{}: {}".format(key, latest[key]))

		return self.on_match_t2_units
