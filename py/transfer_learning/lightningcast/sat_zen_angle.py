import numpy as np


def sat_zen_angle(xlat, xlon, satlat=0, satlon=-75.0):

    DTOR = np.pi / 180.0

    if isinstance(xlat, list):
        xlat = np.array(xlat)
    if isinstance(xlon, list):
        xlon = np.array(xlon)

    lon = (xlon - satlon) * DTOR
    lat = (xlat - satlat) * DTOR

    beta = np.arccos(np.cos(lat) * np.cos(lon))
    sin_beta = np.sin(beta)

    zenith = np.arcsin(
        42164.0 * sin_beta / np.sqrt(1.808e09 - 5.3725e08 * np.cos(beta))
    )

    zenith = zenith / DTOR

    return zenith
