import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import os, sys
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import time
from pyproj import Proj

#pltg = os.environ["PLTG"]

# state_borders = cfeature.NaturalEarthFeature(category='cultural', name='admin_1_states_provinces_lakes', scale='50m', facecolor='none')
state_borders = cfeature.STATES.with_scale("50m")


def skill_img(
    grid, x, y, title, cbar_label, geoproj, vmin=0, vmax=1, vinc=0.1, ticks=np.array([])
):
    fs = 14
    mapcolor = "black" if ("POD" in title) else "white"

    cmap = plt.get_cmap("plasma")  # .copy()
    cmap.set_bad("gray")
    fig = plt.figure(figsize=(10, 8))  # plt.figure(figsize=(10,8))
    ax = fig.add_axes([0, 0, 1, 1], projection=geoproj)

    # goes east/west CONUS
    ax.set_extent([x.min(), x.max(), y.min(), y.max()], crs=geoproj)
    img = ax.pcolormesh(x, y, grid, vmin=vmin, vmax=vmax, cmap=cmap)

    # goes east oconus
    # ax.set_extent([x[700],x[3900],y[3000],y[500]], crs=geoproj)
    # img = ax.pcolormesh(x[700:3900], y[500:3000], grid, vmin=vmin, vmax=vmax, cmap=cmap)

    # goes west "oconus" FD sector
    # ax.set_extent([x[1100],x[4100],y[2900],y[400]], crs=geoproj)
    # img = ax.pcolormesh(x[1100:4100], y[400:2900], grid, vmin=vmin, vmax=vmax, cmap=cmap)

    extend = "max" if ("flash-extent" in title) else "both"
    if len(ticks):
        cbar = plt.colorbar(
            img,
            orientation="horizontal",
            pad=0.01,
            shrink=0.8,
            extend=extend,
            drawedges=0,
            ticks=ticks,
        )
    else:
        cbar = plt.colorbar(
            img,
            orientation="horizontal",
            pad=0.01,
            shrink=0.8,
            extend=extend,
            drawedges=0,
            ticks=np.arange(vmin, vmax + vinc, vinc),
        )
    cbar.set_label(cbar_label, fontsize=fs + 2)
    cbar.ax.tick_params(labelsize=fs + 2)
    ax.add_feature(state_borders, edgecolor=mapcolor, linewidth=1.0, facecolor="none")
    ax.coastlines(color=mapcolor, linewidth=1.0)
    # plt.suptitle(f'GOES-R Probability of Lightning (PLTG)',fontsize=fs+4)
    print(title)
    plt.title(title, fontsize=fs + 2)


def main(
    spatial_counts_file,
    problevel,
    outdir=None,
    #georeference_file=f"{pltg}/lightningcast/static/GOES_East.nc",
    georeference_file=f"",
    stride=1,
    ticks=np.array([]),
):
    datafile = spatial_counts_file  #'tf/c02051315/model-11/spatial_counts.nc'
    nc = netCDF4.Dataset(datafile, "r")
    all_hits = nc["hits"][
        :
    ]  # [:,22:1522,362:2862]     #these are the PACUS sector extruded from the FD
    all_misses = nc["misses"][:]  # [:,22:1522,362:2862]
    all_FAs = nc["FAs"][:]  # [:,22:1522,362:2862]
    thresholds = nc["thresholds"][:]
    nc.close()

    # outdir
    if outdir is None:
        outdir = os.path.dirname(datafile)

    # CONUS sector georeference file
    nc = netCDF4.Dataset(georeference_file, "r")
    gip = nc.variables["goes_imager_projection"]
    x = nc.variables["x"][::stride]
    y = nc.variables["y"][::stride]
    p = Proj(
        "+proj=geos +lon_0="
        + str(gip.longitude_of_projection_origin)
        + " +h="
        + str(gip.perspective_point_height)
        + " +x_0=0.0 +y_0=0.0 +a="
        + str(gip.semi_major_axis)
        + " +b="
        + str(gip.semi_minor_axis)
    )
    x *= gip.perspective_point_height
    y *= gip.perspective_point_height

    geoproj = ccrs.Geostationary(
        central_longitude=gip.longitude_of_projection_origin,
        sweep_axis="x",
        satellite_height=gip.perspective_point_height,
    )

    # make some images
    mask_val = 2
    glm_count = all_hits + all_misses
    glm_count = np.ma.masked_less(glm_count, mask_val)
    mask = glm_count.mask

    all_hits.mask = mask
    all_misses.mask = mask
    all_FAs.mask = mask

    all_csi = all_hits / (all_hits + all_misses + all_FAs)
    maxind = np.argmax(all_csi, axis=0)
    best_thresh = np.ma.zeros(maxind.shape)
    ny, nx = maxind.shape
    for ii in range(ny):
        for jj in range(nx):
            best_thresh[ii, jj] = thresholds[maxind[ii, jj]]
    best_thresh *= 100
    best_thresh = np.ma.array(
        best_thresh, mask=mask[0]
    )  # mask is 3D, but we only need 1 slice, since all slices should be the same
    all_pod = all_hits / (all_hits + all_misses)
    all_far = all_FAs / (all_FAs + all_hits)

    problevel = float(problevel)

    assert problevel in thresholds
    threshIdx = np.where(thresholds == problevel)[0][0]
    thresh = str(int(np.round(thresholds[threshIdx], 2) * 100))

    skill_img(
        all_csi[threshIdx],
        x,
        y,
        title=f"Critical Success Index (CSI) at probability ≥ {thresh}%",
        cbar_label="CSI []",
        geoproj=geoproj,
        vmin=0.1,
        vmax=0.8,
        vinc=0.1,
    )
    plt.savefig(f"{outdir}/geo_csi.png", bbox_inches="tight")
    plt.close()
    skill_img(
        all_pod[threshIdx],
        x,
        y,
        title=f"Probability of Detection (POD) at probability ≥ {thresh}%",
        cbar_label="POD []",
        geoproj=geoproj,
        vmin=0.1,
        vmax=0.8,
        vinc=0.1,
    )
    plt.savefig(f"{outdir}/geo_pod.png", bbox_inches="tight")
    plt.close()
    skill_img(
        all_far[threshIdx],
        x,
        y,
        title=f"False Alarm Ratio (FAR) at probability ≥ {thresh}%",
        cbar_label="FAR []",
        geoproj=geoproj,
        vmin=0.1,
        vmax=0.8,
        vinc=0.1,
    )
    plt.savefig(f"{outdir}/geo_far.png", bbox_inches="tight")
    plt.close()

    # get vmax
    if len(ticks) == 0:
        vmax = int(
            np.round(0.8 * np.max(glm_count[threshIdx]), -1)
        )  # round to nearest 10
        ticks = np.array(
            [
                mask_val,
                int(np.round(vmax * 0.2, -1)),
                int(np.round(vmax * 0.4, -1)),
                int(np.round(vmax * 0.6, -1)),
                int(np.round(vmax * 0.8, -1)),
                vmax,
            ]
        )
    else:
        vmax = np.max(ticks)

    skill_img(
        glm_count[threshIdx],
        x,
        y,
        title="Event count of GLM flash-extent density ≥ 1 flash in next 60 min",
        cbar_label="Number of events",
        geoproj=geoproj,
        vmin=mask_val,
        vmax=vmax,
        ticks=ticks,
    )  # np.array([mask_val,100,200,300,400])) #,500,600,700,800]))
    plt.savefig(f"{outdir}/obs_cts.png", bbox_inches="tight")
    plt.close()
    skill_img(
        best_thresh,
        x,
        y,
        title="Best CSI threshold",
        cbar_label="probability [%]",
        geoproj=geoproj,
        vmin=20,
        vmax=90,
        ticks=np.array([20, 30, 40, 50, 60, 70, 80, 90]),
    )
    plt.savefig(f"{outdir}/best_thresh.png", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    spatial_counts_file = sys.argv[1]
    problevel = sys.argv[2]  # e.g., 0.2 or 0.35
    georeference_file = sys.argv[3]
    if len(sys.argv) == 5:
        outdir = sys.argv[4]
        from lightningcast import utils

        utils.mkdir_p(outdir)
    else:
        outdir = os.path.dirname(spatial_counts_file)
    main(
        spatial_counts_file,
        problevel,
        georeference_file=georeference_file,
        outdir=outdir,
    )
