import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import os, sys
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import time
from pyproj import Proj

state_borders = cfeature.STATES.with_scale("50m")


def skill_img(
    grid, x, y, title, cbar_label, geoproj, vmin=0, vmax=1, vinc=0.1, ticks=np.array([])
):
    fs = 14
    mapcolor = "black" if ("POD" in title) else "white"

    #cmap = plt.get_cmap("plasma")  # .copy()
    cmap = plt.get_cmap("seismic_r")  # .copy()
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

def calculate_metrics(spatial_counts_file):
    datafile = spatial_counts_file  #'tf/c02051315/model-11/spatial_counts.nc'
    nc = netCDF4.Dataset(datafile, "r")
    all_hits = nc["hits"][:]  
    all_misses = nc["misses"][:] 
    all_FAs = nc["FAs"][:]  
    thresholds = nc["thresholds"][:]
    nc.close()
    
    mask_val = 2
    glm_count = all_hits + all_misses
    glm_count = np.ma.masked_less(glm_count, mask_val)
    mask = glm_count.mask
    all_hits.mask = mask
    all_misses.mask = mask
    all_FAs.mask = mask

    all_csi = all_hits / (all_hits + all_misses + all_FAs)
    all_pod = all_hits / (all_hits + all_misses)
    all_far = all_FAs / (all_FAs + all_hits)
    glm_count = all_hits + all_misses

    return all_csi, all_pod, all_far, thresholds


def main(
    spatial_counts_file1,
    spatial_counts_file2,
    problevel,
    outdir=None,
    georeference_file=f"",
    stride=1,
    ticks=np.array([]),
):

    all_csi_1, all_pod_1, all_far_1, thresholds = calculate_metrics(spatial_counts_file1)
    all_csi_2, all_pod_2, all_far_2, thresholds = calculate_metrics(spatial_counts_file2)

    # outdir
    if outdir is None:
        outdir = os.path.dirname(datafile)

    # georeference file
    nc = netCDF4.Dataset(georeference_file, "r")
    gip = nc.variables["goes_imager_projection"]
    x = nc.variables["x"][::stride]
    y = nc.variables["y"][::stride]
    print(nc.variables["x"].shape)
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


    problevel = float(problevel)

    assert problevel in thresholds
    threshIdx = np.where(thresholds == problevel)[0][0]
    thresh = str(int(np.round(thresholds[threshIdx], 2) * 100))

    dif_csi = all_csi_2[threshIdx] - all_csi_1[threshIdx]
    dif_far = all_far_2[threshIdx] - all_far_1[threshIdx]
    dif_pod = all_pod_2[threshIdx] - all_pod_1[threshIdx]
    print(np.sum(dif_csi[dif_csi>0]))
    plt.imshow(dif_csi)
    plt.show()

    print(x.shape, y.shape)
    skill_img(
        dif_csi,
        x,
        y,
        title=f"Critical Success Index (CSI) at probability ≥ {thresh}%",
        cbar_label="CSI []",
        geoproj=geoproj,
        vmin=-0.2,
        vmax=0.2,
        vinc=0.05,
    )
    plt.savefig(f"{outdir}/dif_geo_csi.png", bbox_inches="tight")
    plt.close()
    skill_img(
        dif_pod,
        x,
        y,
        title=f"Probability of Detection (POD) at probability ≥ {thresh}%",
        cbar_label="POD []",
        geoproj=geoproj,
        vmin=-0.2,
        vmax=0.2,
        vinc=0.05,
    )
    plt.savefig(f"{outdir}/dif_geo_pod.png", bbox_inches="tight")
    plt.close()
    skill_img(
        dif_far,
        x,
        y,
        title=f"False Alarm Ratio (FAR) at probability ≥ {thresh}%",
        cbar_label="FAR []",
        geoproj=geoproj,
        vmin=-0.2,
        vmax=0.2,
        vinc=0.05,
    )
    plt.savefig(f"{outdir}/dif_geo_far.png", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    spatial_counts_file1 = sys.argv[1]
    spatial_counts_file2 = sys.argv[2]
    problevel = sys.argv[3]  # e.g., 0.2 or 0.35
    georeference_file = sys.argv[4]
    if len(sys.argv) == 6:
        outdir = sys.argv[5]
        from lightningcast import utils

        utils.mkdir_p(outdir)
    else:
        outdir = os.path.dirname(spatial_counts_file2)

    main(
        spatial_counts_file1,
        spatial_counts_file2,
        problevel,
        georeference_file=georeference_file,
        outdir=outdir,
        stride=7,
    )
