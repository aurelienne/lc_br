"""Methods for plotting attributes diagram."""
import inspect, os, sys

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, os.path.dirname(current_dir))

import glob
from lightningcast import bootstrap
import numpy as np
import shapely.geometry
import matplotlib as mpl
import matplotlib.colors
import matplotlib.pyplot as plt
import pickle

DEFAULT_NUM_BINS = 20
RELIABILITY_LINE_COLOUR = np.array([228, 26, 28], dtype=float) / 255
RELIABILITY_LINE_WIDTH = 1.5
PERFECT_LINE_COLOUR = np.full(3, 152.0 / 255)
PERFECT_LINE_WIDTH = 2

NO_SKILL_LINE_COLOUR = np.array([31, 120, 180], dtype=float) / 255
NO_SKILL_LINE_WIDTH = 2
SKILL_AREA_TRANSPARENCY = 0.2
CLIMATOLOGY_LINE_COLOUR = np.full(3, 152.0 / 255)
CLIMATOLOGY_LINE_WIDTH = 2

HISTOGRAM_FACE_COLOUR = np.array([228, 26, 28], dtype=float) / 255
HISTOGRAM_EDGE_COLOUR = np.full(3, 0.0)
HISTOGRAM_EDGE_WIDTH = 0.5

HISTOGRAM_LEFT_EDGE_COORD = 0.18 #0.575
HISTOGRAM_BOTTOM_EDGE_COORD = 0.62 #0.175
HISTOGRAM_WIDTH = 0.25
HISTOGRAM_HEIGHT = 0.25

HISTOGRAM_X_TICK_VALUES = np.linspace(0, 1, num=6, dtype=float)
HISTOGRAM_Y_TICK_SPACING = 0.1

FIGURE_WIDTH_INCHES = 4
FIGURE_HEIGHT_INCHES = 4

FONT_SIZE = 14
plt.rc("font", size=FONT_SIZE)
plt.rc("axes", titlesize=FONT_SIZE)
plt.rc("axes", labelsize=FONT_SIZE)
plt.rc("xtick", labelsize=FONT_SIZE)
plt.rc("ytick", labelsize=FONT_SIZE)
plt.rc("legend", fontsize=FONT_SIZE)
plt.rc("figure", titlesize=FONT_SIZE)

def get_month_name(month):
    if month == "01-02":
        month_name = "Jan/Feb"
    elif month == "01":
        month_name = "Jan"
    elif month == "02":
        month_name = "Feb"
    elif month == "03":
        month_name = "Mar"
    elif month == "04":
        month_name = "Apr"
    elif month == "05":
        month_name = "May"
    elif month == "06":
        month_name = "Jun"
    elif month == "07":
        month_name = "Jul"
    elif month == "08":
        month_name = "Aug"
    elif month == "09":
        month_name = "Sep"
    elif month == "10":
        month_name = "Oct"
    elif month == "11":
        month_name = "Nov"
    elif month == "12":
        month_name = "Dec"
    elif month == "11-12":
        month_name = "Nov/Dec"

    return month_name


def _get_histogram(input_values, num_bins, min_value, max_value):
    """Creates histogram with uniform bin-spacing.
    E = number of input values
    B = number of bins
    :param input_values: length-E np array of values to bin.
    :param num_bins: Number of bins (B).
    :param min_value: Minimum value.  Any input value < `min_value` will be
        assigned to the first bin.
    :param max_value: Max value.  Any input value > `max_value` will be
        assigned to the last bin.
    :return: inputs_to_bins: length-E np array of bin indices (integers).
    """

    bin_cutoffs = np.linspace(min_value, max_value, num=num_bins + 1)
    inputs_to_bins = np.digitize(input_values, bin_cutoffs, right=False) - 1

    inputs_to_bins[inputs_to_bins < 0] = 0
    inputs_to_bins[inputs_to_bins > num_bins - 1] = num_bins - 1

    return inputs_to_bins


def _get_points_in_relia_curve(
    observed_labels, forecast_probabilities, num_bins, nboots=0
):
    """Creates points for reliability curve.
    The reliability curve is the main component of the attributes diagram.
    E = number of examples
    B = number of bins
    :param observed_labels: length-E np array of class labels (integers in
        0...1).
    :param forecast_probabilities: length-E np array with forecast
        probabilities of label = 1.
    :param num_bins: Number of bins for forecast probability.
    :param nboots: Number of times to bootstrap sample to obtain 95% CI
    :return: mean_forecast_probs: length-B np array of mean forecast
        probabilities.
    :return: mean_event_frequencies: length-B np array of conditional mean
        event frequencies.  mean_event_frequencies[j] = frequency of label 1
        when forecast probability is in the [j]th bin.
    :return: num_examples_by_bin: length-B np array with number of examples
        in each forecast bin.
    """

    assert np.all(np.logical_or(observed_labels == 0, observed_labels == 1))

    assert np.all(
        np.logical_and(forecast_probabilities >= 0, forecast_probabilities <= 1)
    )

    assert num_bins > 1

    inputs_to_bins = _get_histogram(
        input_values=forecast_probabilities,
        num_bins=num_bins,
        min_value=0.0,
        max_value=1.0,
    )

    mean_forecast_probs = np.full(num_bins, np.nan)
    mean_event_frequencies = np.full(num_bins, np.nan)
    num_examples_by_bin = np.full(num_bins, -1, dtype=int)

    if nboots > 0:
        forecast_probs_05 = np.full(num_bins, np.nan)
        forecast_probs_95 = np.full(num_bins, np.nan)
        event_freqs_05 = np.full(num_bins, np.nan)
        event_freqs_95 = np.full(num_bins, np.nan)

    for k in range(num_bins):
        these_example_indices = np.where(inputs_to_bins == k)[0]
        num_examples_by_bin[k] = len(these_example_indices)

        mean_forecast_probs[k] = np.mean(
            forecast_probabilities[these_example_indices]
        )

        mean_event_frequencies[k] = np.mean(
            observed_labels[these_example_indices].astype(float)
        )

        if nboots > 0:
            boot_mean_prob = bootstrap.bootstrap(
                forecast_probabilities[these_example_indices],
                bootnum=nboots,
                bootfunc=np.mean,
            )
            boot_mean_freq = bootstrap.bootstrap(
                observed_labels[these_example_indices],
                bootnum=nboots,
                bootfunc=np.mean,
            )

            forecast_probs_05[k] = np.percentile(boot_mean_prob, 5)
            forecast_probs_95[k] = np.percentile(boot_mean_prob, 95)
            event_freqs_05[k] = np.percentile(boot_mean_freq, 5)
            event_freqs_95[k] = np.percentile(boot_mean_freq, 95)

    if nboots > 0:
        return (
            mean_forecast_probs,
            mean_event_frequencies,
            num_examples_by_bin,
            forecast_probs_05,
            forecast_probs_95,
            event_freqs_05,
            event_freqs_95,
        )
    else:
        return mean_forecast_probs, mean_event_frequencies, num_examples_by_bin


def _vertices_to_polygon_object(x_vertices, y_vertices):
    """Converts two arrays of vertices to `shapely.geometry.Polygon` object.
    V = number of vertices
    This method allows for simple polygons only (no disjoint polygons, no
    holes).
    :param x_vertices: length-V np array of x-coordinates.
    :param y_vertices: length-V np array of y-coordinates.
    :return: polygon_object: Instance of `shapely.geometry.Polygon`.
    """

    list_of_vertices = []
    for i in range(len(x_vertices)):
        list_of_vertices.append((x_vertices[i], y_vertices[i]))

    return shapely.geometry.Polygon(shell=list_of_vertices)


def _plot_background(axes_object, observed_labels, climo=-1):
    """Plots background of attributes diagram.
    E = number of examples
    :param axes_object: Instance of `matplotlib.axes._subplots.AxesSubplot`.
        Will plot on these axes.
    :param observed_labels: length-E np array of class labels (integers in
        0...1).
    """

    if climo > 0:
        # We usually don't do this, but let the program determine the climo. value.
        # This feature may be helpful when plotting multiple rel. curves and we want
        # to overried the computed climo.
        climatology = climo
    else:
        # Plot positive-skill area.
        climatology = np.mean(observed_labels.astype(float))

    skill_area_colour = matplotlib.colors.to_rgba(
        NO_SKILL_LINE_COLOUR, SKILL_AREA_TRANSPARENCY
    )

    x_vertices_left = np.array([0, climatology, climatology, 0, 0])
    y_vertices_left = np.array([0, 0, climatology, climatology / 2, 0])

    axes_object.fill(x_vertices_left,y_vertices_left,color=skill_area_colour)

    x_vertices_right = np.array([climatology, 1, 1, climatology, climatology])
    y_vertices_right = np.array(
        [climatology, (1 + climatology) / 2, 1, 1, climatology]
    )

    axes_object.fill(x_vertices_right,y_vertices_right,color=skill_area_colour)

    # Plot no-skill line (at edge of positive-skill area).
    no_skill_x_coords = np.array([0, 1], dtype=float)
    no_skill_y_coords = np.array([climatology, 1 + climatology]) / 2
    axes_object.plot(
        no_skill_x_coords,
        no_skill_y_coords,
        color=NO_SKILL_LINE_COLOUR,
        linestyle="solid",
        linewidth=NO_SKILL_LINE_WIDTH,
    )

    # Plot climatology line (vertical).
    climo_line_x_coords = np.full(2, climatology)
    climo_line_y_coords = np.array([0, 1], dtype=float)
    axes_object.plot(
        climo_line_x_coords,
        climo_line_y_coords,
        color=CLIMATOLOGY_LINE_COLOUR,
        linestyle="dashed",
        linewidth=CLIMATOLOGY_LINE_WIDTH,
    )

    # Plot no-resolution line (horizontal).
    no_resolution_x_coords = climo_line_y_coords + 0.0
    no_resolution_y_coords = climo_line_x_coords + 0.0
    axes_object.plot(
        no_resolution_x_coords,
        no_resolution_y_coords,
        color=CLIMATOLOGY_LINE_COLOUR,
        linestyle="dashed",
        linewidth=CLIMATOLOGY_LINE_WIDTH,
    )


def _floor_to_nearest(input_value_or_array, increment):
    """Rounds number(s) down to the nearest multiple of `increment`.
    :param input_value_or_array: Input (either scalar or np array).
    :param increment: Increment (or rounding base -- whatever you want to call
        it).
    :return: output_value_or_array: Rounded version of `input_value_or_array`.
    """

    return increment * np.floor(input_value_or_array / increment)


def _plot_forecast_histogram(
    figure_object, num_examples_by_bin, facecolor=HISTOGRAM_FACE_COLOUR
):
    """Plots forecast histogram as inset in the attributes diagram.
    B = number of bins
    :param figure_object: Instance of `matplotlib.figure.Figure`.  Will plot in
        this figure.
    :param num_examples_by_bin: length-B np array, where
        num_examples_by_bin[j] = number of examples in [j]th forecast bin.
    """

    num_bins = len(num_examples_by_bin)
    bin_frequencies = num_examples_by_bin.astype(float) / np.sum(num_examples_by_bin)

    forecast_bin_edges = np.linspace(0, 1, num=num_bins + 1, dtype=float)
    forecast_bin_width = forecast_bin_edges[1] - forecast_bin_edges[0]
    forecast_bin_centers = forecast_bin_edges[:-1] + forecast_bin_width / 2

    inset_axes_object = figure_object.add_axes(
        [
            HISTOGRAM_LEFT_EDGE_COORD,
            HISTOGRAM_BOTTOM_EDGE_COORD,
            HISTOGRAM_WIDTH,
            HISTOGRAM_HEIGHT,
        ]
    )

    inset_axes_object.bar(
        forecast_bin_centers,
        bin_frequencies,
        forecast_bin_width,
        color=facecolor,
        edgecolor=HISTOGRAM_EDGE_COLOUR,
        linewidth=HISTOGRAM_EDGE_WIDTH,
    )

    max_y_tick_value = _floor_to_nearest(
        1.05 * np.max(bin_frequencies), HISTOGRAM_Y_TICK_SPACING
    )
    num_y_ticks = 1 + int(np.round(max_y_tick_value / HISTOGRAM_Y_TICK_SPACING))

    y_tick_values = np.linspace(0, max_y_tick_value, num=num_y_ticks)
    plt.yticks(y_tick_values, axes=inset_axes_object, fontsize=FONT_SIZE - 6)
    plt.xticks(
        HISTOGRAM_X_TICK_VALUES, axes=inset_axes_object, fontsize=FONT_SIZE - 6
    )

    inset_axes_object.set_xlim(0, 1)
    inset_axes_object.set_ylim(0, 1.05 * np.max(bin_frequencies))


def _plot_forecast_histogram_same_axes(
    axes_object, num_examples_by_bin, facecolor=HISTOGRAM_FACE_COLOUR
):
    """Plots forecast histogram as inset in the attributes diagram.
    B = number of bins
    :param axes_object: Instance of `matplotlib.Axes`.  Will make new axes based off this one.
    :param num_examples_by_bin: length-B np array, where
        num_examples_by_bin[j] = number of examples in [j]th forecast bin.
    """

    num_bins = len(num_examples_by_bin)
    bin_frequencies = num_examples_by_bin.astype(float) / np.sum(num_examples_by_bin)

    forecast_bin_edges = np.linspace(0, 1, num=num_bins + 1, dtype=float)
    forecast_bin_width = forecast_bin_edges[1] - forecast_bin_edges[0]
    forecast_bin_centers = forecast_bin_edges[:-1] + forecast_bin_width / 2

    # second axis
    width = 0.1
    color = facecolor
    ax2 = axes_object.twinx()
    ax2.set_ylim(0, 1.05 * np.max(bin_frequencies))
    ax2.set_xlim(0, 1)
    ax2.spines["right"].set_color(color)
    ax2.tick_params(axis="y", colors=color, which="both")
    cnt_rects = ax2.bar(
        forecast_bin_centers,
        bin_frequencies,
        forecast_bin_width,
        fill=False,
        linewidth=HISTOGRAM_EDGE_WIDTH + 0.5,
        edgecolor=color,
    )
    ax2.set_ylabel("Normalized prediction frequency [bars]")

    max_y_tick_value = _floor_to_nearest(
        1.05 * np.max(bin_frequencies), HISTOGRAM_Y_TICK_SPACING
    )
    num_y_ticks = 1 + int(np.round(max_y_tick_value / HISTOGRAM_Y_TICK_SPACING))

    y_tick_values = np.linspace(0, max_y_tick_value, num=num_y_ticks)
    plt.yticks(y_tick_values, axes=ax2)
    plt.xticks(HISTOGRAM_X_TICK_VALUES, axes=ax2)


def plot_reliability_curve(
    observed_labels,
    forecast_probabilities,
    label,
    num_bins=DEFAULT_NUM_BINS,
    axes_object=None,
    color=RELIABILITY_LINE_COLOUR,
    linewidth=RELIABILITY_LINE_WIDTH,
    **kwargs
):
    """Plots reliability curve.
    E = number of examples
    :param observed_labels: length-E np array of class labels (integers in
        0...1).
    :param forecast_probabilities: length-E np array with forecast
        probabilities of label = 1.
    :param num_bins: Number of bins for forecast probability.
    :param axes_object: Instance of `matplotlib.axes._subplots.AxesSubplot`.
        Will plot on these axes.
    :return: mean_forecast_probs: See doc for `_get_points_in_relia_curve`.
    :return: mean_event_frequencies: Same.
    :return: num_examples_by_bin: Same.
    """

    (
        mean_forecast_probs,
        mean_event_frequencies,
        num_examples_by_bin,
    ) = _get_points_in_relia_curve(
        observed_labels=observed_labels,
        forecast_probabilities=forecast_probabilities,
        num_bins=num_bins,
    )

    if axes_object is None:
        _, axes_object = plt.subplots(
            1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
        )

    perfect_x_coords = np.array([0, 1], dtype=float)
    perfect_y_coords = perfect_x_coords + 0.0
    axes_object.plot(
        perfect_x_coords,
        perfect_y_coords,
        color=PERFECT_LINE_COLOUR,
        linestyle="dashed",
        linewidth=PERFECT_LINE_WIDTH,
    )

    real_indices = np.where(
        np.invert(
            np.logical_or(
                np.isnan(mean_forecast_probs), np.isnan(mean_event_frequencies)
            )
        )
    )[0]

    (line_obj,) = axes_object.plot(
        mean_forecast_probs[real_indices],
        mean_event_frequencies[real_indices],
        color=color,
        linestyle="solid",
        linewidth=linewidth,
        label=label,
    )

    if "probs95" in kwargs:
        probs95 = kwargs["probs95"]
        freqs95 = kwargs["freqs95"]
        probs05 = kwargs["probs05"]
        freqs05 = kwargs["freqs05"]
        axes_object.fill_between(
            mean_forecast_probs[real_indices],
            freqs05[real_indices],
            freqs95[real_indices],
            facecolor=RELIABILITY_LINE_COLOUR,
            alpha=0.3,
        )

    axes_object.set_xlabel("Forecast probability")
    axes_object.set_ylabel("Conditional event frequency")
    axes_object.set_xlim(0.0, 1.0)
    axes_object.set_ylim(0.0, 1.0)

    # return mean_forecast_probs, mean_event_frequencies, num_examples_by_bin
    return line_obj


def plot_attributes_diagram(
    observed_labels,
    forecast_probabilities,
    label='',
    num_bins=DEFAULT_NUM_BINS,
    return_main_axes=False,
    nboots=0,
    plot_hist=True,
    color=RELIABILITY_LINE_COLOUR,
    linewidth=RELIABILITY_LINE_WIDTH,
    climo=-1,
):
    """Plots attributes diagram.
    :param observed_labels: See doc for `plot_reliability_curve`.
    :param forecast_probabilities: Same.
    :param num_bins: Same.
    :param nboots: See doc for `_get_points_in_relia_curve`.
    :return: mean_forecast_probs: See doc for `_get_points_in_relia_curve`.
    :return: mean_event_frequencies: Same.
    :return: num_examples_by_bin: Same.
    """

    if nboots > 0:
        (
            mean_forecast_probs,
            mean_event_frequencies,
            num_examples_by_bin,
            forecast_probs_05,
            forecast_probs_95,
            event_freqs_05,
            event_freqs_95,
        ) = _get_points_in_relia_curve(
            observed_labels=observed_labels,
            forecast_probabilities=forecast_probabilities,
            num_bins=num_bins,
            nboots=nboots,
        )
    else:
        (
            mean_forecast_probs,
            mean_event_frequencies,
            num_examples_by_bin,
        ) = _get_points_in_relia_curve(
            observed_labels=observed_labels,
            forecast_probabilities=forecast_probabilities,
            num_bins=num_bins,
        )

    figure_object, axes_object = plt.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    _plot_background(
        axes_object=axes_object, observed_labels=observed_labels, climo=climo
    )
    if plot_hist:
        _plot_forecast_histogram(
            figure_object=figure_object, num_examples_by_bin=num_examples_by_bin
        )
        # _plot_forecast_histogram_same_axes(axes_object=axes_object,num_examples_by_bin=num_examples_by_bin)

    if nboots > 0:
        line_obj = plot_reliability_curve(
            observed_labels=observed_labels,
            forecast_probabilities=forecast_probabilities,
            num_bins=num_bins,
            axes_object=axes_object,
            probs95=forecast_probs_95,
            probs05=forecast_probs_05,
            freqs95=event_freqs_95,
            freqs05=event_freqs_05,
            color=color,
            linewidth=linewidth,
            label=label,
        )
    else:
        line_obj = plot_reliability_curve(
            observed_labels=observed_labels,
            forecast_probabilities=forecast_probabilities,
            num_bins=num_bins,
            axes_object=axes_object,
            color=color,
            linewidth=linewidth,
            label=label,
        )

    if return_main_axes:
        return (
            mean_forecast_probs,
            mean_event_frequencies,
            num_examples_by_bin,
            axes_object,
            figure_object,
        )  # , line_obj
    else:
        return mean_forecast_probs, mean_event_frequencies, num_examples_by_bin

#---------------------------------------------------------------------------------------------------

def rel_with_obs_and_fcst_cts(obs_cts, fcst_cts, bins, outfilename,
                              obs_cts2=None, fcst_cts2=None, labels=None):

    obs_cts = np.array(obs_cts)
    fcst_cts = np.array(fcst_cts)
    bins = np.array(bins)

    rel = obs_cts / fcst_cts
    
    figure_object, axes_object = plt.subplots(
          1, 1, figsize=(6, 6)
      )
    
    #observed labels
    climo_value = np.sum(obs_cts) / np.sum(fcst_cts)
    # Estimate of the climo
    observed_labels = np.zeros(1000)
    observed_labels[0:int(np.round(climo_value,3)*1000)]=1
    
    #perfect line
    perfect_x_coords = np.array([0, 1], dtype=float)
    perfect_y_coords = perfect_x_coords + 0.
    
    color = np.full(3, 152. / 255)
    
    axes_object.plot(
        perfect_x_coords, perfect_y_coords, color=color,
        linestyle='dashed', linewidth=2)
    
    _plot_background(axes_object=axes_object, observed_labels=observed_labels)
    
    _plot_forecast_histogram(figure_object=figure_object,
                           num_examples_by_bin=fcst_cts)
    
    line_obj, = axes_object.plot(bins, rel, color='red', linestyle='solid', linewidth=4)
   
    if obs_cts2 is not None:
        rel2 = obs_cts2 / fcst_cts2
        line_obj2, = axes_object.plot(bins, rel2, color='orange', linestyle='solid', linewidth=4)
        if(labels is not None):
            leg = axes_object.legend([line_obj,line_obj2], labels,loc='lower right',fontsize=14) 

    axes_object.set_xlabel('Forecast probability')
    axes_object.set_ylabel('Conditional event frequency')
    axes_object.set_xlim(0., 1.)
    axes_object.set_ylim(0., 1.)
    
    plt.savefig(outfilename,bbox_inches="tight")
    print(f"Saved {outfilename}")

#--------------------------------------------------------------------------------------------------------

def rel_with_pkls(pkl, pkl2=None, outdir=None, index=[0,1], labels=['1fl','10fl']):
    if isinstance(index, int):
        index = [index]
    elif isinstance(index, list):
        for _ in index:
            assert(isinstance(_, int))
    else:
        print("index must be an int or a list of ints")
        sys.exit(1)

    if outdir is None:
        outdir = os.path.dirname(pkl)

    scores = pickle.load(open(pkl,'rb'))

    fcst_cts = {}
    obs_cts = {}
    bins = {}

    # Assuming keys look like this: 'fcstct05_index0', 'obsct45_index0', etc
    for ii in index: # For each index / output
        idx = f'index{ii}'
        fcst_cts[idx] = []
        obs_cts[idx] = []
        bins[idx] = []
        for key, val in scores.items():
            if 'obsct' in key and idx in key:
                prob = key.split('_')[0][-2:]
                obs_cts[idx].append(val)
                bins[idx].append((float(prob) - 2.5) / 100)
            if 'fcstct' in key and idx in key:
                prob = key.split('_')[0][-2]
                fcst_cts[idx].append(val)   

        # Sort values in ascending order of probability labels
        fcst_cts[idx] = np.array(fcst_cts[idx])
        obs_cts[idx] = np.array(obs_cts[idx])
        bins[idx] = np.array(bins[idx])
        fcst_cts[idx] = fcst_cts[idx][bins[idx].argsort()]
        obs_cts[idx] = obs_cts[idx][bins[idx].argsort()]
        bins[idx] = np.sort(bins[idx])
   
        outfilename = f"{outdir}/attributes_diagram_{idx}.png"
        rel_with_obs_and_fcst_cts(obs_cts[idx], fcst_cts[idx], bins[idx], outfilename)

    if 'index1' in fcst_cts and 1 in index:
        # do index0 alongside index1
     
        outfilename = f"{outdir}/attributes_diagram_index0_index1.png"
        rel_with_obs_and_fcst_cts(obs_cts['index0'], fcst_cts['index0'], bins['index0'], outfilename,
                                  obs_cts2=obs_cts['index1'], fcst_cts2=fcst_cts['index1'], labels=labels)

    if pkl2 is not None:
        # Compare index0 ouptut for two different models

        scores2 = pickle.load(open(pkl2,'rb'))    

        fcst_cts2 = {}
        obs_cts2 = {}
        bins2 = {}

        # Assuming keys look like this: 'fcstct05_index0', 'obsct45_index0', etc
        idx = f'index0'
        fcst_cts2[idx] = []
        obs_cts2[idx] = []
        bins2[idx] = []
        for key, val in scores2.items():
            if 'obsct' in key and idx in key:
                prob = key.split('_')[0][-2:]
                obs_cts2[idx].append(val)
                bins2[idx].append((float(prob) - 2.5) / 100)
            if 'fcstct' in key and idx in key:
                prob = key.split('_')[0][-2]
                fcst_cts2[idx].append(val)
    
        # Sort values in ascending order of probability labels
        fcst_cts2[idx] = np.array(fcst_cts2[idx])
        obs_cts2[idx] = np.array(obs_cts2[idx])
        bins2[idx] = np.array(bins2[idx])
        fcst_cts2[idx] = fcst_cts2[idx][bins2[idx].argsort()]
        obs_cts2[idx] = obs_cts2[idx][bins2[idx].argsort()]
    
        outfilename = f"{outdir}/attributes_diagram_{labels[0]}_{labels[1]}.png"
        #idx is 'index0' so it's fine here
        rel_with_obs_and_fcst_cts(obs_cts[idx], fcst_cts[idx], bins[idx], outfilename,
                                  obs_cts2=obs_cts2[idx], fcst_cts2=fcst_cts2[idx], labels=labels)


#---------------------------------------------------------------------------------------------------------

def parsed_reliability_by(outdir, figtype="month", index=0):
    '''
    For making one reliability diagram with a curve for each month or hour.
    - outdir should contain directories like "month01-02", "month10", "hour22", etc.
    - index is for the ith model output. Usually there is only 1 (index=0) or 2 (index=1).
    '''

    assert figtype == "month" or figtype == "hour"

    cmap = plt.get_cmap("hsv")
    dirs = np.sort(glob.glob(f"{outdir}/{figtype}??") +
                   glob.glob(f"{outdir}/{figtype}??-??"))
    ndirs = len(dirs)

    # month reliability diagram
    figure_object, axes_object = plt.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )
    perfect_x_coords = np.array([0, 1], dtype=float)
    perfect_y_coords = perfect_x_coords + 0.0
    img = axes_object.plot(
        perfect_x_coords,
        perfect_y_coords,
        color=PERFECT_LINE_COLOUR,
        linestyle="dashed",
        linewidth=PERFECT_LINE_WIDTH,
    )
    axes_object.set_xlim([0, 1])
    axes_object.set_ylim([0, 1])

    # plot grid lines
    for ll in [0.2, 0.4, 0.6, 0.8]:
        axes_object.plot([ll, ll], [0, 1], linewidth=0.5, linestyle=":", color="gray")
        axes_object.plot([0, 1], [ll, ll], linewidth=0.5, linestyle=":", color="gray")

    if figtype == "month":
        inds4colors = np.linspace(0, 1, ndirs)
        divisions = ndirs
    if figtype == "hour":
        inds4colors = np.linspace(0, 1, ndirs)
        divisions = ndirs
    for ii,thedir in enumerate(dirs):

        # Organize and plot the reliability curves

        val_color = cmap(inds4colors[ii])

        scores = pickle.load(open(f"{thedir}/eval_results.pkl", "rb"))
        idx = f'index{index}'
        fcst_cts = []
        obs_cts = []
        bins = []
        for key, val in scores.items():
            if 'obsct' in key and idx in key:
                prob = key.split('_')[0][-2:]
                obs_cts.append(val)
                bins.append((float(prob) - 2.5) / 100)
            if 'fcstct' in key and idx in key:
                prob = key.split('_')[0][-2]
                fcst_cts.append(val)
        # Sort values in ascending order of probability labels
        fcst_cts = np.array(fcst_cts)
        obs_cts = np.array(obs_cts)
        bins = np.array(bins)
        fcst_cts = fcst_cts[bins.argsort()]
        obs_cts = obs_cts[bins.argsort()]
        bins = np.sort(bins)

        rel = obs_cts / fcst_cts

        axes_object.plot(
            bins,
            rel,
            color=val_color,
            linewidth=1,
        )

    axes_object.tick_params(axis="x", labelsize=FONT_SIZE)
    axes_object.tick_params(axis="y", labelsize=FONT_SIZE)
    axes_object.set_xlabel("Forecast probability", fontsize=FONT_SIZE)
    axes_object.set_ylabel("Conditional event frequency", fontsize=FONT_SIZE)
    axes_object.set_title(f"Reliability curve by {figtype}")

    # This allows us to obtain an "image" for colorbar().
    # See https://stackoverflow.com/questions/8342549/matplotlib-add-colorbar-to-a-sequence-of-line-plots
    norm = mpl.colors.Normalize(vmin=1 / divisions, vmax=1)
    cmap_SM = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.hsv)
    cmap_SM.set_array([])

    cbar_axes = figure_object.add_axes(
        [0.01, -0.08, 0.9, 0.05]
    )  # left bottom width height
    cbar = figure_object.colorbar(
        cmap_SM,
        orientation="horizontal",
        pad=0.00,
        drawedges=0,
        ax=axes_object,
        ticks=np.arange(1 / divisions, 1.0001, 1 / divisions),
        cax=cbar_axes,
    )
    if figtype == "month":
        month_names = []
        for dd in dirs:
            month_names.append(get_month_name(dd.split('month')[1]))
        cbar.ax.set_xticklabels(month_names, rotation="vertical")
    elif figtype == "hour": #FIXME: 24 hours assumed
        cbar.ax.set_xticklabels(
            [str(hh).zfill(2) for hh in np.arange(24, dtype=int)], rotation="vertical"
        )
    cbar.ax.tick_params(size=0, labelsize=10)
    plt.savefig(f"{outdir}/{figtype}_reliability_index{index}.png", bbox_inches="tight", dpi=300)
    print(f"Saved {outdir}/{figtype}_reliability_index{index}.png")
