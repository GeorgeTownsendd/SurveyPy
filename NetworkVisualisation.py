import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
import matplotlib.patches as patches

from AdjustmentNetwork import *
from LeastSquaresAdjustment import adjust_network


def plot_adjustment_process(network, action='show', custom_xaxislimit=False, custom_yaxislimit=False):
    plt.figure(figsize=(8, 8))
    plt.rcParams.update({'font.size': 16, 'font.family': 'sans-serif'})

    # Generate a color map based on the number of stations
    colors = cm.get_cmap('tab10', len(network.stations))

    for i, station in enumerate(network.stations):
        color = colors(i)

        # Plot the original position with barely visible opacity
        plt.scatter(station.original_coordinates[0], station.original_coordinates[1],
                    c=[color], marker='x', s=200, alpha=0.3)

        # Plot the adjustment path
        if not station.fixed:
            coordinates = np.array([station.original_coordinates] + station.prior_coordinates + [station.coordinates])
            n_steps = len(coordinates)-1

            n_subsegments = 3  # Increase this number for smoother transitions

            # Calculate cumulative distance
            cumulative_distance = [0]
            for j in range(n_steps):
                start_point = coordinates[j]
                end_point = coordinates[j + 1]
                distance = np.linalg.norm(end_point - start_point)
                cumulative_distance.append(cumulative_distance[-1] + distance)

            total_distance = cumulative_distance[-1]

            for j in range(n_steps):
                start_point = coordinates[j]
                end_point = coordinates[j + 1]
                previous_point = start_point

                for k in range(n_subsegments + 1):
                    t_subseg = k / n_subsegments
                    t = (cumulative_distance[j] + t_subseg * (
                                cumulative_distance[j + 1] - cumulative_distance[j])) / total_distance
                    intermediate_point = start_point * (1 - t_subseg) + end_point * t_subseg
                    line_alpha = np.clip(0.1 + 0.9 * t**5, 0, 1)
                    line_color = np.array(color)
                    line_color[3] = line_alpha  # Set the alpha channel of the color
                    dot_size = np.clip(60 * (1 - t), 10, 60)

                    if k > 0:
                        plt.plot([previous_point[0], intermediate_point[0]], [previous_point[1], intermediate_point[1]],
                                 linestyle='-', alpha=line_alpha, c=line_color, linewidth=2)

                    previous_point = intermediate_point

                # Plot mini markers at the end of each adjustment vector with the same opacity as the line
                plt.scatter(end_point[0], end_point[1], c=[color], marker='o', s=dot_size, alpha=line_alpha)

        # Plot the final position
        plt.scatter(station.coordinates[0], station.coordinates[1],
                    c=[color], marker='o', s=150)

    plt.xlabel('Easting', fontsize=20)
    plt.ylabel('Northing', fontsize=20)

    # Create a custom legend icon
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=f'{station.identifier}',
                              markerfacecolor=colors(i), markersize=15,
                              markeredgewidth=1.5, markeredgecolor=colors(i)) for i, station in enumerate(network.stations)]

    plt.legend(handles=legend_elements, loc='upper right', fontsize=14, title="Stations")
    plt.title('Adjustment Process', fontsize=40)
    marker_explanation = "Markers:\n  Cross (x) - Start\n  Large Circle (o) - End"
    plt.annotate(marker_explanation, xy=(0.75, -0.15), xycoords='axes fraction', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), ha='left', va='bottom')

    plt.grid()

    if custom_xaxislimit:
        plt.xlim(custom_xaxislimit)
    if custom_yaxislimit:
        plt.ylim(custom_yaxislimit)
    if not custom_xaxislimit and not custom_yaxislimit:
        plt.tight_layout()

    if action == 'show':
        plt.show()
    elif action == 'return':
        return plt.gcf()


def generate_transition_frames(dataset_name, value_start, value_stop, n_steps, value_index, value_type='coordinates_fixed'):
    for step_value in np.linspace(value_start, value_stop, n_steps):
        if value_type == 'coordinates_fixed':
            network = load_dataset(dataset_name=dataset_name)
            network.fixed_stations[value_index].coordinates += np.array([step_value, step_value])
            adjust_network(network, max_iterations=50)

        elif value_type == 'weight':
            network = load_dataset(dataset_name=dataset_name)
            network.P[value_index, value_index] = step_value
            adjust_network(network, max_iterations=50)

            #self.P = np.linalg.inv(sigma * np.diag([obs.stddev ** 2 for obs in self.observations]))

        plot = plot_adjustment_process(network, action='return')
        plot.savefig(f'frames/image{step_value}.png')


def plot_error_ellipses(network):
    stations = network.unknown_stations
    ellipses = [[s.EE_major_semi_axis * 1000, s.EE_minor_semi_axis * 1000, s.EE_orientation] for s in stations]
    # Convert to mm

    # Find the maximum semi-major axis
    max_a = max(ellipse[0] for ellipse in ellipses)

    # Calculate grid dimensions
    grid_size = int(np.ceil(np.sqrt(len(ellipses))))
    fig, axs = plt.subplots(grid_size, grid_size)

    for i, ax in enumerate(axs.flatten()):
        if i < len(ellipses):
            a, b, theta = ellipses[i]

            ellipse = patches.Ellipse((0, 0), 2*a, 2*b, angle=90-theta, fill=False) #theta-90 to convert from postive x axis to positive y axis angle measurement
            ax.add_patch(ellipse)
            ax.set_aspect('equal')

            ax.set_xlim(-max_a, max_a)
            ax.set_ylim(-max_a, max_a)

            ax.set_title('Station ' + str(stations[i].identifier))

            # Only label the outer axes
            if i % grid_size == 0:  # first column
                ax.set_ylabel('Northing (mm)')
            if i // grid_size == grid_size - 1:  # last row
                ax.set_xlabel('Easting (mm)')
        else:
            ax.axis('off')  # hide extra subplots

    plt.tight_layout()
    plt.show()


def visualize_adjustments(dataset_name, station_names, side_length=100, coord_range=100, max_iterations=10):
    if isinstance(coord_range, tuple):
        dx_list = np.linspace(coord_range[0][0], coord_range[0][1], side_length)
        dy_list = np.linspace(coord_range[1][0], coord_range[1][1], side_length)
    else:
        dx_list = np.linspace(-coord_range, coord_range, side_length)
        dy_list = np.linspace(-coord_range, coord_range, side_length)

    grid_size = int(np.ceil(np.sqrt(len(station_names))))
    fig, axs = plt.subplots(grid_size, grid_size)

    for i, ax in enumerate(axs.flatten()):
        if i < len(station_names):
            station = station_names[i]
            grid = np.zeros((side_length, side_length))

            for x, dx in enumerate(dx_list):
                for y, dy in enumerate(dy_list):
                    network = load_dataset(dataset_name=dataset_name, adjust_station=(station, dx, dy))
                    iterations = adjust_network(network, max_iterations=max_iterations, return_type='n_iterations')
                    if iterations == 'not_converged':
                        iterations = -1
                    grid[x, y] = iterations

            ax.imshow(grid, extent=[dx_list[0], dx_list[-1], dy_list[0], dy_list[-1]])
            ax.set_title(f'Station {station}')
            if i % grid_size == 0:
                ax.set_ylabel('Y (m)')
            if i // grid_size == grid_size - 1:
                ax.set_xlabel('X (m)')
        else:
            ax.axis('off')

    plt.tight_layout()
    plt.show()


def visualize_network(network, show_error_ellipse=True):
    """
    Visualize a network of stations. Stations are represented as either circles or squares,
    while different line styles represent different types of measurements between them.

    Parameters:
    network: The network of stations to visualize.
    show_error_ellipse: Whether to show the error ellipse for each station. Defaults to True.
    """
    all_stations = network.unknown_stations + network.fixed_stations

    # Calculate the ranges of your data in x and y
    x_range = max(st.coordinates[0] for st in all_stations) - min(st.coordinates[0] for st in all_stations)
    y_range = max(st.coordinates[1] for st in all_stations) - min(st.coordinates[1] for st in all_stations)

    # Calculate the aspect ratio of your data and set the figure size accordingly
    data_aspect_ratio = x_range / y_range
    fig_height = 6  # height in inches, fixed as per your requirements
    fig_width = fig_height * data_aspect_ratio  # adjust the width according to the data aspect ratio
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Create dictionaries to store station legend elements and their colors
    station_legend_elements = []
    color_dict = {}

    # Plot stations and create station legend elements
    for station, marker in zip([network.fixed_stations, network.unknown_stations], ['s', 'o']):
        for st in station:
            plot, = ax.plot(*st.coordinates, marker, markersize=10, zorder=5)
            color_dict[st.identifier] = plot.get_color()
            station_legend_elements.append(
                Line2D([0], [0], linestyle='', marker=marker, color=color_dict[st.identifier],
                       markerfacecolor=color_dict[st.identifier], markersize=10, label=st.identifier))

    # Create an invisible point to ensure enough space for legends
    min_x, max_x = min(st.coordinates[0] for st in all_stations), max(st.coordinates[0] for st in all_stations)
    avg_y = np.mean([st.coordinates[1] for st in all_stations])
    ax.plot(max_x + 0.25 * (max_x - min_x), avg_y, alpha=0)

    # Store observations between stations
    observation_dict = {}  # {(station1, station2): [observation_type1, observation_type2, ...]}
    for obs in network.observations:
        if isinstance(obs, SingleTargetObservation):
            key = tuple(sorted([obs.st_obs.identifier, obs.st_tar.identifier]))
            observation_dict.setdefault(key, []).append(obs.observation_type)
        elif isinstance(obs, MultiTargetObservation):
            key1 = tuple(sorted([obs.st_obs.identifier, obs.st_tar1.identifier]))
            key2 = tuple(sorted([obs.st_obs.identifier, obs.st_tar2.identifier]))
            observation_dict.setdefault(key1, []).append(obs.observation_type)
            observation_dict.setdefault(key2, []).append(obs.observation_type)

    for (station1_id, station2_id), observation_types in observation_dict.items():
        station1 = next(station for station in all_stations if station.identifier == station1_id)
        station2 = next(station for station in all_stations if station.identifier == station2_id)

        # If both distance and direction are present, plot a dashed red arrow
        if 'distance' in observation_types and 'direction' in observation_types:
            line_style = 'dashed'
            line_color = 'red'
            ax.annotate("", xy=station1.coordinates, xytext=station2.coordinates,
                        arrowprops=dict(arrowstyle="-|>", mutation_scale=30, fc=line_color, color=line_color, lw=1.5,
                                        linestyle=line_style))
        # Else plot the measurements separately
        else:
            # Plot distance
            if 'distance' in observation_types:
                line_style = 'dotted'
                line_color = 'grey'
                ax.annotate("", xy=station1.coordinates, xytext=station2.coordinates,
                            arrowprops=dict(arrowstyle="-", mutation_scale=30, fc=line_color, color=line_color, lw=1.5,
                                            linestyle=line_style, connectionstyle="arc3,rad=0.15"))
            # Plot direction
            if 'direction' in observation_types:
                line_style = 'solid'
                line_color = 'red'
                ax.annotate("", xy=station1.coordinates, xytext=station2.coordinates,
                            arrowprops=dict(arrowstyle="-|>", mutation_scale=30, fc=line_color, color=line_color,
                                            lw=1.5, linestyle=line_style))
        # Plot azimuth
        if 'azimuth' in observation_types:
            line_style = 'solid'
            line_color = 'blue'
            ax.annotate("", xy=station1.coordinates, xytext=station2.coordinates,
                        arrowprops=dict(arrowstyle="-|>", mutation_scale=30, fc=line_color, color=line_color, lw=1.5,
                                        linestyle=line_style))

    # Plot error ellipses at stations
    if show_error_ellipse:
        for station in network.unknown_stations:
            a, b = station.EE_major_semi_axis * 1000, station.EE_minor_semi_axis * 1000  # Convert to mm
            theta = station.EE_orientation
            ellipse = patches.Ellipse(station.coordinates, 2 * a, 2 * b, angle=90 - theta, fill=False, zorder=5)
            ax.add_patch(ellipse)

    # Set figure properties and show legends
    ax.set_aspect('equal')
    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')
    plt.title(f'Network Visualization')

    # Create and add legends to the figure
    station_type_legend_elements = [
        Line2D([0], [0], linestyle='', marker=marker, color='black', markerfacecolor='black',
               markersize=10, label=label) for marker, label in zip(['s', 'o'], ['Fixed', 'Unknown'])]
    line_legend_elements = [Line2D([0], [0], color=color, linestyle=style, label=label) for color, style, label in
                            zip(['grey', 'red', 'red', 'blue'], ['dotted', 'solid', 'dashed', 'solid'], ['Dist', 'Dir', 'Dir+Dist', 'Azm'])]

    ax.add_artist(ax.legend(handles=station_type_legend_elements, loc='upper right', bbox_to_anchor=[1, 1]))
    ax.add_artist(ax.legend(handles=line_legend_elements, loc='upper right', bbox_to_anchor=[1, 0.8925]))
    ax.add_artist(ax.legend(handles=station_legend_elements, loc='upper right', bbox_to_anchor=[1, 0.7],title='Station Name'))

    plt.tight_layout()
    plt.show()