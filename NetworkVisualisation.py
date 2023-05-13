import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
import matplotlib.patches as patches

from AdjustmentNetwork import load_dataset
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
    ellipses = [[s.EE_major_axis * 1000, s.EE_minor_axis * 1000, s.EE_orientation] for s in stations]
    # Convert to mm

    # Find the maximum semi-major axis
    max_a = max(ellipse[0] for ellipse in ellipses)

    # Calculate grid dimensions
    grid_size = int(np.ceil(np.sqrt(len(ellipses))))
    fig, axs = plt.subplots(grid_size, grid_size)

    for i, ax in enumerate(axs.flatten()):
        if i < len(ellipses):
            a, b, theta = ellipses[i]

            ellipse = patches.Ellipse((0, 0), 2*a, 2*b, angle=theta, fill=False)
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