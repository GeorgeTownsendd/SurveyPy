import numpy as np
from AdjustmentNetwork import *


def quad_check(x, y):
    out = np.arctan(x/y)
    if x > 0 and y > 0:
        return out
    elif x > 0 and y < 0:
        return out + np.pi
    elif x < 0 and y < 0:
        return out + np.pi
    else:
        return out + (2 * np.pi)


def compute_design_matrix(network):
    stations = network.stations
    observations = network.observations

    unknown_stations = network.unknown_stations
    unknown_stations_names = network.unknown_station_names

    num_observations = len(observations)
    num_unknown_coordinates = 2 * network.n_unknown_stations

    A = np.zeros((num_observations, num_unknown_coordinates + network.n_directed_stations))

    for obs_n, observation in enumerate(observations):
        if isinstance(observation, SingleTargetObservation):
            station_obs = observation.st_obs
            station_target = observation.st_tar

            dx = station_target.coordinates[0] - station_obs.coordinates[0]
            dy = station_target.coordinates[1] - station_obs.coordinates[1]

            bearing_ij = quad_check(dx, dy)
            distance_ij = np.sqrt(dx**2 + dy**2)
            c_ij = np.cos(bearing_ij)
            s_ij = np.sin(bearing_ij)
            a_ij = (np.sin(bearing_ij) / distance_ij) * RHO
            b_ij = (-np.cos(bearing_ij) / distance_ij) * RHO

            if observation.observation_type == 'distance':
                if not station_obs.fixed:
                    index = unknown_stations_names.index(station_obs.identifier)
                    A[obs_n, 2 * index] = -s_ij
                    A[obs_n, 2 * index + 1] = -c_ij

                if not station_target.fixed:
                    index = unknown_stations_names.index(station_target.identifier)
                    A[obs_n, 2 * index] = s_ij
                    A[obs_n, 2 * index + 1] = c_ij

            elif observation.observation_type == 'azimuth':
                if not station_obs.fixed:
                    index = unknown_stations_names.index(station_obs.identifier)
                    A[obs_n, 2 * index] = b_ij
                    A[obs_n, 2 * index + 1] = a_ij

                if not station_target.fixed:
                    index = unknown_stations_names.index(station_target.identifier)
                    A[obs_n, 2 * index] = -b_ij
                    A[obs_n, 2 * index + 1] = -a_ij

            elif observation.observation_type == 'direction':
                if not station_obs.fixed:
                    index = unknown_stations_names.index(station_obs.identifier)
                    A[obs_n, 2 * index] = b_ij
                    A[obs_n, 2 * index + 1] = a_ij

                if not station_target.fixed:
                    index = unknown_stations_names.index(station_target.identifier)
                    A[obs_n, 2 * index] = -b_ij
                    A[obs_n, 2 * index + 1] = -a_ij

                # Add Mu0 correction
                mu_column = num_unknown_coordinates + network.directed_station_names.index(station_obs.identifier)
                A[obs_n, mu_column] = 1

        elif isinstance(observation, MultiTargetObservation):
            station_obs = observation.st_obs #j
            station_target1 = observation.st_tar1 #i
            station_target2 = observation.st_tar2 #k

            dx_ji = station_target1.coordinates[0] - station_obs.coordinates[0]
            dy_ji = station_target1.coordinates[1] - station_obs.coordinates[1]

            bearing_ji = quad_check(dx, dy)
            distance_ji = np.sqrt(dx_ji**2 + dy_ji**2)
            c_ji = np.cos(bearing_ji)
            s_ji = np.sin(bearing_ji)
            a_ji = (observation.observation_value * RHO * s_ji) / distance_ji
            b_ji = (-(observation.observation_value * RHO) * c_ji) / distance_ji


            dx_jk = station_target1.coordinates[0] - station_obs.coordinates[0]
            dy_jk = station_target1.coordinates[1] - station_obs.coordinates[1]

            bearing_jk = quad_check(dx, dy)
            distance_jk = np.sqrt(dx_jk**2 + dy_jk**2)
            c_jk = np.cos(bearing_jk)
            s_jk = np.sin(bearing_jk)
            a_jk = (observation.observation_value * RHO * s_jk) / distance_jk
            b_jk = (-(observation.observation_value * RHO) * c_jk) / distance_jk

            # Assuming 'angle' observation_type is the one with two target stations
            if observation.observation_type == 'angle':
                if not station_obs.fixed:
                    index = unknown_stations_names.index(station_obs.identifier)
                    A[obs_n, 2 * index] = b_jk - b_ji * RHO
                    A[obs_n, 2 * index + 1] = a_jk - a_ji * RHO

                if not station_target1.fixed:
                    index = unknown_stations_names.index(station_target1.identifier)
                    A[obs_n, 2 * index] = b_ji * RHO
                    A[obs_n, 2 * index + 1] = a_ji * RHO

                if not station_target2.fixed:
                    index = unknown_stations_names.index(station_target2.identifier)
                    A[obs_n, 2 * index] = -b_jk * RHO
                    A[obs_n, 2 * index + 1] = -a_jk * RHO

    return A


def compute_observation_vector(observations):
    num_observations = len(observations)
    L = np.zeros(num_observations)

    for obs_n, observation in enumerate(observations):
        if isinstance(observation, SingleTargetObservation):
            station_obs = observation.st_obs
            station_target = observation.st_tar

            dx_ij = station_target.coordinates[0] - station_obs.coordinates[0]
            dy_ij = station_target.coordinates[1] - station_obs.coordinates[1]

            bearing_ij = quad_check(dx_ij, dy_ij)
            distance_ij = np.sqrt(dx_ij ** 2 + dy_ij ** 2)

            if observation.observation_type == 'distance':
                L[obs_n] = distance_ij - observation.observation_value

            elif observation.observation_type == 'azimuth':
                L[obs_n] = (bearing_ij - observation.observation_value) * RHO

            elif observation.observation_type == 'direction':
                estimated_direction = bearing_ij + station_obs.orientation - observation.observation_value
                if estimated_direction > np.pi:
                    estimated_direction -= 2 * np.pi
                elif estimated_direction < -np.pi:
                    estimated_direction += 2 * np.pi
                L[obs_n] = estimated_direction * RHO

        elif isinstance(observation, MultiTargetObservation):
            station_obs = observation.st_obs
            station_target1 = observation.st_tar1
            station_target2 = observation.st_tar2

            if observation.observation_type == 'angle':
                angle1 = quad_check(station_target1.coordinates[0] - station_obs.coordinates[0], station_target1.coordinates[1] - station_obs.coordinates[1])
                angle2 = quad_check(station_target2.coordinates[0] - station_obs.coordinates[0], station_target2.coordinates[1] - station_obs.coordinates[1])
                estimated_angle = (angle2 - angle1) - observation.observation_value
                L[obs_n] = estimated_angle * RHO

    return L


def adjust_network(network, max_iterations=1, return_type='number_iterations'):
    network.adjustment_state = 'in_progress'
    P = network.P
    p = np.array([obs.observation_value for obs in network.observations])

    converged = False
    iterations = 0
    while not converged:
        iterations += 1
        A = compute_design_matrix(network)
        L = compute_observation_vector(network.observations)

        N = np.linalg.inv(A.T @ P @ A)
        Delta = -N @ A.T @ P @ L

        for n, data  in enumerate(zip(network.directed_stations, Delta[network.n_unknown_coordinates:])):
            station, orientation_update = data

            station.orientation += orientation_update / RHO
            station.orientation_variance = N[network.n_unknown_stations+n, network.n_unknown_stations+n] / RHO
            station.orientation_std = np.sqrt(station.orientation_variance)

        up = Delta[:network.n_unknown_coordinates].reshape((network.n_unknown_stations, 2))
        for n, data  in enumerate(zip(up, network.unknown_stations)):
            shift, station = data
            new_station_coordinates = station.coordinates + shift
            station.update_coordinates(new_station_coordinates)

            station.coordinate_variance = np.array([N[(n*2), (n*2)], N[(n*2)+1, (n*2)+1]])
            station.coordinate_covariance = N[(n*2)+1, (n*2)]
            station.coordinate_std = np.sqrt(station.coordinate_variance)
            station.determine_error_ellipse()


        vhat = A @ Delta + L
        vhat[network.n_distance_obs:] /= RHO
        variance = vhat.T @ P @ vhat / 2

        #print(f'var: {variance} (i={iterations})')

        if np.max(np.abs(Delta[:network.n_unknown_coordinates])) < 10 ** -3:
            converged = True
            network.adjustment_state = 'adjusted_converged'
        elif np.max(np.abs(Delta[:network.n_unknown_coordinates])) > 1000000 or iterations == max_iterations:
            if iterations == max_iterations:
                network.adjustment_state = 'adjusted_limited'
            else:
                network.adjustment_state = 'adjusted_exploded'
            break

    network.final_adjustment_state = {
        'A': A,
        'L': L,
        'P': P,
        'N': N,
        'V': vhat
    }

    print(network.adjustment_state, f' - {iterations} iterations')
    network.perform_global_model_test()

    if return_type == 'final_matrices':
        return A, L, vhat
    elif return_type == 'n_iterations':
        if network.adjustment_state == 'adjusted_converged':
            return iterations
        else:
            return 'not_converged'
    elif return_type == 'adjustment_state':
        return network.adjustment_state
