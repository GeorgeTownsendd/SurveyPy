import numpy as np
import pandas as pd
from scipy.stats.distributions import chi2

RHO = 206264.8

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


def decdeg2dms(deg):
    """Convert decimal degrees to degrees, minutes, seconds"""
    d = int(deg)
    md = abs(deg - d) * 60
    m = int(md)
    sd = (md - m) * 60
    return d, m, sd


class AdjustmentNetwork:
    def __init__(self, stations, observation_order='original'):
        self.adjustment_state = 'original'
        self.stations = sorted(stations, key=lambda x: x.identifier)
        self.observations = np.concatenate([st.observations for st in stations if isinstance(st, ObservationStation)])
        self.n_observations = len(self.observations)
        if observation_order == 'type':
            self.sort_observations_by_type()
        elif observation_order == 'original':
            self.observations = sorted(self.observations, key = lambda x:x.observation_n)
        self.angular_observation_mask = [False if obs.observation_type == 'distance' else True for obs in self.observations]

        self.n_distance_obs = sum([1 for x in self.observations if x.observation_type == 'distance'])
        self.n_azimuth_obs = sum([1 for x in self.observations if x.observation_type == 'azimuth'])
        self.n_direction_obs = sum([1 for x in self.observations if x.observation_type == 'direction'])
        self.n_angle_obs = sum([1 for x in self.observations if x.observation_type == 'angle'])

        self.fixed_stations = [st for st in self.stations if st.fixed]
        self.fixed_station_names = [st.identifier for st in self.fixed_stations]
        self.n_fixed_stations = len(self.fixed_stations)

        self.unknown_stations = [st for st in self.stations if not st.fixed]
        self.unknown_station_names = [st.identifier for st in self.unknown_stations]
        self.n_unknown_stations = len(self.unknown_stations)
        self.n_unknown_coordinates = self.n_unknown_stations * 2 #\\TODO implement independant fixing of easting/northing

        self.observation_stations = [st for st in self.stations if isinstance(st, ObservationStation)]
        self.observation_station_names = [st.identifier for st in self.observation_stations]
        self.n_observation_stations = len(self.observation_stations)

        self.directed_stations = [st for st in self.observation_stations if 'direction' in [obs.observation_type for obs in st.observations]]
        self.directed_station_names = [st.identifier for st in self.directed_stations]
        self.n_directed_stations = len(self.directed_stations)

        sigma = 1
        self.P = np.linalg.inv(sigma * np.diag([obs.stddev ** 2 for obs in self.observations]))
        self.L = np.array([obs.observation_value for obs in self.observations])

        self.redundant_observations = len(self.observations) - (len(self.unknown_stations) * 2) - len(self.directed_stations)
        self.final_adjustment_state = 'unadjusted_network'

    def sort_observations_by_type(self):
        new_observations = []
        for obs_type in ['distance', 'azimuth', 'direction', 'angle']:
            new_observations += [x for x in self.observations if x.observation_type == obs_type]

        self.observations = new_observations

    def determine_adjusted_distance_and_bearing(self, p1, p2):
        if p1.fixed or p2.fixed:
            print('Error: fixed stations')

        N = self.final_adjustment_state['N']
        p1 = [p for p in self.unknown_stations if str(p.identifier) == str(p1.identifier)][0]
        p1_i = self.unknown_station_names.index(p1.identifier) * 2
        p2 = [p for p in self.unknown_stations if str(p.identifier) == str(p2.identifier)][0]
        p2_i = self.unknown_station_names.index(p2.identifier) * 2

        dx = p2.coordinates[0] - p1.coordinates[0]
        dy = p2.coordinates[1] - p1.coordinates[1]

        bearing_ij = quad_check(dx, dy)
        distance_ij = np.sqrt(dx ** 2 + dy ** 2)

        Ex = np.zeros((4, 4))

        # Fill the matrix
        Ex[0:2, 0:2] = N[p1_i:p1_i + 2, p1_i:p1_i + 2]
        Ex[2:4, 2:4] = N[p2_i:p2_i + 2, p2_i:p2_i + 2]

        # Fill the off-diagonal blocks
        Ex[0:2, 2:4] = N[p1_i:p1_i + 2, p2_i:p2_i + 2]
        Ex[2:4, 0:2] = N[p2_i:p2_i + 2, p1_i:p1_i + 2]

        J = np.zeros((2, 4))

        J[0, 0] = -dx / distance_ij
        J[0, 1] = -dy / distance_ij
        J[0, 2] = dx / distance_ij
        J[0, 3] = dy / distance_ij
        J[1, 0] = -dy / (distance_ij ** 2)
        J[1, 1] = dx / (distance_ij ** 2)
        J[1, 2] = dy / (distance_ij ** 2)
        J[1, 3] = -dx / (distance_ij ** 2)

        Ey = J @ Ex @ J.T

        distance_std = np.sqrt(Ey[0, 0]) * 1000  # mm
        bearing_std = np.sqrt(Ey[1, 1])  # seconds

        return {'value': (distance_ij, bearing_ij),
                'std': (distance_std, bearing_std)}

    def perform_global_model_test(self, confidence_level=0.99):
        threshold = chi2.ppf(confidence_level, df=self.redundant_observations)

        V = self.final_adjustment_state['V']
        Ginv = np.linalg.inv(self.final_adjustment_state['G'])

        value = (V.T @ Ginv @ V) / self.redundant_observations

        test_passed = value < threshold
        test_passed_string = 'Test passed' if test_passed else 'Test failed'

        print('Estimated Variance Factor (VF): ' + str(round(value, 3)))
        print('Chi2 Threshold: ' + str(round(threshold, 3)))
        print(f'{test_passed_string} for a={confidence_level}')



    def __repr__(self):
        return f"AdjustmentNetwork(stations:{','.join([st.identifier for st in self.stations])})"


class Observation:
    def __init__(self):
        self.observation_type = 'uninitialised'
        self.observation_n = -1
        self.involved_stations = []

    def __repr__(self):
        return f"Observation({self.observation_type},{'-'.join([str(x) for x in self.involved_stations])})"


class SingleTargetObservation(Observation):
    def __init__(self, observation_station, target_station, observation_value, observation_type, stddev):
        self.st_obs = observation_station
        self.st_tar = target_station
        self.involved_stations = [self.st_obs, self.st_tar]

        self.observation_value = observation_value
        self.observation_type = observation_type
        self.stddev = stddev


class MultiTargetObservation(Observation):
    def __init__(self, observation_station, target_station1, target_station2, observation_value, observation_type, stddev):
        self.st_obs = observation_station
        self.st_tar1 = target_station1
        self.st_tar2 = target_station2
        self.involved_stations = [self.st_obs, self.st_tar1, self.st_tar2]
        print('Involved stations: ', self.involved_stations)

        self.observation_value = observation_value
        self.observation_type = observation_type
        self.stddev = stddev

class Station:
    def __init__(self, coordinates, fixed=False, identifier=None):
        self.original_coordinates = np.array(coordinates)
        self.coordinates = np.array(coordinates)
        self.coordinate_variance = 'unknown'
        self.coordinate_covariance = 'unknown'
        self.coordinate_std = 'unknown'
        self.EE_major_semi_axis = 'unknown'
        self.EE_minor_semi_axis = 'unknown'
        self.EE_orientation = 'unknown'

        self.fixed = fixed
        self.identifier = str(identifier)

        self.prior_coordinates = []
        self.prior_adjustments = []

    def __repr__(self):
        return f"Station(identifier={self.identifier})"

    def update_coordinates(self, new_coordinates):
        if self.fixed:
            print('Warning: Updating fixed coordinates!')

        adjustment_vector = new_coordinates - self.coordinates
        self.prior_adjustments.append(adjustment_vector)

        self.prior_coordinates.append(self.coordinates)
        self.coordinates = new_coordinates

    def determine_error_ellipse(self):
        Sn, Se = self.coordinate_variance
        Sne = self.coordinate_covariance

        self.EE_major_semi_axis = np.sqrt(0.5 * (Sn + Se + np.sqrt((Sn - Se) ** 2 + 4 * (Sne ** 2))))
        self.EE_minor_semi_axis = np.sqrt(0.5 * (Sn + Se - np.sqrt((Sn - Se) ** 2 + 4 * (Sne ** 2))))

        excel_top = 2 * Sne
        excel_bottom = Sn-Se

        excel_bottom, excel_top = excel_top, excel_bottom

        angle = np.arctan2(excel_bottom, excel_top)
        if angle < np.pi:
            angle += np.pi
            angle = -angle

        self.EE_orientation = (np.degrees(angle) % 360) / 2.0


class ObservationStation(Station):
    def __init__(self, coordinates, observations, fixed=False, identifier=None, orientation_prior='estimate'):
        super().__init__(coordinates, fixed=fixed, identifier=identifier)
        self.orientation_variance = 'unknown'
        self.orientation_std = 'unknown'

        self.observations = observations
        if 'direction' in [x.observation_type for x in self.observations]:
            if orientation_prior == 'estimate':
                self.orientation = 'uninitialized'
            else:
                self.orientation = orientation_prior
        else:
            self.orientation = None

    def __repr__(self):
        return f"Station(identifier={self.identifier}, orientation={self.orientation})"

    def estimate_orientation_correction(self):
        dir_obs = [x for x in self.observations if x.observation_type == 'direction'][0]
        dx = dir_obs.st_tar.coordinates[0] - dir_obs.st_obs.coordinates[0]
        dy = dir_obs.st_tar.coordinates[1] - dir_obs.st_obs.coordinates[1]
        bearing_ij = quad_check(dx, dy)

        self.orientation = dir_obs.observation_value - bearing_ij


def load_dataset(dataset_name, return_type='network', adjust_station=False):
    station_input_file = f'network_inputs/{dataset_name}/station_input.csv'
    observation_input_file = f'network_inputs/{dataset_name}/observation_input.csv'

    # Load observation data
    observation_df = pd.read_csv(observation_input_file)
    observations = []
    occupied_stations = set()
    for _, row in observation_df.iterrows():
        observation_number = int(_)
        station_from = row['station_from']
        station_to = row['station_to']
        observation_value = row['observation_value']
        observation_type = row['observation_type']
        deviation = row['deviation']

        # Add station_from to the set of occupied stations
        occupied_stations.add(station_from)

        if observation_type == 'angle':
            station_to1, station_to2 = station_to.split('_')
            observation = MultiTargetObservation(station_from, station_to1, station_to2, observation_value, observation_type, deviation)
        else:
            # Create SingleTargetObservation instances (with temporary station references)
            observation = SingleTargetObservation(station_from, station_to, observation_value, observation_type, deviation)

        observation.observation_n = observation_number
        observations.append(observation)

    # Load station data
    station_df = pd.read_csv(station_input_file)
    stations = {}
    for _, row in station_df.iterrows():
        point = row['point']
        easting = row['easting']
        northing = row['northing']
        fixed = row['fixed']

        if isinstance(adjust_station, list) or isinstance(adjust_station, tuple):
            print(f'Adjusting {point} by {adjust_station[1]}dx {adjust_station[2]}dy')
            print('adjusting', easting, northing)
            if str(point) == str(adjust_station[0]):
                easting += adjust_station[1]
                northing += adjust_station[2]


        coordinates = np.array([easting, northing])

        # Create ObservationStation instances for occupied stations
        if point in occupied_stations:
            station_observations = [obs for obs in observations if obs.st_obs == point]
            station = ObservationStation(coordinates, station_observations, fixed=fixed, identifier=point)
        else:
            station = Station(coordinates, fixed=fixed, identifier=point)

        # Update station references in SingleTargetObservation instances
        for obs in observations:
            if isinstance(obs, SingleTargetObservation):
                if obs.st_obs == point:
                    obs.st_obs = station
                if obs.st_tar == point:
                    obs.st_tar = station

            elif isinstance(obs, MultiTargetObservation):
                if obs.observation_type == 'angle':
                    if obs.st_obs == point:
                        obs.st_obs = station
                    if obs.st_tar1 == point:
                        obs.st_tar1 = station
                    if obs.st_tar2 == point:
                        obs.st_tar2 = station

        stations[point] = station

    stations = [stations[k] for k in stations.keys()]
    for station in stations:
        if isinstance(station, ObservationStation):
            if [x for x in station.observations if x.observation_type == 'direction']:
                station.estimate_orientation_correction()

    if return_type == 'network':
        return AdjustmentNetwork(stations=stations)
    elif return_type == 'station_obs':
        return stations, observations