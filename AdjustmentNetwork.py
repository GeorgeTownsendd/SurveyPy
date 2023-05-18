import numpy as np
import pandas as pd

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

    def sort_observations_by_type(self):
        new_observations = []
        for obs_type in ['distance', 'azimuth', 'direction', 'angle']:
            new_observations += [x for x in self.observations if x.observation_type == obs_type]

        self.observations = new_observations

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
        self.identifier = identifier

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

        print(Sn, Se)

        top = 2 * Sne#-0.000017122510207663#2 * Sne
        bottom = Sn-Se#-0.000003252305863673#Sn-Se

        #e = 6.5046117274020825e-06
        #print('e', e)

        #if self.identifier == 6:
        #    print('Sn, Se', (Sn, Se))
        #    print('True: ', -0.000003252305863673)
        #    print('Calc: ', Sn-Se)
        #    print('Err:', e)
            #print(top, bottom)
            #print(f'Error: {(2 * Sne) - -0.000017122510207663}, {(Sn - Se) - -0.000003252305863673}')
            #print(Sn, Se, Sn-Se)

        #bottom -= 0

        self.EE_orientation = ((np.degrees(np.arctan2(top,bottom)))%360)/2#np.degrees(quad_check(Sn - Se, 2 * Sne))
        print(self.EE_orientation)
        x = np.degrees(np.arctan2(2 * Sne, Sn - Se + np.sqrt((Sn - Se)**2 + 4 * (Sne**2)))) % 360
        print(x)


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