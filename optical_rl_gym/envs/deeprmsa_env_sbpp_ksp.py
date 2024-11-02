import gym
import numpy as np

from .rmsa_env_sbpp import RMSASBPPEnv
from .optical_network_env import OpticalNetworkEnv


class DeepRMSASBPPKSPEnv(RMSASBPPEnv):

    def __init__(self, topology=None, j=1,
                 episode_length=1000,
                 mean_service_holding_time=25.0,
                 mean_service_inter_arrival_time=.1,
                 num_spectrum_resources=1000,
                 node_request_probabilities=None,
                 seed=None,
                 k_paths=5,
                 allow_rejection=False):  # Do we need to add another J?
        super().__init__(topology=topology,
                         episode_length=episode_length,
                         load=mean_service_holding_time / mean_service_inter_arrival_time,
                         mean_service_holding_time=mean_service_holding_time,
                         num_spectrum_resources=num_spectrum_resources,
                         node_request_probabilities=node_request_probabilities,
                         seed=seed,
                         k_paths=k_paths,
                         allow_rejection=allow_rejection,
                         reset=False)

        self.j = j
        shape = 1 + 2 * self.topology.number_of_nodes() + (2 * self.j + 3) * self.k_paths * 2  # Doubled
        self.observation_space = gym.spaces.Box(low=0, high=1, dtype=np.uint8, shape=(shape,))
        self.action_space = gym.spaces.Discrete(
            self.k_paths * self.j * 2 + self.reject_action)  # Need to be doubled (modified)
        self.action_space.seed(self.rand_seed)
        self.observation_space.seed(self.rand_seed)
        self.reset(only_counters=False)


    def step(self, action: int):
            working_path, backup_path = shortest_available_path_first_fit(self) #added band here and in function
            if working_path < self.k_paths and backup_path < self.k_paths:
                working_initial_indices, working_lengths = self.get_available_blocks_working(working_path)  #here
                block_working = 0
                backup_initial_indices, backup_lengths = self.get_available_blocks_backup(backup_path) #here
                block_backup = 0
                backup_dpp_initial_indices, backup_dpp_lengths = self.get_available_blocks_working(backup_path) #here
                block_backup = 0
                if (block_working < len(working_initial_indices)) and (block_backup < len(backup_initial_indices)) and (block_backup < len(backup_dpp_initial_indices)): #shared and dedicated
                    return super().step([working_path, working_initial_indices[block_working], backup_path,
                                        backup_initial_indices[block_backup], backup_path, backup_dpp_initial_indices[block_backup]])
                
                if ((block_working < len(working_initial_indices)) and (block_backup < len(backup_dpp_initial_indices))): #shared or dedicated only
                    return super().step([working_path, working_initial_indices[block_working],  backup_path,                                    
                                          backup_dpp_initial_indices[block_backup],  backup_path, backup_dpp_initial_indices[block_backup]])

                if ((block_working < len(working_initial_indices)) and (block_backup < len(backup_initial_indices))): #shared or dedicated only
                    return super().step([working_path, working_initial_indices[block_working], backup_path,
                                        backup_initial_indices[block_backup], backup_path, backup_initial_indices[block_backup]])

                else:
                    return super().step([self.k_paths, self.num_spectrum_resources,self.k_paths, self.num_spectrum_resources,self.k_paths, self.num_spectrum_resources])
            else:
                return super().step([self.k_paths, self.num_spectrum_resources,self.k_paths, self.num_spectrum_resources,self.k_paths, self.num_spectrum_resources])  
    
    def observation(self):
        # observation space defined as in https://github.com/xiaoliangchenUCD/DeepRMSA/blob/eb2f2442acc25574e9efb4104ea245e9e05d9821/DeepRMSA_Agent.py#L384
        source_destination_tau = np.zeros((2, self.topology.number_of_nodes()))
        min_node = min(self.service.source_id, self.service.destination_id)
        max_node = max(self.service.source_id, self.service.destination_id)
        source_destination_tau[0, min_node] = 1
        source_destination_tau[1, max_node] = 1
        spectrum_obs = np.full((self.k_paths*2, (2 * self.j + 3)), fill_value=-1.)
        for idp, path in enumerate(self.k_shortest_paths[self.service.source, self.service.destination]):
            available_slots = self.get_available_slots_working(path)
            num_slots = self.get_number_slots(path)
            initial_indices, lengths = self.get_available_blocks_working(idp)

            for idb, (initial_index, length) in enumerate(zip(initial_indices, lengths)):
                # initial slot index
                spectrum_obs[idp, idb * 2 + 0] = 2 * (
                        initial_index - .5 * self.num_spectrum_resources) / self.num_spectrum_resources

                # number of contiguous FS available
                spectrum_obs[idp, idb * 2 + 1] = (length - 8) / 8
            spectrum_obs[idp, self.j * 2] = (num_slots - 5.5) / 3.5  # number of FSs necessary

            idx, values, lengths = DeepRMSASBPPKSPEnv.rle(available_slots)

            av_indices = np.argwhere(values == 1)  # getting indices which have value 1
            spectrum_obs[idp, self.j * 2 + 1] = 2 * (np.sum(
                available_slots) - .5 * self.num_spectrum_resources) / self.num_spectrum_resources  # total number available FSs
            spectrum_obs[idp, self.j * 2 + 2] = (np.mean(
                lengths[av_indices]) - 4) / 4  # avg. number of FS blocks available
        bit_rate_obs = np.zeros((1, 1))
        bit_rate_obs[0, 0] = self.service.bit_rate / 100
        
        return np.concatenate((bit_rate_obs, source_destination_tau.reshape((1, np.prod(source_destination_tau.shape))),
                               spectrum_obs.reshape((1, np.prod(spectrum_obs.shape)))), axis=1) \
            .reshape(self.observation_space.shape)

    def reward(self):
        return 1 if self.service.accepted else -1
        # if self.service.shared:
        #   return 2

        # if self.service.dpp:
        #   return 1
        
        # if not self.service.accepted:
        #   return -1

    def reset(self, only_counters=True):
        return super().reset(only_counters=only_counters)

    def _get_path_block_id(self, action: int) -> (int, int, int, int): 
        
        #Changed to take into account the 4 variables 
        working_path = action // (self.j * self.j * self.k_paths) % self.k_paths

        block_working = action // (self.j*self.k_paths) %self.j 

        backup_path = action % (self.j*self.k_paths) // self.j

        block_backup = action % self.j

        return working_path, block_working, backup_path, block_backup


def shortest_path_first_fit(env: DeepRMSASBPPKSPEnv) -> int:
    if not env.allow_rejection:
        return 0
    else:
        initial_indices, lengths = env.get_available_blocks(0)
        if len(initial_indices) > 0:  # if there are available slots
            return 0
        else:
            return env.k_paths * env.j


#try if both shared and dedicated fit constraints or if dedicated or dedicated AND shared
def shortest_available_path_first_fit(self: DeepRMSASBPPKSPEnv) -> int:
    for idp_working, path in enumerate(self.k_shortest_paths[self.service.source, self.service.destination]):
        working_initial_indices, lengths = self.get_available_blocks_working(idp_working)
        if len(working_initial_indices) > 0:  # if there are available slots
            for idp_backup, path in enumerate(self.k_shortest_paths[self.service.source, self.service.destination]):
                backup_initial_indices, lengths = self.get_available_blocks_backup(idp_backup)
                backup_dpp_initial_indices, lengths = self.get_available_blocks_working(idp_backup)
                if len(backup_initial_indices) > 0 and self.is_disjoint(self.k_shortest_paths[self.service.source, self.service.destination][idp_working],
                                                                    self.k_shortest_paths[self.service.source, self.service.destination][idp_backup]) and self.is_working_disjoint(self.k_shortest_paths[self.service.source, self.service.destination][idp_working],
                                                                    self.k_shortest_paths[self.service.source, self.service.destination][idp_backup],working_initial_indices[0], backup_initial_indices[0]):
                    return idp_working * self.j, idp_backup * self.j  # this path uses the first one

                if len(backup_dpp_initial_indices) > 0 and self.is_disjoint(self.k_shortest_paths[self.service.source, self.service.destination][idp_working],
                                                                            self.k_shortest_paths[self.service.source, self.service.destination][idp_backup]):                                              
                    return idp_working * self.j, idp_backup * self.j  # this path uses the first one
    return self.k_paths * self.j, self.k_paths * self.j