from typing import List, Tuple
import copy
import math
import heapq
import logging
import functools
import gym
import numpy as np

from optical_rl_gym.utils import Service, Path
from .optical_network_env import OpticalNetworkEnv


class RMSADPPEnv(OpticalNetworkEnv):

    metadata = {
        "metrics": [
            "service_blocking_rate",
            "episode_service_blocking_rate",
            "bit_rate_blocking_rate",
            "episode_bit_rate_blocking_rate",
            "failure",
            "episode_failure",
            "failure_slots",
            "episode_failure_slots",
            "failure_disjointness",
            "episode_failure_disjointness",
        ]
    }

    def __init__(
        self,
        topology=None,
        episode_length=1000,
        load=10,
        mean_service_holding_time=10800.0,
        num_spectrum_resources=100,
        node_request_probabilities=None,
        bit_rate_lower_bound=50,
        bit_rate_higher_bound=200,
        seed=None,
        allow_rejection=False,
        reset=True,
    ):
        super().__init__(
            topology,
            episode_length=episode_length,
            load=load,
            mean_service_holding_time=mean_service_holding_time,
            num_spectrum_resources=num_spectrum_resources,
            node_request_probabilities=node_request_probabilities,
            seed=seed,
            allow_rejection=allow_rejection,
        )
        assert "modulations" in self.topology.graph
        # specific attributes for elastic optical networks

        self.bit_rate_requested = 0
        self.bit_rate_provisioned = 0
        self.episode_bit_rate_requested = 0
        self.episode_bit_rate_provisioned = 0
        self.bit_rate_lower_bound = bit_rate_lower_bound
        self.bit_rate_higher_bound = bit_rate_higher_bound
        self.failure_counter = 0
        self.failure_disjointness = 0
        self.episode_failure_disjointness = 0
        self.episode_failure_counter = 0
        self.current_service = None
        self.episode_failure_counter = 0 

        self.spectrum_slots_allocation = np.full(
            (self.topology.number_of_edges(), self.num_spectrum_resources),
            fill_value=-1,
            dtype=int
        )

        # do we allow proactive rejection or not?
        self.reject_action = 1 if allow_rejection else 0

        # defining the observation and action spaces
        self.actions_output = np.zeros(
            (
                self.k_paths + 1,
                self.num_spectrum_resources + 1,
                self.k_paths + 1,
                self.num_spectrum_resources + 1,
            ),
            dtype=int,
        )
        self.episode_actions_output = np.zeros(
            (
                self.k_paths + 1,
                self.num_spectrum_resources + 1,
                self.k_paths + 1,
                self.num_spectrum_resources + 1,
            ),
            dtype=int,
        )

        self.actions_taken = np.zeros(
            (
                self.k_paths + 1,
                self.num_spectrum_resources + 1,
                self.k_paths + 1,
                self.num_spectrum_resources + 1,
            ),
            dtype=int,
        )
        self.episode_actions_taken = np.zeros(
            (
                self.k_paths + 1,
                self.num_spectrum_resources + 1,
                self.k_paths + 1,
                self.num_spectrum_resources + 1,
            ),
            dtype=int,
        )

        self.action_space = gym.spaces.MultiDiscrete(
            (
                self.k_paths + self.reject_action,
                self.num_spectrum_resources + self.reject_action,
                self.k_paths + self.reject_action,
                self.num_spectrum_resources + self.reject_action,
            )
        )
        self.observation_space = gym.spaces.Dict(
            {
                "topology": gym.spaces.Discrete(10),
                "current_service": gym.spaces.Discrete(10),
            }
        )
        self.action_space.seed(self.rand_seed)
        self.observation_space.seed(self.rand_seed)

        self.logger = logging.getLogger("rmsaenv")
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.warning(
                "Logging is enabled for DEBUG which generates a large number of messages. "
                "Set it to INFO if DEBUG is not necessary."
            )
        self._new_service = False
        if reset:
            self.reset(only_counters=False)

    def step(
        self, action: List[int]
    ) -> Tuple[object, float, bool, dict]:
        working_path, initial_slot_working, backup_path, initial_slot_backup = action
        self.actions_output[working_path, initial_slot_working, backup_path, initial_slot_backup] += 1
        
        if all((
            working_path < self.k_paths,
            initial_slot_working < self.num_spectrum_resources,
            backup_path < self.k_paths,
            initial_slot_backup < self.num_spectrum_resources
        )):
            working_path_obj = self.k_shortest_paths[self.current_service.source, self.current_service.destination][working_path]
            backup_path_obj = self.k_shortest_paths[self.current_service.source, self.current_service.destination][backup_path]
            
            if not self.is_disjoint(working_path_obj, backup_path_obj):
                self.current_service.accepted = False
                self.failure_disjointness += 1
                self.episode_failure_disjointness += 1
            else:
                working_slots = self.get_number_slots(working_path_obj)
                backup_slots = self.get_number_slots(backup_path_obj)
                
                if self.is_path_free(working_path_obj, initial_slot_working, working_slots):
                    if self.is_path_free(backup_path_obj, initial_slot_backup, backup_slots):
                        self._provision_path(
                            working_path_obj,
                            initial_slot_working,
                            working_slots,
                            backup_path_obj,
                            initial_slot_backup,
                            backup_slots
                        )
                        self.current_service.accepted = True
                        self.actions_taken[working_path, initial_slot_working, backup_path, initial_slot_backup] += 1
                        self._add_release(self.current_service)
                    else:
                        self.current_service.accepted = False
                else:
                    self.current_service.accepted = False
        else:
            self.current_service.accepted = False

        if not self.current_service.accepted:
            self.actions_taken[self.k_paths, self.num_spectrum_resources, self.k_paths, self.num_spectrum_resources] += 1
            self.failure_counter += 1
            self.episode_failure_counter += 1

        # Update metrics
        self.services_processed += 1
        self.episode_services_processed += 1
        self.bit_rate_requested += self.current_service.bit_rate
        self.episode_bit_rate_requested += self.current_service.bit_rate

        if self.current_service.accepted:
            self.bit_rate_provisioned += self.current_service.bit_rate
            self.episode_bit_rate_provisioned += self.current_service.bit_rate

        self.topology.graph["services"].append(self.current_service)

        # Calculate reward and info
        reward = self.reward()
        info = self._get_info()
        
        self._new_service = False
        self._next_service()
        
        return (
            self.observation(),
            reward,
            self.episode_services_processed >= self.episode_length,
            info
        )

    def _get_info(self) -> dict:
        return {
            "service_blocking_rate": (self.services_processed - self.services_accepted) / self.services_processed,
            "episode_service_blocking_rate": (self.episode_services_processed - self.episode_services_accepted) / self.episode_services_processed,
            "bit_rate_blocking_rate": (self.bit_rate_requested - self.bit_rate_provisioned) / self.bit_rate_requested,
            "episode_bit_rate_blocking_rate": (self.episode_bit_rate_requested - self.episode_bit_rate_provisioned) / self.episode_bit_rate_requested,
            "failure": self.failure_counter / self.services_processed,
            "episode_failure": self.episode_failure_counter / self.episode_services_processed,
            "failure_slots": (self.failure_counter - self.failure_disjointness) / self.services_processed,
            "episode_failure_slots": (self.episode_failure_counter - self.episode_failure_disjointness) / self.episode_services_processed,
            "failure_disjointness": self.failure_disjointness / self.services_processed,
            "episode_failure_disjointness": self.episode_failure_disjointness / self.episode_services_processed,
        }

    def reset(self, only_counters: bool = True) -> object:
        self.episode_bit_rate_requested = 0
        self.episode_bit_rate_provisioned = 0
        self.episode_services_processed = 0
        self.episode_services_accepted = 0
        self.episode_failure_disjointness = 0
        self.episode_failure_counter = 0

        self.episode_actions_output = np.zeros_like(self.actions_output)
        self.episode_actions_taken = np.zeros_like(self.actions_taken)

        if not only_counters:
            super().reset()
            self.bit_rate_requested = 0
            self.bit_rate_provisioned = 0
            self.topology.graph["available_slots"] = np.ones(
                (self.topology.number_of_edges(), self.num_spectrum_resources), dtype=int
            )
            self.spectrum_slots_allocation.fill(-1)
            self.topology.graph["compactness"] = 0.0
            self.topology.graph["throughput"] = 0.0
            
            for _, _, link in self.topology.edges(data=True):
                link["external_fragmentation"] = 0.0
                link["compactness"] = 0.0

        self._new_service = False
        self._next_service()
        return self.observation()

    def render(self, mode: str = "human") -> None:
        pass

    def _provision_path(
        self,
        working_path: Path,
        initial_slot_working: int,
        number_slots_working: int,
        backup_path: Path,
        initial_slot_backup: int,
        number_slots_backup: int,
    ) -> None:
        # Validate both paths
        if not self.is_path_free(working_path, initial_slot_working, number_slots_working):
            raise ValueError(f"Working path {working_path.node_list} has insufficient capacity")
        if not self.is_path_free(backup_path, initial_slot_backup, number_slots_backup):
            raise ValueError(f"Backup path {backup_path.node_list} has insufficient capacity")

        # Provision working path
        for i in range(len(working_path.node_list) - 1):
            edge = self.topology[working_path.node_list[i]][working_path.node_list[i+1]]
            idx = edge["index"]
            self.topology.graph["available_slots"][idx, initial_slot_working:initial_slot_working+number_slots_working] = 0
            self.spectrum_slots_allocation[idx, initial_slot_working:initial_slot_working+number_slots_working] = self.current_service.service_id
            edge["services"].append(self.current_service)
            edge["running_services"].append(self.current_service)
            self._update_link_stats(working_path.node_list[i], working_path.node_list[i+1])

        # Provision backup path
        for i in range(len(backup_path.node_list) - 1):
            edge = self.topology[backup_path.node_list[i]][backup_path.node_list[i+1]]
            idx = edge["index"]
            self.topology.graph["available_slots"][idx, initial_slot_backup:initial_slot_backup+number_slots_backup] = 0
            self.spectrum_slots_allocation[idx, initial_slot_backup:initial_slot_backup+number_slots_backup] = self.current_service.service_id
            edge["services"].append(self.current_service)
            edge["running_services"].append(self.current_service)
            self._update_link_stats(backup_path.node_list[i], backup_path.node_list[i+1])

        # Update service properties
        self.current_service.route = working_path
        self.current_service.initial_slot = initial_slot_working
        self.current_service.number_slots = number_slots_working
        self.current_service.backup_route = backup_path
        self.current_service.initial_slot_backup = initial_slot_backup
        self.current_service.number_slots_backup = number_slots_backup
        
        self.topology.graph["running_services"].append(self.current_service)
        self.services_accepted += 1
        self.episode_services_accepted += 1
        self._update_network_stats()

    def _release_path(self, service: Service) -> None:
        # Release working path
        for i in range(len(service.route.node_list) - 1):
            edge = self.topology[service.route.node_list[i]][service.route.node_list[i+1]]
            idx = edge["index"]
            self.topology.graph["available_slots"][idx, service.initial_slot:service.initial_slot+service.number_slots] = 1
            self.spectrum_slots_allocation[idx, service.initial_slot:service.initial_slot+service.number_slots] = -1
            if service in edge["running_services"]:
                edge["running_services"].remove(service)
            self._update_link_stats(service.route.node_list[i], service.route.node_list[i+1])

        # Release backup path
        for i in range(len(service.backup_route.node_list) - 1):
            edge = self.topology[service.backup_route.node_list[i]][service.backup_route.node_list[i+1]]
            idx = edge["index"]
            self.topology.graph["available_slots"][idx, service.initial_slot_backup:service.initial_slot_backup+service.number_slots_backup] = 1
            self.spectrum_slots_allocation[idx, service.initial_slot_backup:service.initial_slot_backup+service.number_slots_backup] = -1
            if service in edge["running_services"]:
                edge["running_services"].remove(service)
            self._update_link_stats(service.backup_route.node_list[i], service.backup_route.node_list[i+1])

        if service in self.topology.graph["running_services"]:
            self.topology.graph["running_services"].remove(service)

    def _update_network_stats(self) -> None:
        last_update = self.topology.graph["last_update"]
        time_diff = self.current_time - last_update
        
        if time_diff > 0:
            # Throughput calculation
            current_throughput = sum(service.bit_rate for service in self.topology.graph["running_services"])
            self.topology.graph["throughput"] = ((self.topology.graph["throughput"] * last_update) + 
                                                (current_throughput * time_diff)) / self.current_time
            
            # Compactness calculation
            current_compactness = self._get_network_compactness()
            self.topology.graph["compactness"] = ((self.topology.graph["compactness"] * last_update) + 
                                                 (current_compactness * time_diff)) / self.current_time
            
        self.topology.graph["last_update"] = self.current_time

    def _update_link_stats(self, node1: str, node2: str) -> None:
        edge = self.topology[node1][node2]
        time_diff = self.current_time - edge["last_update"]
        
        if time_diff > 0:
            # Utilization calculation
            used_slots = self.num_spectrum_resources - np.sum(
                self.topology.graph["available_slots"][edge["index"], :]
            )
            new_utilization = used_slots / self.num_spectrum_resources
            edge["utilization"] = ((edge["utilization"] * edge["last_update"]) + 
                                  (new_utilization * time_diff)) / self.current_time
            
            # Fragmentation and compactness calculations
            slot_allocation = self.topology.graph["available_slots"][edge["index"], :]
            initial_indices, values, lengths = self.rle(slot_allocation)
            
            # External fragmentation
            available_blocks = [i for i, val in enumerate(values) if val == 1]
            max_empty = max(lengths[available_blocks]) if available_blocks else 0
            external_frag = 1.0 - (max_empty / used_slots) if used_slots > 0 else 0.0
            edge["external_fragmentation"] = ((edge["external_fragmentation"] * edge["last_update"]) + 
                                             (external_frag * time_diff)) / self.current_time
            
            # Link compactness
            used_blocks = [i for i, val in enumerate(values) if val == 0]
            if len(used_blocks) > 1:
                lambda_min = initial_indices[used_blocks[0]]
                lambda_max = initial_indices[used_blocks[-1]] + lengths[used_blocks[-1]]
                unused_in_span = np.sum(slot_allocation[lambda_min:lambda_max])
                compactness = ((lambda_max - lambda_min) / used_slots) * (1 / unused_in_span) if unused_in_span > 0 else 1.0
            else:
                compactness = 1.0
            edge["compactness"] = ((edge["compactness"] * edge["last_update"]) + 
                                  (compactness * time_diff)) / self.current_time
            
        edge["last_update"] = self.current_time

    def _next_service(self) -> None:
        if self._new_service:
            return
            
        self.current_time += self.rng.expovariate(1.0/self.mean_service_inter_arrival_time)
        self.current_service = Service(
            id=self.episode_services_processed,
            source=self.rng.choice(self.topology.nodes()),
            destination=self.rng.choice(self.topology.nodes()),
            arrival_time=self.current_time,
            holding_time=self.rng.expovariate(1.0/self.mean_service_holding_time),
            bit_rate=self.rng.randint(self.bit_rate_lower_bound, self.bit_rate_higher_bound)
        )
        self._new_service = True

    def reward(self) -> float:
        return self.current_service.accepted * self.current_service.bit_rate

    def observation(self) -> dict:
        return {
            "topology": self.topology,
            "current_service": self.current_service,
            "available_slots": copy.deepcopy(self.topology.graph["available_slots"])
        }

    @staticmethod
    def rle(inarray: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        ia = np.asarray(inarray)
        n = len(ia)
        if n == 0: 
            return (np.array([]), np.array([]), np.array([]))
        
        y = ia[1:] != ia[:-1]
        i = np.append(np.where(y), n-1)
        z = np.diff(np.append(-1, i))
        return (np.cumsum(z[:-1]), ia[i], z)

    def is_disjoint(self, path1: Path, path2: Path) -> bool:
        nodes1 = set(path1.node_list[1:-1])
        nodes2 = set(path2.node_list[1:-1])
        return nodes1.isdisjoint(nodes2)

    def get_number_slots(self, path: Path) -> int:
        return math.ceil(path.best_modulation["capacity"] / self.current_service.bit_rate) + 1

    def is_path_free(self, path: Path, initial_slot: int, number_slots: int) -> bool:
        if initial_slot + number_slots > self.num_spectrum_resources:
            return False
        for i in range(len(path.node_list)-1):
            edge = self.topology[path.node_list[i]][path.node_list[i+1]]
            if np.any(self.topology.graph["available_slots"][edge["index"], initial_slot:initial_slot+number_slots] == 0):
                return False
        return True


# Heuristic algorithms for comparison
def shortest_path_first_fit(env: RMSADPPEnv) -> List[int]:
    path = env.k_shortest_paths[env.current_service.source, env.current_service.destination][0]
    slots_needed = env.get_number_slots(path)
    for slot in range(env.num_spectrum_resources - slots_needed + 1):
        if env.is_path_free(path, slot, slots_needed):
            return [0, slot, 0, slot]  # Same path/slot for backup (not optimal)
    return [env.k_paths, env.num_spectrum_resources] * 2

def least_loaded_path_first_fit(env: RMSADPPEnv) -> List[int]:
    best_load = -1
    best_action = [env.k_paths, env.num_spectrum_resources] * 2
    for path_id in range(env.k_paths):
        path = env.k_shortest_paths[env.current_service.source, env.current_service.destination][path_id]
        slots_needed = env.get_number_slots(path)
        for slot in range(env.num_spectrum_resources - slots_needed + 1):
            if env.is_path_free(path, slot, slots_needed):
                load = sum(1 - env.topology.graph["available_slots"][
                    env.topology[path.node_list[i]][path.node_list[i+1]]["index"], 
                    slot:slot+slots_needed
                ].mean() for i in range(len(path.node_list)-1))
                if load > best_load:
                    best_load = load
                    best_action = [path_id, slot, path_id, slot]
    return best_action


def shortest_available_path_first_fit(env: RMSADPPEnv) -> int:
    for idp, path in enumerate(
        env.k_shortest_paths[env.service.source, env.service.destination]
    ):
        num_slots = env.get_number_slots(path)
        for initial_slot in range(
            0, env.topology.graph["num_spectrum_resources"] - num_slots
        ):
            if env.is_path_free(path, initial_slot, num_slots):
                return [idp, initial_slot]
    return [env.topology.graph["k_paths"], env.topology.graph["num_spectrum_resources"]]


def least_loaded_path_first_fit(env: RMSADPPEnv) -> int:
    max_free_slots = 0
    action = [
        env.topology.graph["k_paths"],
        env.topology.graph["num_spectrum_resources"],
    ]
    for idp, path in enumerate(
        env.k_shortest_paths[env.service.source, env.service.destination]
    ):
        num_slots = env.get_number_slots(path)
        for initial_slot in range(
            0, env.topology.graph["num_spectrum_resources"] - num_slots
        ):
            if env.is_path_free(path, initial_slot, num_slots):
                free_slots = np.sum(env.get_available_slots(path))
                if free_slots > max_free_slots:
                    action = [idp, initial_slot]
                    max_free_slots = free_slots
                break  # breaks the loop for the initial slot
    return action


class SimpleMatrixObservation(gym.ObservationWrapper):

    def __init__(self, env: RMSADPPEnv):
        super().__init__(env)
        shape = (
            self.env.topology.number_of_nodes() * 2
            + self.env.topology.number_of_edges() * self.env.num_spectrum_resources
        )
        self.observation_space = gym.spaces.Box(
            low=0, high=1, dtype=np.uint8, shape=(shape,)
        )
        self.action_space = env.action_space

    def observation(self, observation):
        source_destination_tau = np.zeros((2, self.env.topology.number_of_nodes()))
        min_node = min(self.env.service.source_id, self.env.service.destination_id)
        max_node = max(self.env.service.source_id, self.env.service.destination_id)
        source_destination_tau[0, min_node] = 1
        source_destination_tau[1, max_node] = 1
        spectrum_obs = copy.deepcopy(self.topology.graph["available_slots"])
        return np.concatenate(
            (
                source_destination_tau.reshape(
                    (1, np.prod(source_destination_tau.shape))
                ),
                spectrum_obs.reshape((1, np.prod(spectrum_obs.shape))),
            ),
            axis=1,
        ).reshape(self.observation_space.shape)


class PathOnlyFirstFitAction(gym.ActionWrapper):
    def __init__(self, env: RMSADPPEnv):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(
            self.env.k_paths + self.env.reject_action
        )
        self.observation_space = env.observation_space

    def action(self, action):
        if action < self.env.k_paths:
            num_slots = self.env.get_number_slots(
                self.env.k_shortest_paths[
                    self.env.service.source, self.env.service.destination
                ][action]
            )
            for initial_slot in range(
                0, self.env.topology.graph["num_spectrum_resources"] - num_slots
            ):
                if self.env.is_path_free(
                    self.env.k_shortest_paths[
                        self.env.service.source, self.env.service.destination
                    ][action],
                    initial_slot,
                    num_slots,
                ):
                    return [action, initial_slot]
        return [
            self.env.topology.graph["k_paths"],
            self.env.topology.graph["num_spectrum_resources"],
        ]

    def step(self, action):
        return self.env.step(self.action(action))
