import copy
import functools
import logging
import networkx as nx
import numpy as np
import gym

from optical_rl_gym.utils import Service, Path
from .rmcsa_env import RMCSAEnv
from optical_rl_gym.utils import get_best_modulation_format


class RMSCASBPPEnv(RMCSAEnv):
    def __init__(
        self,
        topology: nx.Graph = None,
        episode_length: int = 1000,
        load: float = 10,
        mean_service_holding_time: float = 10800.0,
        num_spectrum_resources: int = 100,
        num_spatial_resources: int = 7,
        channel_width: float = 12.5,
        seed=None,
        allow_rejection: bool = False,
        reset: bool = True,
    ):
        self.num_links = topology.number_of_edges()

        topology.graph["available_slots_working"] = np.ones(
            (
                num_spatial_resources,
                self.num_links,
                num_spectrum_resources,
            ),
            dtype=int,
        )

        topology.graph["available_slots_backup"] = np.ones(
            (
                num_spatial_resources,
                self.num_links,
                num_spectrum_resources,
            ),
            dtype=int,
        )

        super().__init__(
            topology=topology,
            episode_length=episode_length,
            load=load,
            mean_service_holding_time=mean_service_holding_time,
            num_spectrum_resources=num_spectrum_resources,
            num_spatial_resources=num_spatial_resources,
            channel_width=channel_width,
            seed=seed,
            allow_rejection=allow_rejection,
            reset=reset,
        )

        # Failure related variables
        self.failure_counter = 0
        self.failure_disjointness = 0
        self.episode_failure_counter = 0
        self.episode_failure_disjointness = 0
        self.failure_shared_disjointness = 0
        self.episode_failure_shared_disjointness = 0
        self.failure_slots = 0
        self.episode_failure_slots = 0
        self.failure_crosstalk = 0
        self.episode_failure_crosstalk = 0

        # Resilience related variables
        self.dpp_counter = 0
        self.shared_counter = 0
        self.episode_dpp_counter = 0
        self.episode_shared_counter = 0

        # Performance related variables
        self.compactness = 0
        self.throughput = 0
        self.available_slots_working = 0
        self.available_slots_backup = 0

        self.actions_output = np.zeros(
            (
                self.k_paths + 1,
                len(self.modulation_formats) + 1,
                self.num_spatial_resources + 1,
                self.num_spectrum_resources + 1,
                self.k_paths + 1,
                len(self.modulation_formats) + 1,
                self.num_spatial_resources + 1,
                self.num_spectrum_resources + 1,
            ),
            dtype=int,
        )

        # Modify action space for working and backup paths
        self.action_space = gym.spaces.MultiDiscrete(
            (
                self.k_paths + self.reject_action,  # working path
                len(self.modulation_formats),  # working path modulation
                self.num_spatial_resources + self.reject_action,  # working path core
                self.num_spectrum_resources
                + self.reject_action,  # working path initial slot
                self.k_paths + self.reject_action,  # backup path
                len(self.modulation_formats),  # backup path modulation
                self.num_spatial_resources + self.reject_action,  # backup path core
                self.num_spectrum_resources
                + self.reject_action,  # backup path initial slot
            )
        )

        self.logger = logging.getLogger("rmsaenv_sbpp")
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.warning(
                "Logging is enabled for DEBUG which generates a large number of messages. "
                "Set it to INFO if DEBUG is not necessary."
            )

        self._new_service = False
        if reset:
            self.reset(only_counters=False)

    def step(self, action):
        """
        Executa uma ação no ambiente.

        Args:
            action: Uma tupla/array com os valores:
                [0] working_path
                [1] working_modulation
                [2] working_core
                [3] working_initial_slot
                [4] backup_path
                [5] backup_modulation
                [6] backup_core
                [7] backup_initial_slot
        """
        # Descompactando os 8 valores da ação
        (
            working_path,
            working_modulation,
            working_core,
            working_initial_slot,
            backup_path,
            backup_modulation,
            backup_core,
            backup_initial_slot,
        ) = action

        self.actions_output[
            working_path,
            working_modulation,
            working_core,
            working_initial_slot,
            backup_path,
            backup_modulation,
            backup_core,
            backup_initial_slot,
        ] += 1

        if (
            (working_path < self.k_paths)
            and (working_modulation < len(self.modulation_formats))
            and (working_core < self.num_spatial_resources)
            and (working_initial_slot < self.num_spectrum_resources)
            and (backup_path < self.k_paths)
            and (backup_modulation < len(self.modulation_formats))
            and (backup_core < self.num_spatial_resources)
            and (backup_initial_slot < self.num_spectrum_resources)
        ):
            # Obtem os caminhos
            working_path_obj = self.k_shortest_paths[
                self.current_service.source, self.current_service.destination
            ][working_path]

            backup_path_obj = self.k_shortest_paths[
                self.current_service.source, self.current_service.destination
            ][backup_path]

            # Obtém os slots necessários
            working_slots = self.get_number_slots(
                working_path_obj, self.modulation_formats[working_modulation]
            )

            backup_slots = self.get_number_slots(
                backup_path_obj, self.modulation_formats[backup_modulation]
            )

            # Verifica se os caminhos são disjuntos
            if not self.is_disjoint(working_path_obj, backup_path_obj):
                self.current_service.accepted = False
                self.failure_disjointness += 1
                self.episode_failure_disjointness += 1

            else:
                # Verifica disponibilidade do caminho de trabalho
                if not self.is_path_free(
                    working_path_obj, working_core, working_initial_slot, working_slots
                ):
                    self.current_service.accepted = False
                    self.failure_slots += 1
                    self.episode_failure_slots += 1

                # Verifica disponibilidade do caminho de backup
                elif not self.is_backup_path_free(
                    backup_path_obj, backup_core, backup_initial_slot, backup_slots
                ):
                    self.current_service.accepted = False
                    self.failure_slots += 1
                    self.episode_failure_slots += 1

                else:
                    # Verifica crosstalk
                    if self._crosstalk_is_acceptable(
                        self.modulation_formats[working_modulation],
                        working_path_obj.length,
                    ) and self._crosstalk_is_acceptable(
                        self.modulation_formats[backup_modulation],
                        backup_path_obj.length,
                    ):
                        # Provisiona os caminhos
                        self._provision_path(
                            working_path_obj,
                            working_core,
                            working_initial_slot,
                            working_slots,
                            backup_path_obj,
                            backup_core,
                            backup_initial_slot,
                            backup_slots,
                        )

                        self.current_service.accepted = True
                        self.shared_counter += 1
                        self.episode_shared_counter += 1
                        self._add_release(self.current_service)

                    else:
                        self.current_service.accepted = False
                        self.failure_crosstalk += 1
                        self.episode_failure_crosstalk += 1

        else:
            self.current_service.accepted = False
            self.failure_slots += 1
            self.episode_failure_slots += 1

        if not self.current_service.accepted:
            self.actions_taken[
                self.k_paths,
                len(self.modulation_formats),
                self.num_spatial_resources,
                self.num_spectrum_resources,
            ] += 1
            self.failure_counter += 1
            self.episode_failure_counter += 1

        self.services_processed += 1
        self.episode_services_processed += 1
        self.bit_rate_requested += self.current_service.bit_rate
        self.episode_bit_rate_requested += self.current_service.bit_rate

        self.compactness = self.topology.graph["compactness"]
        self.throughput = self.topology.graph["throughput"]
        self.available_slots_working = np.sum(
            self.topology.graph["available_slots_working"]
        )
        self.available_slots_backup = np.sum(
            self.topology.graph["available_slots_backup"]
        )

        self.topology.graph["services"].append(self.current_service)

        reward = self.reward()

        info = {
            "service_blocking_rate": (self.services_processed - self.services_accepted)
            / self.services_processed,
            "episode_service_blocking_rate": (
                self.episode_services_processed - self.episode_services_accepted
            )
            / self.episode_services_processed,
            "bit_rate_blocking_rate": (
                self.bit_rate_requested - self.bit_rate_provisioned
            )
            / self.bit_rate_requested,
            "episode_bit_rate_blocking_rate": (
                self.episode_bit_rate_requested - self.episode_bit_rate_provisioned
            )
            / self.episode_bit_rate_requested,
            "failure": self.failure_counter / self.services_processed,
            "episode_failure": self.episode_failure_counter
            / self.episode_services_processed,
            "failure_slots": self.failure_slots / self.services_processed,
            "episode_failure_slots": self.episode_failure_slots,
            "failure_disjointness": self.failure_disjointness / self.services_processed,
            "episode_failure_disjointness": self.episode_failure_disjointness
            / self.episode_services_processed,
            "failure_shared_disjointness": self.failure_shared_disjointness
            / self.services_processed,
            "episode_failure_shared_disjointness": self.episode_failure_shared_disjointness
            / self.episode_services_processed,
            "failure_crosstalk": self.failure_crosstalk / self.services_processed,
            "episode_failure_crosstalk": self.episode_failure_crosstalk
            / self.services_processed,
            "shared_counter": self.shared_counter / self.services_processed,
            "episode_shared_counter": self.episode_shared_counter
            / self.episode_services_processed,
            "dpp_counter": self.dpp_counter / self.services_processed,
            "episode_dpp_counter": self.episode_dpp_counter
            / self.episode_services_processed,
            "compactness": self.compactness,
            "throughput": self.throughput,
            "available_slots_working": self.available_slots_working,
            "available_slots_backup": self.available_slots_backup,
        }

        self._new_service = False
        self._next_service()
        return (
            self.observation(),
            reward,
            self.episode_services_processed == self.episode_length,
            info,
        )

    def reset(self, only_counters=True):
        self.episode_bit_rate_requested = 0
        self.episode_bit_rate_provisioned = 0

        self.episode_services_processed = 0
        self.episode_services_accepted = 0

        self.episode_failure_counter = 0
        self.episode_failure_disjointness = 0
        self.episode_failure_shared_disjointness = 0
        self.episode_failure_slots = 0
        self.episode_failure_crosstalk = 0
        self.episode_shared_counter = 0
        self.episode_dpp_counter = 0

        self.episode_actions_output = np.zeros(
            (
                self.k_paths + 1,
                len(self.modulation_formats) + 1,
                self.num_spatial_resources + 1,
                self.num_spectrum_resources + 1,
            ),
            dtype=int,
        )
        self.episode_actions_taken = np.zeros(
            (
                self.k_paths + 1,
                len(self.modulation_formats) + 1,
                self.num_spatial_resources + 1,
                self.num_spectrum_resources + 1,
            ),
            dtype=int,
        )

        if only_counters:
            return self.observation()

        super().reset()

        self.bit_rate_requested = 0
        self.bit_rate_provisioned = 0

        self.topology.graph["services"] = []

        self.topology.graph["available_slots_working"] = np.ones(
            (
                self.num_spatial_resources,
                self.num_links,
                self.num_spectrum_resources,
            ),
            dtype=int,
        )
        self.topology.graph["available_slots_backup"] = np.ones(
            (
                self.num_spatial_resources,
                self.num_links,
                self.num_spectrum_resources,
            ),
            dtype=int,
        )

        self.spectrum_slots_allocation = np.empty(
            (
                self.num_spatial_resources,
                self.num_links,
                self.num_spectrum_resources,
            ),
            dtype=object,
        )

        for core, link, slot in np.ndindex(self.spectrum_slots_allocation.shape):
            self.spectrum_slots_allocation[core, link, slot] = []

        self.topology.graph["compactness"] = 0.0
        self.topology.graph["throughput"] = 0.0

        for link in self.topology.edges():
            self.topology[link[0]][link[1]]["external_fragmentation"] = 0.0
            self.topology[link[0]][link[1]]["compactness"] = 0.0

        self._new_service = False
        self._next_service()
        return self.observation()

    def _provision_path(
        self,
        working_path: Path,
        core_working: int,
        initial_slot_working: int,
        number_slots_working: int,
        backup_path: Path,
        core_backup: int,
        initial_slot_backup: int,
        number_slots_backup: int,
    ) -> None:
        if not self.is_path_free(
            working_path, core_working, initial_slot_working, number_slots_working
        ):
            raise ValueError(
                f"Working path {working_path.node_list} has not enough slots available on core {core_working} from {initial_slot_working} to {initial_slot_working + number_slots_working}"
            )
        if not self.is_backup_path_free(
            backup_path, core_backup, initial_slot_backup, number_slots_backup
        ):
            raise ValueError(
                f"Backup path {backup_path.node_list} has not enough slots available on core {core_backup} from {initial_slot_backup} to {initial_slot_backup + number_slots_backup}"
            )

        # Provision the working path
        self.logger.debug(
            "%s - Provisioning working path %s, core %s, slots %s-%s",
            self.current_service.service_id,
            working_path,
            core_working,
            initial_slot_working,
            initial_slot_working + number_slots_working,
        )

        for i in range(1, len(working_path.node_list) - 1):
            link_index = self.topology[working_path.node_list[i]][
                working_path.node_list[i + 1]
            ]["index"]

            self.topology.graph["available_slots_working"][
                core_working,
                link_index,
                initial_slot_working : initial_slot_working + number_slots_working,
            ] = 0

            self.topology.graph["available_slots_backup"][
                core_backup,
                link_index,
                initial_slot_backup : initial_slot_backup + number_slots_backup,
            ] = 0

            allocated_spectrum_slots = self.spectrum_slots_allocation[
                core_working,
                link_index,
                initial_slot_working : initial_slot_working + number_slots_working,
            ]

            for slot in np.ndindex(allocated_spectrum_slots.shape):
                allocated_spectrum_slots[slot].append(self.current_service.service_id)

            self.topology[working_path.node_list[i]][working_path.node_list[i + 1]][
                "services"
            ].append(self.current_service)
            self.topology[working_path.node_list[i]][working_path.node_list[i + 1]][
                "running_services"
            ].append(self.current_service)
            self._update_link_stats(
                core_working,
                working_path.node_list[i],
                working_path.node_list[i + 1],
            )

        # Provision the backup path
        self.logger.debug(
            "%s - Provisioning backup path %s, core %s, slots %s-%s",
            self.current_service.service_id,
            backup_path,
            core_backup,
            initial_slot_backup,
            initial_slot_backup + number_slots_backup,
        )

        for i in range(1, len(backup_path.node_list) - 1):
            link_index = self.topology[backup_path.node_list[i]][
                backup_path.node_list[i + 1]
            ]["index"]

            self.topology.graph["available_slots_working"][
                core_working,
                link_index,
                initial_slot_working : initial_slot_working + number_slots_working,
            ] = 0

            allocated_spectrum_slots = self.spectrum_slots_allocation[
                core_backup,
                link_index,
                initial_slot_backup : initial_slot_backup + number_slots_backup,
            ]

            for slot in np.ndindex(allocated_spectrum_slots.shape):
                allocated_spectrum_slots[slot].append(self.current_service.service_id)

            self.topology[backup_path.node_list[i]][backup_path.node_list[i + 1]][
                "services"
            ].append(self.current_service)
            self.topology[backup_path.node_list[i]][backup_path.node_list[i + 1]][
                "running_services"
            ].append(self.current_service)
            self._update_link_stats(
                core_backup,
                backup_path.node_list[i],
                backup_path.node_list[i + 1],
            )

        self.topology.graph["running_services"].append(self.current_service)
        self.current_service.route = working_path
        self.current_service.core = core_working
        self.current_service.initial_slot = initial_slot_working
        self.current_service.number_slots = number_slots_working

        self.current_service.backup_route = backup_path
        self.current_service.backup_core = core_backup
        self.current_service.backup_initial_slot = initial_slot_backup
        self.current_service.backup_number_slots = number_slots_backup

        self._update_network_stats(core_working)
        self._update_network_stats(core_backup)

        self.services_accepted += 1
        self.episode_services_accepted += 1

        self.bit_rate_provisioned += self.current_service.bit_rate
        self.episode_bit_rate_provisioned += self.current_service.bit_rate

    def _release_path(self, service: Service):
        # Release the working path
        for i in range(len(service.route.node_list) - 1):
            link_index = self.topology[service.route.node_list[i]][
                service.route.node_list[i + 1]
            ]["index"]

            self.topology.graph["available_slots_working"][
                service.core,
                link_index,
                service.initial_slot : service.initial_slot + service.number_slots,
            ] = 1

            allocated_spectrum_slots = self.spectrum_slots_allocation[
                service.core,
                link_index,
                service.initial_slot : service.initial_slot + service.number_slots,
            ]

            for slot in np.ndindex(allocated_spectrum_slots.shape):
                allocated_spectrum_slots[slot].remove(service.service_id)

            self.topology[service.route.node_list[i]][service.route.node_list[i + 1]][
                "running_services"
            ].remove(service)
            self._update_link_stats(
                service.core, service.route.node_list[i], service.route.node_list[i + 1]
            )

        # Release the backup path
        for i in range(len(service.backup_route.node_list) - 1):
            link_index = self.topology[service.backup_route.node_list[i]][
                service.backup_route.node_list[i + 1]
            ]["index"]

            self.topology.graph["available_slots_backup"][
                service.backup_core,
                link_index,
                service.backup_initial_slot : service.backup_initial_slot
                + service.backup_number_slots,
            ] = 1

            allocated_spectrum_slots = self.spectrum_slots_allocation[
                service.backup_core,
                link_index,
                service.backup_initial_slot : service.backup_initial_slot
                + service.backup_number_slots,
            ]

            for slot in np.ndindex(allocated_spectrum_slots.shape):
                allocated_spectrum_slots[slot].remove(service.service_id)

            self.topology[service.backup_route.node_list[i]][
                service.backup_route.node_list[i + 1]
            ]["running_services"].remove(service)
            self._update_link_stats(
                service.backup_core,
                service.backup_route.node_list[i],
                service.backup_route.node_list[i + 1],
            )

    def _update_link_stats(self, core, node1, node2):
        """Creates metrics for:
        Individual node "utilization", overall "core_utilization",
        "external fragmentation", and "link_compactness".

        :param core : number of cores,
        :param node1: number of node1 within the node_list
        :param node2: number of node2 within the node_list
        """

        last_update = self.topology[node1][node2]["last_update"]
        time_diff = self.current_time - last_update

        if self.current_time > 0:
            last_util = self.topology[node1][node2]["utilization"]
            cur_util = (
                self.num_spectrum_resources
                - np.sum(
                    self.topology.graph["available_slots_working"][
                        core, self.topology[node1][node2]["index"], :
                    ]
                )
            ) / self.num_spectrum_resources
            utilization = (last_util * last_update) + (
                cur_util * time_diff
            ) / self.current_time
            self.topology[node1][node2]["utilization"] = utilization

            # Adds each node utilization value to an array
            self.utilization.append(utilization)

            # Adds each node utilization value to the core key within a dictionary
            self.core_utilization[core].append(utilization)

            slots_allocation = self.topology.graph["available_slots_working"][
                core, self.topology[node1][node2]["index"], :
            ]

            # implementing fragmentation from https://ieeexplore.ieee.org/abstract/document/6421472
            last_external_fragmentation = self.topology[node1][node2][
                "external_fragmentation"
            ]
            last_compactness = self.topology[node1][node2]["compactness"]

            cur_external_fragmentation = 0.0
            cur_link_compactness = 0.0
            if np.sum(slots_allocation) > 0:
                initial_indices, values, lengths = RMCSAEnv.rle(slots_allocation)

                # computing external fragmentation from
                # https://ieeexplore.ieee.org/abstract/document/6421472
                unused_blocks = [i for i, x in enumerate(values) if x == 1]
                max_empty = 0
                if len(unused_blocks) > 1 and unused_blocks != [0, len(values) - 1]:
                    max_empty = max(lengths[unused_blocks])
                cur_external_fragmentation = 1.0 - (
                    float(max_empty) / float(np.sum(slots_allocation))
                )

                # computing link spectrum compactness from
                # https://ieeexplore.ieee.org/abstract/document/6421472
                used_blocks = [i for i, x in enumerate(values) if x == 0]

                if len(used_blocks) > 1:
                    lambda_min = initial_indices[used_blocks[0]]
                    lambda_max = (
                        initial_indices[used_blocks[-1]] + lengths[used_blocks[-1]]
                    )

                    # evaluate again the "used part" of the spectrum
                    _internal_idx, internal_values, _internal_lengths = RMCSAEnv.rle(
                        slots_allocation[lambda_min:lambda_max]
                    )
                    unused_spectrum_slots = np.sum(1 - internal_values)

                    if unused_spectrum_slots > 0:
                        cur_link_compactness = (
                            (lambda_max - lambda_min) / np.sum(1 - internal_values)
                        ) * (1 / unused_spectrum_slots)
                    else:
                        cur_link_compactness = 1.0
                else:
                    cur_link_compactness = 1.0

            external_fragmentation = (
                (last_external_fragmentation * last_update)
                + (cur_external_fragmentation * time_diff)
            ) / self.current_time
            self.topology[node1][node2][
                "external_fragmentation"
            ] = external_fragmentation

            link_compactness = (
                (last_compactness * last_update) + (cur_link_compactness * time_diff)
            ) / self.current_time
            self.topology[node1][node2]["compactness"] = link_compactness

        self.topology[node1][node2]["last_update"] = self.current_time

    def is_disjoint(self, working_path: Path, backup_path: Path) -> bool:
        """
        Method that checks if the working and backup paths are disjoint.

        :param working_path: the working path
        :param backup_path: the backup path
        :return: True if the paths are disjoint, False otherwise
        """
        # Check if the working and backup paths are disjoint
        if working_path.node_list == backup_path.node_list:
            return False

        # Convert the node lists to sets to check for disjointness
        working_nodes = set(working_path.node_list[1:-1])
        backup_nodes = set(backup_path.node_list)

        working_nodes.isdisjoint(backup_nodes)

    def is_working_disjoint(
        self,
        working_path: Path,
        backup_path: Path,
        core_backup,
        initial_slot_backup,
        number_slots_backup,
    ) -> bool:
        """
        Method that determines if the working path is disjoint from the backup path

        :param working_path: Index of K shortest paths
        :param backup_path: Index of K shortest paths
        :param core_backup: Number of cores currently being used
        :param initial_slot_backup: The current frequency slot being used
        :param number_slots_backup: The total number of slots being used
        """

        def are_working_paths_disjoint(working_path: Path, service_route: Path) -> bool:
            return self.is_disjoint(working_path, service_route)

        def are_backup_paths_shared(
            backup_path: Path,
            service_id: int,
            initial_slots_backup: int,
            number_slots_backup: int,
        ) -> bool:
            for i in range(1, len(backup_path.node_list) - 1):
                link_index = self.topology[backup_path.node_list[i]][
                    backup_path.node_list[i + 1]
                ]["index"]
                spectrum_slice = self.spectrum_slots_allocation[
                    core_backup,
                    link_index,
                    initial_slots_backup : initial_slots_backup + number_slots_backup,
                ]
                if any(service_id in sublist for sublist in spectrum_slice):
                    return True
            return False

        for service in self.topology.graph["running_services"]:
            if not are_working_paths_disjoint(working_path, service.route):
                if are_backup_paths_shared(
                    backup_path,
                    service.service_id,
                    initial_slot_backup,
                    number_slots_backup,
                ):
                    return False

        return True

    def is_path_free(
        self, path: Path, core: int, initial_slot: int, number_slots: int
    ) -> bool:
        """
        Method that determines if the path is free for the core, path, and initial_slot.

        :param core: Number of cores currently being used
        :param path: Index of K shortest paths
        :param initial_slot: The current frequency slot being used <-carlos pls double check
        :param number_slots: The total number of slots

        :return: True/False
        :rtype: bool
        """
        if initial_slot + number_slots > self.num_spectrum_resources:
            # logging.debug('error index' + env.parameters.rsa_algorithm)
            return False

        for i in range(len(path.node_list) - 1):
            if np.any(
                self.topology.graph["available_slots_working"][
                    core,
                    self.topology[path.node_list[i]][path.node_list[i + 1]]["index"],
                    initial_slot : initial_slot + number_slots,
                ]
                == 0
            ):
                return False

    def is_backup_path_free(
        self, path: Path, core: int, initial_slot: int, number_slots: int
    ) -> bool:
        """
        Method that determines if the backup path is free for core, path, initial_slot and number_slots

        :param path: Index of K shortest paths
        :param core: Number of cores currently being used
        :param initial_slot: The current frequency slot being used
        :param number_slots: The total number of slots being used
        """

        if initial_slot + number_slots > self.num_spectrum_resources:
            # logging.debug('error index' + env.parameters.rsa_algorithm)
            return False
        for i in range(1, len(path.node_list) - 1):
            if np.any(
                self.topology.graph["available_slots_backup"][
                    core,
                    self.topology[path.node_list[i]][path.node_list[i + 1]]["index"],
                    initial_slot : initial_slot + number_slots,
                ]
                == 0
            ):
                return False
        return True

    def get_available_slots_working(self, path: Path):
        available_slots_working = functools.reduce(
            np.multiply,
            self.topology.graph["available_slots_working"][
                [
                    self.topology[path.node_list[i]][path.node_list[i + 1]]["id"]
                    for i in range(1, len(path.node_list) - 1)
                ],
                :,
            ],
        )

        return available_slots_working

    def get_available_slots_backup(self, path: Path):
        available_slots_backup = functools.reduce(
            np.multiply,
            self.topology.graph["available_slots_backup"][
                [
                    self.topology[path.node_list[i]][path.node_list[i + 1]]["id"]
                    for i in range(1, len(path.node_list) - 1)
                ],
                :,
            ],
        )

        return available_slots_backup

    def get_available_blocks_working(self, path: Path):
        shortest_path = self.k_shortest_paths[
            self.current_service.source, self.current_service.destination
        ][path]
        modulation = get_best_modulation_format(path.length, self.modulation_formats)

        available_slots = self.get_available_slots_working(shortest_path)

        slots = self.get_number_slots(shortest_path, modulation)

        initial_indices, values, lengths = RMCSAEnv.rle(available_slots)

        available_indices = np.where(values == 1)

        sufficient_indices = np.where(lengths >= slots)

        final_indices = np.intersect1d(available_indices, sufficient_indices)[: self.j]

        return initial_indices[final_indices], lengths[final_indices]

    def observation(self):
        """
        Returns the current observation from the environment.

        For SBPP we need to consider both working and backup paths,
        so we modify the observation to include both matrices.
        """
        # Get working path spectrum state
        spectrum_obs_working = copy.deepcopy(
            self.topology.graph["available_slots_working"]
        )

        # Get backup path spectrum state
        spectrum_obs_backup = copy.deepcopy(
            self.topology.graph["available_slots_backup"]
        )

        # Combine working and backup observations
        spectrum_obs = np.stack([spectrum_obs_working, spectrum_obs_backup])

        return {
            "topology": spectrum_obs,
            "current_service": self.current_service,
        }


# def shortest_available_path_best_modulation_first_core_first_fit(
#     env: RMSCASBPPEnv,
# ) -> int:
#     """
#     Algorithm for determining the shortest available first core first fit path

#     :param env: OpenAI Gym object containing RMCSASBPP environment
#     :return: Cores, paths, and number of spectrum resources
#     """

#     for idp, path in enumerate(
#         env.k_shortest_paths[
#             env.current_service.source, env.current_service.destination
#         ]
#     ):
#         modulation = get_best_modulation_format(path.length, env.modulation_formats)
#         num_slots = env.get_number_slots(path, modulation)

#         for core in range(env.num_spatial_resources):
#             for initial_slot in range(0, env.num_spectrum_resources - num_slots):
#                 if env.is_working_path_free(path, core, initial_slot, num_slots):
#                     midx = env.modulation_formats.index(modulation)
#                     return idp, midx, core, initial_slot

#     return (
#         env.k_paths,
#         env.num_spatial_resources,
#         env.num_spectrum_resources,
#     )


def shortest_available_path_best_modulation_first_core_first_fit(env: RMSCASBPPEnv):
    # Percorre os caminhos de trabalho possíveis
    for working_idp, working_path in enumerate(
        env.k_shortest_paths[
            env.current_service.source, env.current_service.destination
        ]
    ):
        working_modulation = get_best_modulation_format(
            working_path.length, env.modulation_formats
        )
        working_num_slots = env.get_number_slots(working_path, working_modulation)
        working_midx = env.modulation_formats.index(working_modulation)

        # Percorre os núcleos disponíveis para o caminho de trabalho
        for working_core in range(env.num_spatial_resources):
            # Percorre os slots disponíveis para o caminho de trabalho
            for working_initial_slot in range(
                env.num_spectrum_resources - working_num_slots
            ):
                if not env.is_path_free(
                    working_path, working_core, working_initial_slot, working_num_slots
                ):
                    continue  # Slot não disponível no caminho de trabalho

                # Percorre os caminhos de backup possíveis
                for backup_idp, backup_path in enumerate(
                    env.k_shortest_paths[
                        env.current_service.source, env.current_service.destination
                    ]
                ):
                    # Verifica se os caminhos são disjuntos
                    if not env.is_disjoint(working_path, backup_path):
                        continue  # Caminhos não são disjuntos

                    backup_modulation = get_best_modulation_format(
                        backup_path.length, env.modulation_formats
                    )
                    backup_num_slots = env.get_number_slots(
                        backup_path, backup_modulation
                    )
                    backup_midx = env.modulation_formats.index(backup_modulation)

                    # Percorre os núcleos disponíveis para o caminho de backup
                    for backup_core in range(env.num_spatial_resources):
                        # Percorre os slots disponíveis para o caminho de backup
                        for backup_initial_slot in range(
                            env.num_spectrum_resources - backup_num_slots
                        ):
                            if not env.is_backup_path_free(
                                backup_path,
                                backup_core,
                                backup_initial_slot,
                                backup_num_slots,
                            ):
                                continue  # Slot não disponível no caminho de backup

                            # Retorna a ação com os 8 parâmetros necessários
                            return (
                                working_idp,  # Índice do caminho de trabalho
                                working_midx,  # Índice da modulação de trabalho
                                working_core,  # Núcleo de trabalho
                                working_initial_slot,  # Slot inicial de trabalho
                                backup_idp,  # Índice do caminho de backup
                                backup_midx,  # Índice da modulação de backup
                                backup_core,  # Núcleo de backup
                                backup_initial_slot,  # Slot inicial de backup
                            )
    # Se nenhum recurso estiver disponível, retorna a ação de rejeição
    return (
        env.k_paths,  # Índice de rejeição para caminho de trabalho
        len(env.modulation_formats),  # Índice de rejeição para modulação de trabalho
        env.num_spatial_resources,  # Índice de rejeição para núcleo de trabalho
        env.num_spectrum_resources,  # Índice de rejeição para slot de trabalho
        env.k_paths,  # Índice de rejeição para caminho de backup
        len(env.modulation_formats),  # Índice de rejeição para modulação de backup
        env.num_spatial_resources,  # Índice de rejeição para núcleo de backup
        env.num_spectrum_resources,  # Índice de rejeição para slot de backup
    )


class SimpleMatrixObservation(gym.ObservationWrapper):
    def __init__(self, env: RMSCASBPPEnv):
        super().__init__(env)
        # Forma do espaço de observação:
        # - 2 * número de nós (fonte e destino)
        # - 2 * número de enlaces * número de recursos espectrais * número de recursos espaciais
        #   (slots de trabalho e backup)
        shape = (
            self.env.topology.number_of_nodes() * 2
            + self.env.topology.number_of_edges()
            * self.env.num_spectrum_resources
            * self.env.num_spatial_resources
            * 2  # multiplicado por 2 para working e backup
        )

        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            dtype=np.uint8,
            shape=(shape,),
        )
        self.action_space = env.action_space

    def observation(self, observation):
        # Cria vetor one-hot para nós fonte e destino
        source_destination_tau = np.zeros((2, self.env.topology.number_of_nodes()))
        min_node = min(
            self.env.current_service.source_id, self.env.current_service.destination_id
        )
        max_node = max(
            self.env.current_service.source_id, self.env.current_service.destination_id
        )
        source_destination_tau[0, min_node] = 1
        source_destination_tau[1, max_node] = 1

        # Obtém estados dos slots de trabalho e backup
        spectrum_obs_working = copy.deepcopy(
            self.env.topology.graph["available_slots_working"]
        )
        spectrum_obs_backup = copy.deepcopy(
            self.env.topology.graph["available_slots_backup"]
        )

        # Concatena todas as informações em um único vetor
        return np.concatenate(
            (
                source_destination_tau.reshape(
                    (1, np.prod(source_destination_tau.shape))
                ),
                spectrum_obs_working.reshape((1, np.prod(spectrum_obs_working.shape))),
                spectrum_obs_backup.reshape((1, np.prod(spectrum_obs_backup.shape))),
            ),
            axis=1,
        ).reshape(self.observation_space.shape)
