import copy
import functools
import heapq
import math
from collections import defaultdict
from typing import Dict, Optional, Sequence, Tuple, List
from datetime import datetime
import csv

import gym
import gym.spaces
import networkx as nx
import numpy as np

from optical_rl_gym.utils import Modulation, Path, Service, get_best_modulation_format
from .rmcsa_env import RMCSAEnv
import logging  # Adicionado


class RMCSADPPSparseEnv(RMCSAEnv):

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
            "failure_crosstalk",
            "episode_failure_crosstalk",
        ]
    }

    def __init__(
        self,
        topology: nx.Graph = None,
        episode_length: int = 1000,
        load: float = 10,
        mean_service_holding_time: float = 10800.0,
        num_spectrum_resources: int = 100,
        num_spatial_resources: int = 7,  # number of cores
        modulation_formats=None,
        worst_xt=None,
        node_request_probabilities=None,
        bit_rate_selection: str = "continuous",
        bit_rates: Sequence[int] = None,
        bit_rate_probabilities: Optional[np.array] = None,
        bit_rate_lower_bound=25,
        bit_rate_higher_bound=100,
        seed=None,
        allow_rejection=False,
        reset=True,
        channel_width: float = 12.5,
        # Parâmetros adicionais para DPP
        # protection=True,
    ):
        # Inicialização do arquivo CSV de log
        self.csv_file = open(f"rmcsa_actions_log_{seed}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv", mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            'timestamp',
            'action',
            'working_path_nodes',
            'backup_path_nodes',
            'working_path_length',
            'backup_path_length',
            'working_modulation',
            'backup_modulation',
            'working_core',
            'backup_core',
            'working_slots_range',
            'backup_slots_range'
        ])

        super().__init__(
            topology=topology,
            episode_length=episode_length,
            load=load,
            mean_service_holding_time=mean_service_holding_time,
            num_spectrum_resources=num_spectrum_resources,
            num_spatial_resources=num_spatial_resources,
            modulation_formats=modulation_formats,
            worst_xt=worst_xt,
            node_request_probabilities=node_request_probabilities,
            bit_rate_selection=bit_rate_selection,
            bit_rates=bit_rates,
            bit_rate_probabilities=bit_rate_probabilities,
            bit_rate_lower_bound=bit_rate_lower_bound,
            bit_rate_higher_bound=bit_rate_higher_bound,
            seed=seed,
            allow_rejection=allow_rejection,
            reset=reset,
            channel_width=channel_width,
        )

        # Failure related variables
        self.failure_counter = 0
        self.episode_failure_counter = 0
        self.failure_disjointness = 0
        self.episode_failure_disjointness = 0
        self.failure_slots = 0
        self.episode_failure_slots = 0
        self.failure_crosstalk = 0
        self.episode_failure_crosstalk = 0
        self.failure_modulation = 0
        self.episode_failure_modulation = 0

        # Resilience related variables
        self.dpp_counter = 0
        self.episode_dpp_counter = 0

        # Performance related variables
        self.compactness = 0
        self.throughput = 0
        self.available_slots = 0
        
        # defining the observation and action spaces
        default_actions_shape = (
            # working path
            self.k_paths + 1,
            len(self.modulation_formats) + 1,
            self.num_spatial_resources + 1,
            self.num_spectrum_resources + 1,
            # protection path
            self.k_paths + 1,
            len(self.modulation_formats) + 1,
            self.num_spatial_resources + 1,
            self.num_spectrum_resources + 1,
        )

        self.actions_output = np.zeros(default_actions_shape, dtype=int)
        self.episode_actions_output = np.zeros(default_actions_shape, dtype=int)
        self.actions_taken = np.zeros(default_actions_shape, dtype=int)
        self.episode_actions_taken = np.zeros(default_actions_shape, dtype=int)

        self.action_space = gym.spaces.MultiDiscrete(
            (
                # working path
                self.k_paths + self.reject_action,
                len(self.modulation_formats),
                self.num_spatial_resources + self.reject_action,
                self.num_spectrum_resources + self.reject_action,
                # protection path
                self.k_paths + self.reject_action,
                len(self.modulation_formats),
                self.num_spatial_resources + self.reject_action,
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

        # Configuração do logger
        self.logger = logging.getLogger("rmcsa_env_dpp")
        self.logger.setLevel(logging.DEBUG)  # Definir nível de log desejado

        handler = logging.FileHandler(
            f"rmcsa_env_dpp_dense_{seed}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
        )
        csv_formatter = logging.Formatter(
            '%(asctime)s,%(levelname)s,%(message)s'
        )
        handler.setFormatter(csv_formatter)
        self.logger.addHandler(handler)

        self._new_service = None
        if reset:
            self.reset(only_episode_counters=False)

    def step(self, action: List[int]):
        """
        Executa uma ação que provisiona caminhos de trabalho e proteção para um serviço.

        Args:
            action: Lista com [working_path, working_modulation, working_core, working_initial_slot,
                             backup_path, backup_modulation, backup_core, backup_initial_slot]
        """

        # Desempacota os parâmetros da ação
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

        # Atualiza contadores de ações tentadas
        self._update_action_counters(action)

        # Valida os parâmetros da ação
        if not self._validate_action_parameters(action):
            self._handle_invalid_action("params")
            return self._get_step_result()

        # Obtém os objetos dos caminhos e calcula slots necessários
        working_path_obj = self._get_path_object(working_path)
        backup_path_obj = self._get_path_object(backup_path)

        working_slots = self.get_number_slots(
            working_path_obj, self.modulation_formats[working_modulation]
        )
        backup_slots = self.get_number_slots(
            backup_path_obj, self.modulation_formats[backup_modulation]
        )

        rejection_reason = None
        rejected = False

        # Verifica disjunção dos caminhos
        if not self.is_disjoint(working_path_obj, backup_path_obj):
            rejected = True
            rejection_reason = "disjointness"
        else:
            self.current_service.is_disjoint = True

        # Tenta provisionar o caminho de trabalho
        if not self.is_path_free(
            working_path_obj, working_core, working_initial_slot, working_slots
        ):
            if rejection_reason is None:
                rejected = True
                rejection_reason = "slots"
        else:
            self.current_service.slots_assigned = True

        # Tenta provisionar o caminho de backup
        if not self.is_path_free(
            backup_path_obj, backup_core, backup_initial_slot, backup_slots
        ):
            if rejection_reason is None:
                rejected = True
                rejection_reason = "slots"
        else:
            self.current_service.slots_assigned_backup = True

        # Verifica restrições de crosstalk
        # if not self.validate_crosstalk(
        #     working_path_obj, working_modulation
        # ) or not self.validate_crosstalk(
        #     backup_path_obj, backup_modulation
        # ):
        #     if rejection_reason is None:
        #         rejected = True
        #         rejection_reason = "crosstalk"
        # else:
        #     self.current_service.is_crosstalk_acceptable = True
        
        # Verifica restrições de modulação
        if not self.validate_modulation(working_path_obj, self.modulation_formats[working_modulation]) or not self.validate_modulation(backup_path_obj, self.modulation_formats[backup_modulation]):
            if rejection_reason is None:
                rejected = True
                rejection_reason = "modulation"
        else:
            self.current_service.is_modulation_acceptable = True

        # Se a ação foi rejeitada, atualiza contadores e registra log
        if rejected:
            # print(f"Ação rejeitada - {action} - {rejection_reason}")
            self._handle_invalid_action(rejection_reason)
            self.logger.warning(
                "Ação rejeitada: %s, Razão: %s, Failures - Disjointness: %s, Slots: %s, Crosstalk: %s, Modulation: %s",
                action, rejection_reason, self.failure_disjointness, self.failure_slots, self.failure_crosstalk, self.failure_modulation
            )
        else:
            # Provisiona os caminhos
            self._provision_paths(
                working_path_obj,
                working_modulation,
                working_core,
                working_initial_slot,
                working_slots,
                backup_path_obj,
                backup_modulation,
                backup_core,
                backup_initial_slot,
                backup_slots,
            )
            self.logger.info(
                "Ação aceita: %s, Caminho de Trabalho: %s, Modulação: %s, Núcleo: %s, Slot Inicial: %s, Slots: %s, Caminho de Backup: %s, Modulação Backup: %s, Núcleo Backup: %s, Slot Inicial Backup: %s, Slots Backup: %s",
                action,
                working_path_obj.node_list,
                working_modulation,
                working_core,
                working_initial_slot,
                working_slots,
                backup_path_obj.node_list,
                backup_modulation,
                backup_core,
                backup_initial_slot,
                backup_slots,
            )
            self.logger.debug(
                "Provisionamento completo para serviço ID: %s",
                self.current_service.service_id
            )

        # Registro adicional no CSV
        self.csv_writer.writerow([
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            action,
            working_path_obj.node_list,
            backup_path_obj.node_list,
            len(working_path_obj.node_list),
            len(backup_path_obj.node_list),
            working_modulation,
            backup_modulation,
            working_core,
            backup_core,
            f"{working_initial_slot}-{working_initial_slot + working_slots}",
            f"{backup_initial_slot}-{backup_initial_slot + backup_slots}",
            f"{self.bit_rate_requested}",
            f"{self.bit_rate_provisioned}",
        ])

        # Preparar o registro de log como um dicionário
        log_record = {
            "action": [int(value) for value in action],
            "working_path": [int(node) for node in working_path_obj.node_list],
            "backup_path": [int(node) for node in backup_path_obj.node_list],
            "failures": self.failure_counter,
            "failure_disjointness": self.failure_disjointness,
            "failure_slots": self.failure_slots,
            "failure_crosstalk": self.failure_crosstalk,
            "failure_modulation": self.failure_modulation,
            "is_rejected": rejected,
            "rejection_reason": rejection_reason,
            "reward": self.reward(),
        }

        # Escrever o registro usando o logger padrão
        self.logger.debug(str(log_record))

        step_result = self._get_step_result()
        
        self._new_service = False
        self._next_service()
        
        return step_result

    def _update_action_counters(self, action):
        """Atualiza contadores das ações executadas"""
        self.actions_output[tuple(action)] += 1

    def _update_invalid_action_taken_counters(self):
        self._update_action_taken_counters(
            [
                self.k_paths,
                len(self.modulation_formats),
                self.num_spatial_resources,
                self.num_spectrum_resources,
                self.k_paths,
                len(self.modulation_formats),
                self.num_spatial_resources,
                self.num_spectrum_resources,
            ]
        )

    def _update_action_taken_counters(self, action):
        """Atualiza contadores das ações tomadas"""
        self.actions_taken[tuple(action)] += 1

    def _validate_action_parameters(self, action) -> bool:
        """Valida se os parâmetros da ação estão dentro dos limites permitidos"""
        w_path, w_mod, w_core, w_slot, b_path, b_mod, b_core, b_slot = action
        return (
            w_path < self.k_paths
            and w_mod < len(self.modulation_formats)
            and w_core < self.num_spatial_resources
            and w_slot < self.num_spectrum_resources
            and b_path < self.k_paths
            and b_mod < len(self.modulation_formats)
            and b_core < self.num_spatial_resources
            and b_slot < self.num_spectrum_resources
        )

    def _handle_invalid_action(self, reason: str):
        """Trata uma ação inválida"""
        self.current_service.accepted = False
        self.failure_counter += 1
        self.episode_failure_counter += 1

        self._update_invalid_action_taken_counters()

        if reason == "slots":
            self.failure_slots += 1
            self.episode_failure_slots += 1
            self.current_service.slots_assigned = False
            self.current_service.slots_assigned_backup = False
        elif reason == "disjointness":
            self.failure_disjointness += 1
            self.episode_failure_disjointness += 1
            self.current_service.is_disjoint = False
        elif reason == "crosstalk":
            self.failure_crosstalk += 1
            self.episode_failure_crosstalk += 1
            self.current_service.is_crosstalk_acceptable = False
        elif reason == "modulation":
            self.failure_modulation += 1
            self.episode_failure_modulation += 1
            self.current_service.is_modulation_acceptable = False

    def _get_path_object(self, path_index: int) -> Path:
        """Retorna o objeto Path para o índice fornecido"""
        return self.k_shortest_paths[
            self.current_service.source, self.current_service.destination
        ][path_index]

    def validate_crosstalk(self, path: Path, modulation_index: int) -> bool:
        """Valida as restrições de crosstalk para ambos os caminhos"""
        return self._crosstalk_is_acceptable(
            self.modulation_formats[modulation_index], path.length
        )

    def _provision_paths(
        self,
        working_path: Path,
        working_modulation: int,
        working_core: int,
        working_slot: int,
        working_slots: int,
        backup_path: Path,
        backup_modulation: int,
        backup_core: int,
        backup_slot: int, 
        backup_slots: int,
    ):
        """Provisiona os caminhos de trabalho e backup"""
        self._provision_path(
            working_path,
            working_modulation,
            working_core,
            working_slot,
            working_slots,
            backup_path,
            backup_modulation,
            backup_core,
            backup_slot,
            backup_slots,
        )
        self.current_service.accepted = True
        self._add_release(self.current_service)

    def _get_step_result(self):
        """Retorna o resultado do passo"""
        self._update_service_counters()
        self._update_network_metrics()

        info = self._get_step_info()

        return (
            self.observation(),
            self.reward(),
            self.episode_services_processed == self.episode_length,
            info,
        )

    def _update_service_counters(self):
        """Atualiza contadores relacionados ao serviço"""
        self.services_processed += 1
        self.episode_services_processed += 1
        self.bit_rate_requested += self.current_service.bit_rate
        self.episode_bit_rate_requested += self.current_service.bit_rate

    def _update_network_metrics(self):
        """Atualiza métricas da rede"""
        self.compactness = self.topology.graph["compactness"]
        self.throughput = self.topology.graph["throughput"]
        self.available_slots = np.sum(self.topology.graph["available_slots"])
        self.topology.graph["services"].append(self.current_service)

    def _get_step_info(self) -> dict:
        """Retorna as informações do passo atual"""
        return {
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
            "failure_crosstalk": self.failure_crosstalk / self.services_processed,
            "episode_failure_crosstalk": self.episode_failure_crosstalk
            / self.episode_services_processed,
            "failure_modulation": self.failure_modulation / self.services_processed,
            "episode_failure_modulation": self.episode_failure_modulation
            / self.episode_services_processed,
            "compactness": self.compactness,
            "throughput": self.throughput,
            "available_slots": self.available_slots,
        }

    def reset(self, only_episode_counters=True):
        # Resetting the counters and statistics
        self.episode_bit_rate_requested = 0
        self.episode_bit_rate_provisioned = 0

        self.episode_services_processed = 0
        self.episode_services_accepted = 0

        self.episode_failure_counter = 0
        self.episode_failure_disjointness = 0
        self.episode_failure_slots = 0
        self.episode_failure_crosstalk = 0
        self.episode_failure_modulation = 0

        default_actions_shape = (
            # working path
            self.k_paths + 1,
            len(self.modulation_formats) + 1,
            self.num_spatial_resources + 1,
            self.num_spectrum_resources + 1,
            # protection path
            self.k_paths + 1,
            len(self.modulation_formats) + 1,
            self.num_spatial_resources + 1,
            self.num_spectrum_resources + 1,
        )

        self.episode_actions_output = np.zeros(default_actions_shape, dtype=int)
        self.episode_actions_taken = np.zeros(default_actions_shape, dtype=int)

        if only_episode_counters:
            return self.observation()
        else:
            # Reconfigurar o handler do logger para um novo arquivo
            for handler in self.logger.handlers[:]:
                self.logger.removeHandler(handler)
                handler.close()
            now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            handler = logging.FileHandler(f"rmcsa_env_dpp_dense_{self.rand_seed}_{now}.csv")
            csv_formatter = logging.Formatter(
                '%(asctime)s,%(levelname)s,%(message)s'
            )
            handler.setFormatter(csv_formatter)
            self.logger.addHandler(handler)
            self.logger.info("Reset do ambiente realizado.")

            # Resetar o arquivo CSV de log
            if hasattr(self, 'csv_file') and self.csv_file:
                self.csv_file.close()
            self.csv_file = open(f"rmcsa_actions_log_{self.rand_seed}_{now}.csv", mode='w', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow([
                'timestamp',
                'action',
                'working_path_nodes',
                'backup_path_nodes',
                'working_path_length',
                'backup_path_length',
                'working_modulation',
                'backup_modulation',
                'working_core',
                'backup_core',
                'working_slots_range',
                'backup_slots_range'
            ])

        super().reset(only_episode_counters=only_episode_counters)
        
        self.bit_rate_requested = 0
        self.bit_rate_provisioned = 0
        
        self.failure_counter = 0
        self.failure_disjointness = 0
        self.failure_slots = 0
        self.failure_crosstalk = 0
        self.failure_modulation = 0
        
        self.actions_output = np.zeros(default_actions_shape, dtype=int)
        self.actions_taken = np.zeros(default_actions_shape, dtype=int)

        self.topology.graph["available_slots"] = np.ones(
            (
                self.num_spatial_resources,
                self.topology.number_of_edges(),
                self.num_spectrum_resources,
            ),
            dtype=int,
        )

        self.spectrum_slots_allocation = [
            [
                [[] for _ in range(self.num_spectrum_resources)]
                for _ in range(self.topology.number_of_edges())
            ]
            for _ in range(self.num_spatial_resources)
        ]

        self.topology.graph["compactness"] = 0.0
        self.topology.graph["throughput"] = 0.0

        self.topology.graph["services"] = []
        self.topology.graph["running_services"] = []
        self.topology.graph["last_update"] = 0.0

        for _, link in enumerate(self.topology.edges()):
            self.topology[link[0]][link[1]]["running_services"] = []
            self.topology[link[0]][link[1]]["services"] = []
            self.topology[link[0]][link[1]]["last_update"] = 0.0
            self.topology[link[0]][link[1]]["utilization"] = 0.0
            self.topology[link[0]][link[1]]["external_fragmentation"] = 0.0
            self.topology[link[0]][link[1]]["compactness"] = 0.0

            # self.topology[link[0]][link[1]]["crosstalk"] = 0 # check if it is necessary

        self._new_service = False
        self._next_service()
        return self.observation()

    def __del__(self):
        # Fechar o arquivo CSV ao deletar a instância, se existir
        if hasattr(self, 'csv_file') and self.csv_file:
            self.csv_file.close()
        for handler in self.logger.handlers:
            handler.close()

    def _provision_path(
        self,
        working_path: Path,
        working_modulation: int,
        core_working: int,
        initial_slot_working: int,
        number_slots_working: int,
        backup_path: Path,
        backup_modulation: int,
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
        if not self.is_path_free(
            backup_path, core_backup, initial_slot_backup, number_slots_backup
        ):
            raise ValueError(
                f"Backup path {backup_path.node_list} has not enough slots available on core {core_backup} from {initial_slot_backup} to {initial_slot_backup + number_slots_backup}"
            )

        # Provision the working path
        for i in range(len(working_path.node_list) - 1):
            link_index = self.topology[working_path.node_list[i]][
                working_path.node_list[i + 1]
            ]["index"]

            self.topology.graph["available_slots"][
                core_working,
                link_index,
                initial_slot_working : initial_slot_working + number_slots_working,
            ] = 0

            allocated_spectrum_slots_working = self.spectrum_slots_allocation[core_working][
                link_index
            ][initial_slot_working : initial_slot_working + number_slots_working]

            for slot in allocated_spectrum_slots_working:
                slot.append(self.current_service.service_id)

            self.logger.debug(
                f"Provisionando caminho de trabalho - Serviço ID: {self.current_service.service_id}, "
                f"Link: {link_index}, Núcleo: {core_working}, "
                f"Slots: {initial_slot_working}-{initial_slot_working + number_slots_working}"
            )

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
        for i in range(len(backup_path.node_list) - 1):
            link_index = self.topology[backup_path.node_list[i]][
                backup_path.node_list[i + 1]
            ]["index"]

            self.topology.graph["available_slots"][
                core_backup,
                link_index,
                initial_slot_backup : initial_slot_backup + number_slots_backup,
            ] = 0

            allocated_spectrum_slots_backup = self.spectrum_slots_allocation[core_backup][
                link_index
            ][initial_slot_backup : initial_slot_backup + number_slots_backup]

            for slot in allocated_spectrum_slots_backup:
                slot.append(self.current_service.service_id)

            self.logger.debug(
                f"Provisionando caminho de backup - Serviço ID: {self.current_service.service_id}, "
                f"Link: {link_index}, Núcleo: {core_backup}, "
                f"Slots: {initial_slot_backup}-{initial_slot_backup + number_slots_backup}"
            )

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
        self.current_service.working_path = working_path
        self.current_service.working_modulation = self.modulation_formats[working_modulation]
        self.current_service.working_core = core_working
        self.current_service.working_initial_slot = initial_slot_working
        self.current_service.working_number_slots = number_slots_working

        self.current_service.backup_path = backup_path
        self.current_service.backup_modulation = self.modulation_formats[backup_modulation]
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
        for i in range(len(service.working_path.node_list) - 1):
            link_index = self.topology[service.working_path.node_list[i]][
                service.working_path.node_list[i + 1]
            ]["index"]

            self.topology.graph["available_slots"][
                service.working_core,
                link_index,
                service.working_initial_slot : service.working_initial_slot + service.working_number_slots,
            ] = 1

            allocated_spectrum_slots_working = self.spectrum_slots_allocation[service.working_core][
                link_index
            ][service.working_initial_slot : service.working_initial_slot + service.working_number_slots]

            self.logger.debug(
                f"Liberação de caminho de trabalho - Serviço ID: {service.service_id}, "
                f"Link: {link_index}, Núcleo: {service.working_core}, "
                f"Slots: {service.working_initial_slot}-{service.working_initial_slot + service.working_number_slots}"
            )

            for slot in allocated_spectrum_slots_working:
                slot.remove(service.service_id)

            self.topology[service.working_path.node_list[i]][service.working_path.node_list[i + 1]][
                "services"
            ].remove(service)
            self.topology[service.working_path.node_list[i]][service.working_path.node_list[i + 1]][
                "running_services"
            ].remove(service)
            self._update_link_stats(
                service.working_core,
                service.working_path.node_list[i],
                service.working_path.node_list[i + 1],
            )

        # Release the backup path
        for i in range(len(service.backup_path.node_list) - 1):
            link_index = self.topology[service.backup_path.node_list[i]][
                service.backup_path.node_list[i + 1]
            ]["index"]

            self.topology.graph["available_slots"][
                service.backup_core,
                link_index,
                service.backup_initial_slot : service.backup_initial_slot
                + service.backup_number_slots,
            ] = 1

            allocated_spectrum_slots_backup = self.spectrum_slots_allocation[
                service.backup_core
            ][link_index][
                service.backup_initial_slot : service.backup_initial_slot
                + service.backup_number_slots
            ]

            self.logger.debug(
                f"Liberação de caminho de backup - Serviço ID: {service.service_id}, "
                f"Link: {link_index}, Núcleo: {service.backup_core}, "
                f"Slots: {service.backup_initial_slot}-{service.backup_initial_slot + service.backup_number_slots}"
            )

            for slot in allocated_spectrum_slots_backup:
                slot.remove(service.service_id)

            self.topology[service.backup_path.node_list[i]][
                service.backup_path.node_list[i + 1]
            ]["services"].remove(service)
            self.topology[service.backup_path.node_list[i]][
                service.backup_path.node_list[i + 1]
            ]["running_services"].remove(service)
            self._update_link_stats(
                service.backup_core,
                service.backup_path.node_list[i],
                service.backup_path.node_list[i + 1],
            )

        self.topology.graph["running_services"].remove(service)
        self._update_network_stats(service.working_core)
        self._update_network_stats(service.backup_core)

    def is_disjoint(self, working_path: Path, backup_path: Path) -> bool:
        """Verifica se os caminhos são disjuntos"""
        if working_path.node_list == backup_path.node_list:
            return False

        # Verifica se os caminhos compartilham algum nó (não considera a origem e o destino)
        for node in working_path.node_list[1:-1]:
            if node in backup_path.node_list[1:-1]:
                return False

        return True

    def calculate_modulation_fitness(self, path: Path, modulation: Modulation) -> float:
        sorted_modulations = sorted(
            self.modulation_formats, key=lambda x: x.spectral_efficiency, reverse=True
        )
        
        best_modulation = get_best_modulation_format(path.length, self.modulation_formats)
        
        # calcula um valor de fitness para a modulação com base na distância do index de melhor formato de modulação
        # quanto mais distante, pior é o fitness, mas se o formato de modulação escolhido tiver um alcance menor que 
        # o melhor formato de modulação, a penalidade é maior
        index_diff = abs(sorted_modulations.index(modulation) - sorted_modulations.index(best_modulation))
        length_diff = 1 if modulation.maximum_length < best_modulation.maximum_length else 0
        return index_diff + length_diff

    def validate_modulation(self, path: Path, modulation: Modulation) -> bool:
        """Valida a modulação para o caminho fornecido"""
        return modulation.maximum_length >= path.length

    def reward(self):
        return 1 if self.current_service.accepted else -1

def shortest_available_path_best_modulation_first_core_first_fit(env: RMCSADPPSparseEnv):
    # print("shortest_available_path_best_modulation_first_core_first_fit")
    # Percorre os caminhos de trabalho possíveis
    for working_idp, working_path in enumerate(
        env.k_shortest_paths[
            env.current_service.source, env.current_service.destination
        ]
    ):
        # print("working_idp:", working_idp)
        working_modulation = get_best_modulation_format(
            working_path.length, env.modulation_formats
        )
        working_num_slots = env.get_number_slots(working_path, working_modulation)
        working_midx = env.modulation_formats.index(working_modulation)
        
        # if not env.validate_crosstalk(working_path, working_midx):
        #     continue  # Restrição de crosstalk violada

        # Percorre os núcleos disponíveis para o caminho de trabalho
        for working_core in range(env.num_spatial_resources):
            # Percorre os slots disponíveis para o caminho de trabalho
            # print("\tworking_core:", working_core)
            for working_initial_slot in range(
                env.num_spectrum_resources - working_num_slots
            ):
                # print("\t\tworking_initial_slot:", working_initial_slot)
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
                    # print("\tbackup_idp:", backup_idp)
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
                    
                    # if not env.validate_crosstalk(backup_path, backup_midx):
                    #     continue  # Restrição de crosstalk violada

                    # Percorre os núcleos disponíveis para o caminho de backup
                    for backup_core in range(env.num_spatial_resources):
                        # print("\t\tbackup_core:", backup_core)

                        # Percorre os slots disponíveis para o caminho de backup
                        for backup_initial_slot in range(
                            env.num_spectrum_resources - backup_num_slots
                        ):
                            # print("\t\t\tbackup_initial_slot:", backup_initial_slot)
                            if not env.is_path_free(
                                backup_path,
                                backup_core,
                                backup_initial_slot,
                                backup_num_slots,
                            ):
                                continue  # Slot não disponível no caminho de backup
                            
                            # print("Ação válida")
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
    
    # print("Nenhuma ação válida")
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

    # ... outros métodos específicos para DPP ...


class SimpleMatrixObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        shape = (
            self.env.topology.number_of_nodes() * 2
            + self.env.topology.number_of_edges()
            * self.env.num_spatial_resources
            * self.env.num_spectrum_resources
        )

        self.observation_space = gym.spaces.Box(
            low=0, high=1, dtype=np.uint8, shape=(shape,)
        )
        self.action_space = env.action_space

    def observation(self, observation):
        source_destination_tau = np.zeros((2, self.env.topology.number_of_nodes()))
        min_node = min(
            self.env.current_service.source_id, self.env.current_service.destination_id
        )
        max_node = max(
            self.env.current_service.source_id, self.env.current_service.destination_id
        )
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
