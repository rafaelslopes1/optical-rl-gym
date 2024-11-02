from gym.envs.registration import register

register(
    id="RWA-v0",
    entry_point="optical_rl_gym.envs:RWAEnv",
)

register(
    id="RMSA-v0",
    entry_point="optical_rl_gym.envs:RMSAEnv",
)

register(
    id="RMSADPP-v0",
    entry_point="optical_rl_gym.envs:RMSADPPEnv",
)

register(
    id="RMSASBPP-v0",
    entry_point="optical_rl_gym.envs:RMSASBPPEnv",
)

register(
    id="DeepRMSA-v0",
    entry_point="optical_rl_gym.envs:DeepRMSAEnv",
)

register(
    id="DeepRMSADPP-v0",
    entry_point="optical_rl_gym.envs:DeepRMSADPPEnv",
)

register(
    id="DeepRMSADPPKSP-v0",
    entry_point="optical_rl_gym.envs:DeepRMSADPPKSPEnv",
)

register(
    id="DeepRMSASBPP-v0",
    entry_point="optical_rl_gym.envs:DeepRMSASBPPEnv",
)

register(
    id="DeepRMSASBPPKSP-v0",
    entry_point="optical_rl_gym.envs:DeepRMSASBPPKSPEnv",
)

register(
    id="QoSConstrainedRA-v0",
    entry_point="optical_rl_gym.envs:QoSConstrainedRA",
)

register(
    id="RMCSA-v0",
    entry_point="optical_rl_gym.envs:RMCSAEnv",
)