import botbowl
import numpy as np
from botbowl import BotBowlEnv, EnvConf
# installed from my pip
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.models import ModelCatalog
from ray.tune import register_env
# custom code
from yay_models import IMPALANetFixed
from yay_multi_agent_wrapper import MultiAgentBotBowlEnv

# constants
MA_POLICY_ID = "main"
RESTORE_PATH = "yay_checkpoint/checkpoint-1000"

class YayBotSubmission(botbowl.Agent):
    env: BotBowlEnv

    def __init__(self, name,
                 env_conf: EnvConf,
                 policy_id=DEFAULT_POLICY_ID,
                 ):
        super().__init__(name)
        self.env = BotBowlEnv(env_conf)
        self.action_queue = []
        '''
        name: botbowl bots need a name
        env_conf: needed for botbowl env settings
        trainer: loaded ray trainer to use for inference. If set to None random action is taken
        policy_id: used by ray trainer to read the correct policy for instance
        '''
        self.trainer = None
        self.policy_id = policy_id

    def new_game(self, game, team):
        self.my_team = team

    def act(self, game):
        # i'm not quite getting why there's a queue but I guess sometimes action idx turn into chained actions?
        if len(self.action_queue) > 0:
            return self.action_queue.pop(0)

        self.env.game = game
        spatial_obs, non_spatial_obs, action_mask = self.env.get_state()

        aa = np.where(action_mask > 0.0)[0]
        obs_dict = self._make_obs(spatial_obs, non_spatial_obs, action_mask)

        # trainer is None when playing against random bot
        if self.trainer is not None:
            action_idx = self.trainer.compute_single_action(obs_dict, policy_id=self.policy_id)
            if action_idx not in aa:
                print("ERROR: action not valid, choosing random valid action")
                action_idx = np.random.choice(aa, 1)[0]
        else:
            # doing a random action
            action_idx = np.random.choice(aa, 1)[0]

        action_objects = self.env._compute_action(action_idx)
        self.action_queue = action_objects

        return self.action_queue.pop(0)

    def end_game(self, game):
        pass

    def _make_obs(self, spatial_obs, non_spatial_obs, action_mask):
        # could do none check
        obs_dict = {
            'spatial': spatial_obs.astype("float32"),
            'non_spatial': non_spatial_obs.astype("float32"),
            'action_mask': action_mask.astype("float32"),
        }
        return obs_dict


def get_env_config_for_ray_wrapper(botbowl_size, combine_obs=False, reward_function='TDReward', is_multi_agent_wrapper=True):
    test_env = BotBowlEnv(env_conf=EnvConf(size=botbowl_size))
    reset_obs = test_env.reset()
    del test_env
    if is_multi_agent_wrapper:
        away_agent_type='human'
    else:
        away_agent_type='random'
    env_config = {
        "env": BotBowlEnv(env_conf=EnvConf(size=botbowl_size), home_agent='human',
                         away_agent=away_agent_type),
        "reset_obs": reset_obs,
        "combine_obs": bool(combine_obs),
        "reward_type": reward_function,
    }

    return env_config

def get_ray_trainer(model_config, is_ma_bot=True):
    # if not is_ma_bot:
    #     pass
    # else:
    # load from multi agent
    policy_id = MA_POLICY_ID
    dummy_ray_env = "ma_botbowl_env"
    rc_policies = {
        MA_POLICY_ID: PolicySpec(),
    }
    def rc_policy_mapping_fn(agent_id, episode, worker, **kwargs):
        return MA_POLICY_ID

    restore_config = {
        "framework": "torch",
        "model": model_config,
        "multiagent": {
            "policies": rc_policies,
            "policy_mapping_fn": rc_policy_mapping_fn,
        },
        "num_workers": 1,
    }

    my_trainer = PPOTrainer(config=restore_config, env=dummy_ray_env)
    my_trainer.restore(RESTORE_PATH)

    return my_trainer, policy_id

def get_bot_policy(bot, model_config):
    bot.trainer, bot.policy_id = get_ray_trainer(model_config)
    return bot

def _make_my_bot(name):
    # create the model config here
    my_model_config = {
        "custom_model": "CustomNN",
        "custom_model_config": {
        }
    }
    ModelCatalog.register_custom_model("CustomNN", IMPALANetFixed)

    # create and register ray wrapper for botbowl envs
    ma_env_config = get_env_config_for_ray_wrapper(11, is_multi_agent_wrapper=True)
    register_env("ma_botbowl_env", lambda _: MultiAgentBotBowlEnv(**ma_env_config))

    # work in restore_ray_trainer
    # make a dynamic path maybe? maybe at top of file?
    yaybot = YayBotSubmission(name=name, env_conf=EnvConf(size=11))
    yaybot = get_bot_policy(yaybot, my_model_config)
    return yaybot

# Register the bot to the framework
# botbowl.register_bot('yay-bot', YayBotSubmission)
botbowl.register_bot("yay-bot", _make_my_bot)