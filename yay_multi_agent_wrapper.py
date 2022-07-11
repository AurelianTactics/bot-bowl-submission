import numpy as np
from botbowl import RewardWrapper
from yay_rewards import A2C_Reward, TDReward
from gym.spaces import Discrete, Box
import gym
from ray.rllib.env.multi_agent_env import MultiAgentEnv
#from yay_utils import get_save_path_and_make_save_directory
import time
import pickle


class MultiAgentBotBowlEnv(MultiAgentEnv):
    def __init__(self, env, reset_obs, combine_obs=False, reward_type="TDReward", debug_mode=False):
        super().__init__()

        self.mask = None
        self.env = env
        self.combine_obs = combine_obs
        self.debug_mode = debug_mode # debug checking inputs
        if self.debug_mode:
            file_dir = "/ray_results/ma_debug/"
            file_name = "debug_inputs_ma_{}".format(int(time.time()))
            # self.debug_save_path = get_save_path_and_make_save_directory(file_name, file_dir)
            self.debug_save_path = None # disabling this for now to make loading for sumission easier
        else:
            self.debug_save_path = None

        # reset_obs has the size of the observations
        # combine obs is whether to combine the spatial with the non-spatial or not
        spatial_obs, non_spatial_obs, mask = reset_obs

        if self.combine_obs:
            spaces = {
                'flat_s_ns': Box(0.0, 1.0, (spatial_obs.flatten().shape[0] + non_spatial_obs.flatten().shape[0],), "float32"),
                'action_mask': Box(0.0, 1.0, (mask.shape), "float32"),
            }
        else:
            spaces = {
                'spatial': Box(0.0, 1.0, (spatial_obs.shape), "float32"),
                'non_spatial': Box(0.0, 2.0, (non_spatial_obs.shape), "float32"),
                'action_mask': Box(0.0, 1.0, (mask.shape), "float32"), # should be 1.0 but having an issue in 1v1 where sometimes it's greater than 1
            }
        # open spiel uses this but I don't think I want to
        # https://github.com/ray-project/ray/blob/master/rllib/utils/pre_checks/env.py#L37
        # I think I have to do this since my obs is nested but base obs is not?
        self._skip_env_checking = True

        # Agent IDs are ints, starting from 0.
        self.num_agents = 2  # home is 0, away is 1.
        # I don't think i need to switch sides since obs will flip automatically

        # default is nested with 41 starting
        self.action_space = Discrete(mask.shape[0])
        # default is a box of only the spatial # Box(0.0, 1.0, (44, 5, 6), float32) for 1v1 hxw differ by env
        self.observation_space = gym.spaces.Dict(spaces)
        # setting reward function
        if reward_type == "A2C_Reward":
            my_reward_func = A2C_Reward()
        else:
            my_reward_func = TDReward()
        self.env = RewardWrapper(self.env, my_reward_func)

    def reset(self):
        (spatial_obs, non_spatial_obs, mask) = self.env.reset()
        self.mask = mask

        if spatial_obs is None or non_spatial_obs is None or mask is None:
            # to do: find a way to flag this better
            print("ERROR: obs is None when it shouldn't be making empty obs")
            obs_dict = self._make_empty_obs()
        else:
            mask = mask.astype("float32")
            if self.combine_obs:
                flat_spatial_obs = spatial_obs.flatten()
                flat_s_ns = np.concatenate((flat_spatial_obs, non_spatial_obs), axis=0).astype("float32")
                obs_dict = {
                    'flat_s_ns': flat_s_ns,
                    'action_mask': mask,
                }
            else:
                obs_dict = {
                    'spatial': spatial_obs.astype("float32"),
                    'non_spatial': non_spatial_obs.astype("float32"),
                    'action_mask': mask,
                }

        agent_obs_dict = {self._get_current_player(): obs_dict}

        if self.debug_mode:
            self.check_input_array(agent_obs_dict, "on_reset")

        return agent_obs_dict

    def step(self, action):
        # multi agent env requires action to be in dict with the key being the current player
        # print("---stepping---")
        # print("test mask is ", np.where(self.mask == True))
        # print("action is ", action)
        current_player = self._get_current_player()
        # print("current player pre action is ", current_player, current_player in action)
        assert current_player in action
        action_from_dict = action[current_player]

        aa = np.where(self.mask > 0.0)[0]
        if action_from_dict not in aa:
            # unclear why this did not happen in single version but is in multi version
            # might be an issue with flipping but I will have to look into it
            action_from_dict = np.random.choice(aa, 1)[0]
            # if current_player == 0:
            #     print("ERROR: choosing random action for player {} from mask {} ".format(current_player, action_from_dict))
            #     print("action is ", action)
            #     print("mask is ", np.where(self.mask == True))
            #   # assert 1 == 0

        (spatial_obs, non_spatial_obs, mask), reward, done, info = self.env.step(action_from_dict)
        # update current player now that action is taken
        current_player = self._get_current_player()
        self.mask = mask

        if done:
            # when done, all these are none
            obs_dict = self._make_empty_obs()
        elif spatial_obs is None or non_spatial_obs is None or mask is None:
            # to do: find a way to flag this better
            print("ERROR: obs is None when it shouldn't be making empty obs")
            obs_dict = self._make_empty_obs()
        else:
            mask = mask.astype("float32")
            if self.combine_obs:
                flat_spatial_obs = spatial_obs.flatten()
                flat_s_ns = np.concatenate((flat_spatial_obs, non_spatial_obs), axis=0).astype("float32")
                obs_dict = {
                    'flat_s_ns': flat_s_ns,
                    'action_mask': mask,
                }
            else:
                obs_dict = {
                    'spatial': spatial_obs.astype("float32"),
                    'non_spatial': non_spatial_obs.astype("float32"),
                    'action_mask': mask,
                }

        agent_obs_dict = {current_player: obs_dict}

        # flip the reward for the opponent
        opponent = (current_player + 1) % 2
        reward_dict = {current_player: reward, opponent: -reward}

        # make the done_dict
        done_dict = {current_player: done, opponent: done, '__all__': done}

        if self.debug_mode:
            # agent_obs_dict[current_player]["non_spatial"][0] = 1.19
            self.check_input_array(agent_obs_dict, done)

        return agent_obs_dict, reward_dict, done_dict, info

    def _make_empty_obs(self):
        # when done, need to return an empty obs
        # also sometimes env is bugging out and not producing an obs, not sure why yet
        mask = np.zeros((self.action_space.n,),
                        dtype="float32")  ##didn't test this as all zeros but think it makes sense

        if self.combine_obs:
            flat_s_ns = np.zeros(self.observation_space['flat_s_ns'].shape, dtype="float32")
            obs_dict = {
                'flat_s_ns': flat_s_ns,
                'action_mask': mask,
            }
        else:
            spatial_obs = np.zeros(self.observation_space['spatial'].shape, dtype="float32")
            non_spatial_obs = np.zeros(self.observation_space['non_spatial'].shape, dtype="float32")
            obs_dict = {
                'spatial': spatial_obs.astype("float32"),
                'non_spatial': non_spatial_obs.astype("float32"),
                'action_mask': mask,
            }
        return obs_dict

    def _get_current_player(self):
        # return based on matching index with active turn
        # TEST THIS
        # print("testing _get_current_player ")
        # print("active team is  ", self.env.game.active_team)
        # print("is home team  ", self.env.game.is_home_team(self.env.game.active_team) )
        if self.env.game.is_home_team(self.env.game.active_team):
            return 0  # home team is always 0
        return 1

    # sanity checking due to possible overflow/NaN, non duplicated bug in early stage
    # also checking if any values above 1, which shouldn't be the case
    def check_input_array(self, obs_dict, done):
        for agent_id in obs_dict.keys():
            temp_dict = obs_dict[agent_id]
            for k, v in temp_dict.items():
                if np.isnan(v).any():
                    print("INPUT CHECK NaN found in key {}, value {}, done {}".format(k, v, done))
                    print(self.env.game)
                    #assert 1 == 0
                    output_dict = {
                        "type": "nan_found",
                        "key": k,
                        "value": v,
                        "done": done,
                        #"game": self.env.game,
                    }
                    self.save_debug_to_file(output_dict)
                if np.isinf(v).any():
                    print("INPUT CHECK Inf found in key {}, value {}, done {}".format(k, v, done))
                    print(self.env.game)
                    #assert 1 == 0
                    output_dict = {
                        "type": "inf_found",
                        "key": k,
                        "value": v,
                        "done": done,
                        #"game": self.env.game,
                    }
                    self.save_debug_to_file(output_dict)
                if len(np.where(v > 1)[0]) > 0:
                    print("INPUT CHECK Value above 1 found in key {}, value {}, done {}".format(k, v, done))
                    print(self.env.game)
                    output_dict = {
                        "type": "value_above_1",
                        "key": k,
                        "value": v,
                        "done": done,
                        #"game": self.env.game,
                    }
                    self.save_debug_to_file(output_dict)

    def save_debug_to_file(self, output_dict):
        with open(self.debug_save_path, 'a+') as handle:
            pickle.dump(output_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # assert 1 == 0

    def render(self, mode=None) -> None:
        if mode == "human":
            pass