from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.torch_utils import FLOAT_MIN, FLOAT_MAX
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym


# feedforward NN with flat input
class BasicFCNN(TorchModelV2, nn.Module):

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs)
        nn.Module.__init__(self)

        orig_space = getattr(obs_space, "original_space", obs_space)
        assert (
                isinstance(orig_space, gym.spaces.Dict)
                and "action_mask" in orig_space.spaces
                and "flat_s_ns" in orig_space.spaces
            #            and "force assert fail" in orig_space.spaces
            #             and "spatial" in orig_space.spaces
            #             and "non_spatial" in orig_space.spaces
            #             and "observations" in orig_space.spaces
        )

        # TorchFC expects an obs space
        # for now just flat spatial and non-spatial which adds up to 1435, though I should make this dynamic
        # internal_obs_space = Box(0.0, 1.0, (1435,), "float32") #1435 #Box(0.0, 1.0, (1,), "float32"),
        # flattening in the env though not sure if this is the best way
        # https://github.com/ray-project/ray/blob/master/rllib/models/torch/fcnet.py#L16
        self.internal_model = TorchFC(
            # orig_space["observations"],
            # orig_space["spatial"],
            orig_space["flat_s_ns"],
            action_space,
            num_outputs,
            model_config,
            name + "_internal",
        )

    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        action_mask = input_dict["obs"]["action_mask"]
        # print("in forward, action_mask action mask shape ", action_mask.shape )
        # if action_mask.shape[0] == 1:
        #     print("in forward, action_mask where > 0 ", torch.where(action_mask[0] > 0))

        # Compute the unmasked logits.
        # logits, _ = self.internal_model({"obs": internal_obs})
        # logits, _ = self.internal_model({"obs": input_dict["obs"]["observations"]})
        # logits, _ = self.internal_model({"obs": input_dict["obs"]["spatial"]})
        logits, _ = self.internal_model({"obs": input_dict["obs"]["flat_s_ns"]})

        # Convert action_mask into a [0.0 || -inf]-type mask.
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN, max=FLOAT_MAX)
        # inf_mask = action_mask.log()
        # print("in forward, inf mask shape ", action_mask.shape)
        # if inf_mask.shape[0] == 1:
        #     print("in forward, inf_mask where > 0 ", torch.where(inf_mask[0] > -1000))

        # print("inf mask ", inf_mask)
        masked_logits = logits + inf_mask
        # print("masked logits ", masked_logits)
        # Return masked logits.
        return masked_logits, state

    def value_function(self):
        return self.internal_model.value_function()


# CNN from A2C examples in botbowl folder. modified slightly to work with ray
class A2CCNN(TorchModelV2, nn.Module):

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs)
        nn.Module.__init__(self)

        orig_space = getattr(obs_space, "original_space", obs_space)
        assert (
                isinstance(orig_space, gym.spaces.Dict)
                and "action_mask" in orig_space.spaces
                and "spatial" in orig_space.spaces
                and "non_spatial" in orig_space.spaces
        )
        # example architecture is https://github.com/njustesen/botbowl/blob/main/examples/a2c/a2c_example.py#L58
        hidden_nodes = 128
        kernels = [32, 64]

        # Spatial input stream
        self.conv1 = nn.Conv2d(orig_space['spatial'].shape[0], out_channels=kernels[0],
                               kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=kernels[0], out_channels=kernels[1],
                               kernel_size=3, stride=1, padding=1)

        # Non-spatial input stream
        self.linear0 = nn.Linear(orig_space['non_spatial'].shape[0], hidden_nodes)

        # Linear layers
        stream_size = kernels[1] * orig_space['spatial'].shape[1] * orig_space['spatial'].shape[2]
        stream_size += hidden_nodes
        self.linear1 = nn.Linear(stream_size, hidden_nodes)

        # The outputs
        self.vf_layers = nn.Linear(hidden_nodes, 1)
        self.policy_output = nn.Linear(hidden_nodes, action_space.n)

        self._output = None  # make output in forward and then put vf_layers over this in value function
        self.reset_parameters()

    def reset_parameters(self):
        relu_gain = nn.init.calculate_gain('relu')
        self.conv1.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)
        self.linear0.weight.data.mul_(relu_gain)
        self.linear1.weight.data.mul_(relu_gain)
        self.policy_output.weight.data.mul_(relu_gain)
        self.vf_layers.weight.data.mul_(relu_gain)

    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        action_mask = input_dict["obs"]["action_mask"]
        spatial_input = input_dict["obs"]["spatial"]
        non_spatial_input = input_dict["obs"]["non_spatial"]

        x1 = self.conv1(spatial_input)
        x1 = F.relu(x1)
        x1 = self.conv2(x1)
        x1 = F.relu(x1)

        # Concatenate the input streams
        flatten_x1 = x1.flatten(start_dim=1)

        x2 = self.linear0(non_spatial_input)
        x2 = F.relu(x2)

        flatten_x2 = x2.flatten(start_dim=1)
        concatenated = torch.cat((flatten_x1, flatten_x2), dim=1)

        # Fully-connected layers
        x3 = self.linear1(concatenated)
        self._output = F.relu(x3)

        # Output streams
        logits = self.policy_output(x3)

        # apply masks to actions: ie make non-feasible actions as small as possible
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN, max=FLOAT_MAX)
        masked_logits = logits + inf_mask

        return masked_logits, state

    def value_function(self):
        assert self._output is not None, "must call forward first!"
        return torch.reshape(self.vf_layers(self._output), [-1])


# for init layers of IMPALANET
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# for impala net
# taken from https://github.com/AIcrowd/neurips2020-procgen-starter-kit/blob/142d09586d2272a17f44481a115c4bd817cf6a94/models/impala_cnn_torch.py
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)

    def forward(self, x):
        inputs = x
        x = nn.functional.relu(x)
        x = self.conv0(x)
        x = nn.functional.relu(x)
        x = self.conv1(x)
        return x + inputs


# for impalanet
class ConvSequence(nn.Module):
    def __init__(self, input_shape, out_channels):
        super().__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels
        self.conv = nn.Conv2d(in_channels=self._input_shape[0], out_channels=self._out_channels, kernel_size=3, padding=1)
        self.res_block0 = ResidualBlock(self._out_channels)
        self.res_block1 = ResidualBlock(self._out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = nn.functional.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.res_block0(x)
        x = self.res_block1(x)
        assert x.shape[1:] == self.get_output_shape()
        return x

    def get_output_shape(self):
        _c, h, w = self._input_shape
        return (self._out_channels, (h + 1) // 2, (w + 1) // 2)


class IMPALANet(TorchModelV2, nn.Module):
    # https://github.com/AIcrowd/neurips2020-procgen-starter-kit/blob/master/models/impala_cnn_torch.py
    # https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_procgen.py
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs)
        nn.Module.__init__(self)

        orig_space = getattr(obs_space, "original_space", obs_space)
        assert (
                isinstance(orig_space, gym.spaces.Dict)
                and "action_mask" in orig_space.spaces
                and "spatial" in orig_space.spaces
                and "non_spatial" in orig_space.spaces
        )
        hidden_nodes = 256

        # (44, 17, 28) is spatial shape. 44 channels 17 height, 28 width
        c, h, w = orig_space["spatial"].shape
        shape = (c, h, w)
        conv_seqs = []
        for out_channels in [16, 32, 32]:
            conv_seq = ConvSequence(shape, out_channels)
            shape = conv_seq.get_output_shape()
            conv_seqs.append(conv_seq)
        # doing my own flatten and concat because of non-spatial features
        # conv_seqs += [
        #     nn.Flatten(),
        #     nn.ReLU(),
        #     nn.Linear(in_features=shape[0] * shape[1] * shape[2], out_features=256),
        #     nn.ReLU(),
        # ]
        self.network = nn.Sequential(*conv_seqs)
        self.actor = layer_init(nn.Linear(hidden_nodes, action_space.n), std=0.01) # traditional impalanet is 256
        self.critic = layer_init(nn.Linear(hidden_nodes, 1), std=1) # traditional impalanet 256 hidden nodes

        # Non-spatial input stream
        self.linear0 = layer_init(nn.Linear(orig_space['non_spatial'].shape[0], hidden_nodes))
        # layer for concatenate after the spatial and non spatial joins
        stream_size =  shape[0] * shape[1] * shape[2] #num outputs of flat spatial
        stream_size += hidden_nodes # num outputs of non-spatial
        self.linear1 = layer_init(nn.Linear(stream_size, hidden_nodes))
        # init output in forward and then put critic over this in value_function
        self._output = None

    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        action_mask = input_dict["obs"]["action_mask"]
        spatial_input = input_dict["obs"]["spatial"]
        non_spatial_input = input_dict["obs"]["non_spatial"]

        # spatial
        spatial_output = self.network(spatial_input)
        flatten_spatial = spatial_output.flatten(start_dim=1) # dim 0 is the batch sized, flatten after that
        flatten_spatial = F.relu(flatten_spatial) # IMPALANet procgen/cleanRL example relu after flatten
        #non spatial
        x2 = self.linear0(non_spatial_input)
        x2 = F.relu(x2)
        flatten_x2 = x2.flatten(start_dim=1)
        # concat spatial and non-spatial, one more hidden layer before outputs (critic output done in vf function)
        concatenated = torch.cat((flatten_spatial, flatten_x2), dim=1)
        x3 = self.linear1(concatenated)
        self._output = F.relu(x3)

        # Output streams
        logits = self.actor(x3)

        # apply masks to actions: ie make non-feasible actions as small as possible
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN, max=FLOAT_MAX)
        masked_logits = logits + inf_mask

        return masked_logits, state

    def value_function(self):
        assert self._output is not None, "must call forward first!"
        #return torch.reshape(self.vf_layers(self._output), [-1])
        return torch.reshape(self.critic(self._output), [-1])

# how clean RL implements impalanet
# class Agent(nn.Module):
#     def __init__(self, envs):
#         super(Agent, self).__init__()
#         h, w, c = envs.single_observation_space.shape
#         shape = (c, h, w)
#         conv_seqs = []
#         for out_channels in [16, 32, 32]:
#             conv_seq = ConvSequence(shape, out_channels)
#             shape = conv_seq.get_output_shape()
#             conv_seqs.append(conv_seq)
#         conv_seqs += [
#             nn.Flatten(),
#             nn.ReLU(),
#             nn.Linear(in_features=shape[0] * shape[1] * shape[2], out_features=256),
#             nn.ReLU(),
#         ]
#         self.network = nn.Sequential(*conv_seqs)
#         self.actor = layer_init(nn.Linear(256, envs.single_action_space.n), std=0.01)
#         self.critic = layer_init(nn.Linear(256, 1), std=1)
#
#     def get_value(self, x):
#         return self.critic(self.network(x.permute((0, 3, 1, 2)) / 255.0))  # "bhwc" -> "bchw"
#
#     def get_action_and_value(self, x, action=None):
#         hidden = self.network(x.permute((0, 3, 1, 2)) / 255.0)  # "bhwc" -> "bchw"
#         logits = self.actor(hidden)
#         probs = Categorical(logits=logits)
#         if action is None:
#             action = probs.sample()
#         return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)


class IMPALANetFixed(TorchModelV2, nn.Module):
    # IMPALANet above except I fix a bug I made where I used ReLU on the critic
    # https://github.com/AIcrowd/neurips2020-procgen-starter-kit/blob/master/models/impala_cnn_torch.py
    # https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_procgen.py
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs)
        nn.Module.__init__(self)

        orig_space = getattr(obs_space, "original_space", obs_space)
        assert (
                isinstance(orig_space, gym.spaces.Dict)
                and "action_mask" in orig_space.spaces
                and "spatial" in orig_space.spaces
                and "non_spatial" in orig_space.spaces
        )
        hidden_nodes = 256

        # (44, 17, 28) is spatial shape. 44 channels 17 height, 28 width
        c, h, w = orig_space["spatial"].shape
        shape = (c, h, w)
        conv_seqs = []
        for out_channels in [16, 32, 32]:
            conv_seq = ConvSequence(shape, out_channels)
            shape = conv_seq.get_output_shape()
            conv_seqs.append(conv_seq)
        # doing my own flatten and concat because of non-spatial features
        # conv_seqs += [
        #     nn.Flatten(),
        #     nn.ReLU(),
        #     nn.Linear(in_features=shape[0] * shape[1] * shape[2], out_features=256),
        #     nn.ReLU(),
        # ]
        self.network = nn.Sequential(*conv_seqs)
        self.actor = layer_init(nn.Linear(hidden_nodes, action_space.n), std=0.01) # traditional impalanet is 256
        self.critic = layer_init(nn.Linear(hidden_nodes, 1), std=1) # traditional impalanet 256 hidden nodes

        # Non-spatial input stream
        self.linear0 = layer_init(nn.Linear(orig_space['non_spatial'].shape[0], hidden_nodes))
        # layer for concatenate after the spatial and non spatial joins
        stream_size =  shape[0] * shape[1] * shape[2] #num outputs of flat spatial
        stream_size += hidden_nodes # num outputs of non-spatial
        self.linear1 = layer_init(nn.Linear(stream_size, hidden_nodes))
        # init output in forward and then put critic over this in value_function
        self._output = None

    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        action_mask = input_dict["obs"]["action_mask"]
        spatial_input = input_dict["obs"]["spatial"]
        non_spatial_input = input_dict["obs"]["non_spatial"]

        # spatial
        spatial_output = self.network(spatial_input)
        flatten_spatial = spatial_output.flatten(start_dim=1) # dim 0 is the batch sized, flatten after that
        flatten_spatial = F.relu(flatten_spatial) # IMPALANet procgen/cleanRL example relu after flatten
        #non spatial
        x2 = self.linear0(non_spatial_input)
        x2 = F.relu(x2)
        flatten_x2 = x2.flatten(start_dim=1)
        # concat spatial and non-spatial, one more hidden layer before outputs (critic output done in vf function)
        concatenated = torch.cat((flatten_spatial, flatten_x2), dim=1)
        x3 = self.linear1(concatenated)
        self._output = x3

        # Output streams
        logits = self.actor(x3)

        # apply masks to actions: ie make non-feasible actions as small as possible
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN, max=FLOAT_MAX)
        masked_logits = logits + inf_mask

        return masked_logits, state

    def value_function(self):
        assert self._output is not None, "must call forward first!"
        #return torch.reshape(self.vf_layers(self._output), [-1])
        return torch.reshape(self.critic(self._output), [-1])
