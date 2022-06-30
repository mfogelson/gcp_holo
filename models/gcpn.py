import torch
import torchvision   
import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool, SAGEConv
from torch.distributions import Categorical, Bernoulli
from torch_geometric.data import Data, DataLoader, DenseDataLoader

from math import ceil
import pdb
import pickle
from concurrent.futures import ProcessPoolExecutor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


'''
class GNN:

define a GNN module

parameters:
    in_channels: number of feature channels for each input node
    hidden_channels: number of feature channels for each hidden node
    batch_normalization: if add a batch normalization after each conv layer
'''
class GNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, max_nodes, num_features, hidden_channels=64, out_channels=64, normalize=False, batch_normalization=False, lin=True, add_loop=False):
        super(GNN, self).__init__(observation_space, features_dim=1)

        self.max_nodes = max_nodes #observation_space['mask'].shape[0]
        self.num_features = num_features
        in_channels = num_features #observation_space['x'].shape[0]// self.max_nodes # 10 = max nodes

        self.add_loop = add_loop
        self.batch_normalization = batch_normalization

        self.conv1 = DenseSAGEConv(in_channels, hidden_channels, normalize)
        self.conv2 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
        self.conv3 = DenseSAGEConv(hidden_channels, out_channels, normalize)
        
        if self.batch_normalization:
            self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
            self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
            self.bn3 = torch.nn.BatchNorm1d(out_channels)

        if lin is True:
            self.lin = torch.nn.Linear(2 * hidden_channels + out_channels,
                                       out_channels)
        else:
            self.lin = None

        self.relu = nn.ReLU()

        self._features_dim = out_channels

    def bn(self, i, x):
        batch_size, num_nodes, in_channels = x.size()

        x = x.view(-1, in_channels)
        x = getattr(self, 'bn{}'.format(i))(x)
        x = x.view(batch_size, num_nodes, in_channels)
        return x

    def forward(self, observations):
        shape_x = self.max_nodes*self.num_features
        shape_adj = self.max_nodes**2
        shape_mask = self.max_nodes
        
        x = observations[:, :shape_x] #['x']
        adj = observations[:, shape_x+1:shape_x+1+shape_adj] #['adj']
        mask = observations[:, shape_x+shape_adj+2:shape_x+2+shape_adj+shape_mask] #['mask']
        
        if len(x.size()) > 1:
            batch_size, _ = x.size()
        else:
            batch_size = 1

        x = x.reshape(batch_size, self.max_nodes, -1) # B, nodes, features
        adj = adj.reshape(batch_size, self.max_nodes, self.max_nodes)

        # pdb.set_trace()

        if self.batch_normalization:
            x1 = self.bn(1, self.relu(self.conv1(x, adj, mask))) #, #self.add_loop)))
            x2 = self.bn(2, self.relu(self.conv2(x1, adj, mask))) #, #self.add_loop)))
            x3 = self.bn(3, self.relu(self.conv3(x2, adj, mask))) #, #self.add_loop)))
        else:
            x1 = self.relu(self.conv1(x, adj, mask))
            x2 = self.relu(self.conv2(x1, adj, mask))
            x3 = self.relu(self.conv3(x2, adj, mask))

        # pdb.set_trace()
        # x = torch.cat([x1, x2], dim=-1)
        x = torch.cat([x1, x2, x3], dim=-1)
        # x = x1
        if self.lin is not None:
            x = self.relu(self.lin(x))

        return x.sum(1)



# class CustomActorCriticPolicy(ActorCriticPolicy):
#     def __init__(
#         self,
#         observation_space: gym.spaces.Space,
#         action_space: gym.spaces.Space,
#         lr_schedule: Callable[[float], float],
#         net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
#         activation_fn: Type[nn.Module] = nn.Tanh,
#         *args,
#         **kwargs,
#     ):

#         super(CustomActorCriticPolicy, self).__init__(
#             observation_space,
#             action_space,
#             lr_schedule,
#             net_arch,
#             activation_fn,
#             # Pass remaining arguments to base class
#             *args,
#             **kwargs,
#         )
#         # Disable orthogonal initialization
#         self.ortho_init = False

#     def _build_mlp_extractor(self) -> None:
#         self.mlp_extractor = CustomNetwork(self.features_dim)