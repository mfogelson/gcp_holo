import torch
import torch.nn as nn
from torch_geometric.nn import DenseSAGEConv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class GNN(BaseFeaturesExtractor):
    """
    Graph Convolution network: adopted from Zhao et. al "Robogrammar"
        Args:
                observation_space (gym.observation): The observation space of the gym environment
                max_nodes (int): maximum number of nodes for linkage graph
                num_features (int): number of points in the trajectory to describe the node features
                hidden_channels (int, optional): hidden channels for the Dense SAGE convolutions. Defaults to 64.
                out_channels (int, optional): number of output features. Defaults to 64.
                normalize (bool, optional): normalization used in Dense SAGE. Defaults to False.
                batch_normalization (bool, optional): Batch Normalization used. Defaults to False.
                lin (bool, optional): Add linear layer to the end. Defaults to True.
                add_loop (bool, optional): Add self loops. Defaults to False.
    """
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
        ## Get shapes for observation
        shape_x = self.max_nodes*self.num_features
        shape_adj = self.max_nodes**2
        shape_mask = self.max_nodes
        
        ## extract information from observation input
        x = observations[:, :shape_x] #['x']
        adj = observations[:, shape_x:shape_x+shape_adj] #['adj']
        mask = observations[:, shape_x+shape_adj:shape_x+shape_adj+shape_mask] #['mask']
        
        if len(x.size()) > 1:
            batch_size, _ = x.size()
        else:
            batch_size = 1

        ## Reshape for model
        x = x.view(batch_size, self.max_nodes, -1) # B, nodes, features
        adj = adj.view(batch_size, self.max_nodes, self.max_nodes)


        ## Forward pass
        if self.batch_normalization:
            x1 = self.bn(1, self.relu(self.conv1(x, adj, mask))) #, #self.add_loop)))
            x2 = self.bn(2, self.relu(self.conv2(x1, adj, mask))) #, #self.add_loop)))
            x3 = self.bn(3, self.relu(self.conv3(x2, adj, mask))) #, #self.add_loop)))
        else:
            x1 = self.relu(self.conv1(x, adj, mask))
            x2 = self.relu(self.conv2(x1, adj, mask))
            x3 = self.relu(self.conv3(x2, adj, mask))

        ## Concatenate latent representations
        x = torch.cat([x1, x2, x3], dim=-1)

        ## Extra linear layer
        if self.lin is not None:
            x = self.relu(self.lin(x))

        ## Aggrigate node features to output Graph latent representation
        return x.sum(1)
