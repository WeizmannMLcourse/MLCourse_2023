import torch
import torch.nn as nn
import dgl


### Fully-connected network ###
class FCN(nn.Module):

    def __init__(self, N_input, N_hidden, N_output):
        super().__init__()

        assert len(N_hidden) > 0, "Pass list of hidden layer sizes for N_hidden"

        self.node_network = nn.Sequential(
            nn.Linear(N_input, N_hidden[0]),
            nn.ReLU(),
        )

        if len(N_hidden) > 1:
            for i in range(1, len(N_hidden)):
                self.node_network.append(nn.Linear(N_hidden[i - 1], N_hidden[i])),
                self.node_network.append(nn.ReLU())


        self.node_network.append(nn.Linear(N_hidden[-1], N_output))

    def forward(self, x):

        return self.node_network(x)
    


### Node update network ###
class NodeNetwork(nn.Module):

    def __init__(self, N_hidden):
        super().__init__()

        self.net = FCN(N_hidden*2, [N_hidden*3, N_hidden*2, N_hidden*2], N_hidden)

    def forward(self, nodes):

        messages    = nodes.mailbox['m']
        aggregate   = torch.mean(messages, dim=1)
        rep_cat_aggregate = torch.cat([nodes.data['h'], aggregate], dim=-1)

        updated_rep = self.net(rep_cat_aggregate)
        return {'h': updated_rep}



### Message-passing neural network ###
class MoleculeMPNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.N_input = 4
        self.N_hidden = 64
        self.N_output = 1
        self.N_message_passing_blocks = 4

        self.node_encoding  =  FCN(self.N_input, 
                                   [self.N_hidden, self.N_hidden], 
                                   self.N_hidden)

        self.node_networks  = nn.ModuleList(
            [ NodeNetwork(self.N_hidden) for _ in range(self.N_message_passing_blocks)]
        )

        self.pred_networks = FCN(self.N_hidden, 
                                 [self.N_hidden*2, self.N_hidden*3, self.N_hidden*2, self.N_hidden], 
                                 self.N_output)

    def forward(self, g):        

        ### prune attributes ###
        if g.ndata['attr'].shape[1] > 1:
            g.ndata['attr'] = g.ndata['attr'][:,5].unsqueeze(-1)

        g.ndata['h'] = self.node_encoding(torch.cat([g.ndata['pos'], g.ndata['attr']],dim=-1))
        
        for i in range(self.N_message_passing_blocks):

            ### Update node representations ###
            g.update_all(dgl.function.copy_u('h','m'),self.node_networks[i])
            ### See https://docs.dgl.ai/en/0.8.x/api/python/dgl.function.html for built-in dgl message functions ###
            ### Note: 'u' refers to source node

        ### construct a global representation by summing over the node representations ###
        global_rep = dgl.sum_nodes(g,'h')
        pred = self.pred_networks(global_rep)

        return pred