import torch
from torch_geometric.data import Data

edgeIndex = torch.tensor([[0,1,1,2],[1,0,2,1]], dtype=torch.long)

featureVector = torch.tensor([[-1,-1], [0,0], [1,1]], dtype=torch.float)

data = Data(x=featureVector, edge_index=edgeIndex)

print(data)


import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(2,2)
        self.conv2 = GCNConv(2,2)

    def forward(self, data):
        x,edge_index = data.x, data.edge_index

        x = self.conv1(x,edge_index)
        x = self.conv2(x,edge_index)
        #return F.log_softmax(x, dim=1)
        return x

model = Net()
out = model(data)

print(out)
