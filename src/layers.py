"""Classes for SimGNN modules."""

import paddorch
import paddorch.nn as nn
import paddle

class AttentionModule(nn.Module):
    """
    SimGNN Attention Module to make a pass on graph.
    """
    def __init__(self, args):
        """
        :param args: Arguments object.
        """
        super(AttentionModule, self).__init__()
        self.args = args
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):
        """
        Defining weights.
        """
        self.weight_matrix = paddorch.nn.Parameter(paddorch.Tensor(self.args.filters_3,
                                                             self.args.filters_3))

    def init_parameters(self):
        """
        Initializing weights.
        """
        paddorch.nn.init.xavier_uniform_(self.weight_matrix)

    def forward(self, embedding):
        """
        Making a forward propagation pass to create a graph level representation.
        :param embedding: Result of the GCN.
        :return representation: A graph level representation vector.
        """
        global_context = paddorch.mean(paddorch.matmul(embedding, self.weight_matrix), dim=0)
        transformed_global = paddorch.tanh(global_context)
        sigmoid_scores = paddorch.sigmoid(paddorch.mm(embedding, transformed_global.view(-1, 1)))
        representation = paddorch.mm(paddle.t(embedding), sigmoid_scores)
        return representation

class TenorNetworkModule(nn.Module):
    """
    SimGNN Tensor Network module to calculate similarity vector.
    """
    def __init__(self, args):
        """
        :param args: Arguments object.
        """
        super(TenorNetworkModule, self).__init__()
        self.args = args
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):
        """
        Defining weights.
        """
        self.weight_matrix = paddorch.nn.Parameter(paddorch.Tensor(self.args.filters_3,
                                                             self.args.filters_3,
                                                             self.args.tensor_neurons))

        self.weight_matrix_block = paddorch.nn.Parameter(paddorch.Tensor(self.args.tensor_neurons,
                                                                   2*self.args.filters_3))
        self.bias = paddorch.nn.Parameter(paddorch.Tensor(self.args.tensor_neurons, 1))

    def init_parameters(self):
        """
        Initializing weights.
        """
        paddorch.nn.init.xavier_uniform_(self.weight_matrix)
        paddorch.nn.init.xavier_uniform_(self.weight_matrix_block)
        paddorch.nn.init.xavier_uniform_(self.bias)

    def forward(self, embedding_1, embedding_2):
        """
        Making a forward propagation pass to create a similarity vector.
        :param embedding_1: Result of the 1st embedding after attention.
        :param embedding_2: Result of the 2nd embedding after attention.
        :return scores: A similarity score vector.
        """
        scoring = paddorch.mm(paddle.t(embedding_1), self.weight_matrix.view(self.args.filters_3, -1))
        scoring = scoring.view(self.args.filters_3, self.args.tensor_neurons)
        scoring = paddorch.mm(paddle.t(scoring), embedding_2)
        combined_representation = paddorch.cat((embedding_1, embedding_2))
        block_scoring = paddorch.mm(self.weight_matrix_block, combined_representation)
        scores = paddorch.nn.functional.relu(scoring + block_scoring + self.bias)
        return scores
