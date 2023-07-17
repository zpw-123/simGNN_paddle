import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

def degree_norm(graph, mode="indegree"):
    """Calculate the degree normalization of a graph

    Args:
        graph: the graph object from (:code:`Graph`)

        mode: which degree to be normalized ("indegree" or "outdegree")

    return:
        A tensor with shape (num_nodes, 1).

    """

    assert mode in [
        'indegree', 'outdegree'
    ], "The degree_norm mode should be in ['indegree', 'outdegree']. But recieve mode=%s" % mode

    if mode == "indegree":
        degree = graph.indegree()
    elif mode == "outdegree":
        degree = graph.outdegree()

    norm = paddle.cast(degree, dtype=paddle.get_default_dtype())
    norm = paddle.clip(norm, min=1.0)
    norm = paddle.pow(norm, -0.5)
    norm = paddle.reshape(norm, [-1, 1])
    return norm 

class GCNConv(nn.Layer):
    """Implementation of graph convolutional neural networks (GCN)

    This is an implementation of the paper SEMI-SUPERVISED CLASSIFICATION
    WITH GRAPH CONVOLUTIONAL NETWORKS (https://arxiv.org/pdf/1609.02907.pdf).

    Args:

        input_size: The size of the inputs.

        output_size: The size of outputs

        activation: The activation for the output.

        norm: If :code:`norm` is True, then the feature will be normalized.

    """
    def __init__(self, input_size, output_size, activation=None, norm=True):
        super(GCNConv, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.linear = nn.Linear(input_size, output_size, bias_attr=False)
        self.bias = self.create_parameter(shape=[output_size], is_bias=True)
        self.norm = norm
        if isinstance(activation, str):
            activation = getattr(F, activation)
        self.activation = activation
    
    def forward(self, graph, feature, norm=None):
        """

        Args:

            graph: `pgl.Graph` instance.

            feature: A tensor with shape (num_nodes, input_size)

            norm: (default None). If :code:`norm` is not None, then the feature will be normalized by given norm. If :code:`norm` is None and :code:`self.norm` is `true`, then we use `lapacian degree norm`.

        Return:

            A tensor with shape (num_nodes, output_size)

        """

        if self.norm and norm is None:
            norm = degree_norm(graph)

        if self.input_size > self.output_size:
            feature = self.linear(feature)

        if norm is not None:
            feature = feature * norm

        output = graph.send_recv(feature, "sum")

        if self.input_size <= self.output_size:
            output = self.linear(output)

        if norm is not None:
            output = output * norm
        output = output + self.bias
        if self.activation is not None:
            output = self.activation(output)
        return output

