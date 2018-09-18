from torch.autograd import Variable
import torch.onnx
from dynav.policy.sarl import ValueNetwork
import time

start = time.time()
data = Variable(torch.randn(1, 5, 13))
model = ValueNetwork(input_dim=13,
                     self_state_dim=7,
                     mlp1_dims=[150, 100],
                     mlp2_dims=[100, 50],
                     mlp3_dims=[100, 100, 1],
                     attention_dims=[150, 100, 100, 1],
                     with_global_state=True,
                     global_om=False,
                     cell_size=1,
                     cell_num=4)
value = model(data)
end = time.time()
print('Forwarding time: {}'.format(end - start))

# Providing input and output names sets the display names for values
# within the model's graph. Setting these does not change the semantics
# of the graph; it is only for readability.
#
# The inputs to the network consist of the flat list of inputs (i.e.
# the values you would pass to the forward() method) followed by the
# flat list of parameters. You can partially specify names, i.e. provide
# a list here shorter than the number of inputs to the model, and we will
# only set that subset of names, starting from the beginning.
# input_names = [ "actual_input_1" ] + [ "learned_%d" % i for i in range(16) ]
# output_names = [ "output1" ]
#
# torch.onnx.export(model, dummy_input, "alexnet.onnx", verbose=True, input_names=input_names, output_names=output_names)

torch.onnx.export(model, data, "data/sarl.onnx", verbose=True, export_params=True)

# convert onnx to caffe2
# convert-onnx-to-caffe2 data/sarl.onnx --output data/sarl_network.pb --init-net-output data/sarl_weights.pb