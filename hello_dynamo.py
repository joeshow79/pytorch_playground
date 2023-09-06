from typing import List
import torch
import tabulate
from torch._dynamo import optimize
import torch._dynamo.config
import logging

torch._dynamo.config.log_level = logging.INFO
torch._dynamo.config.output_code = True

def my_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
	print("my_compiler() called with FX graph:")
	gm.graph.print_tabular()
	return gm.forward  # return a python callable

@torch.compile(backend=my_compiler)
def toy_example(a, b):
	x = a / (torch.abs(a) + 1)
	if b.sum() < 0:
		b = b * -1
	return x * b

for _ in range(5):
	#toy_example(torch.randn(10), torch.randn(10))
        explanation, out_guards, graphs, ops_per_graph, ab, cd = torch._dynamo.explain(toy_example,torch.randn(10), torch.randn(10))
        #explanation = torch._dynamo.explain(toy_example,torch.randn(10), torch.randn(10))
        print((explanation))
        print((out_guards))
        print((graphs))
        print((ops_per_graph))
        print((ab))
        print((cd))

print("------------------------------------------------")
torch._dynamo.utils.compile_times()
