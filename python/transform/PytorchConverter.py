# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

from .MLIRImporter import MLIRImporter
from .BaseConverter import BaseConverter
from numbers import Number
from typing import Union, Iterable, List
from mlir.ir import *
import numpy as np
import random
import copy
import torch

def _get_constant(node):
  """Retrieve a constant associated with this prim::Constant node"""
  attribute_names = node.attributeNames()
  num_attributes = len(attribute_names)
  name = node.output().debugName()
  is_tensor = False
  type = node.output().type().kind()

  if type == "NoneType":
    return name, None, False
  elif num_attributes == 1:
    attr_name = attribute_names[0]
    if type == "IntType":
      value = node.i(attr_name)
    elif type == "BoolType":
      value = bool(node.i(attr_name))
    elif type in ["FloatType", "LongType"]:
      value = node.f(attr_name)
    elif type in ["DeviceObjType", "StringType"]:
      value = node.s(attr_name)
    elif type in ["TensorType", "CompleteTensorType"]:
      is_tensor = True
      tensor = node.t(attr_name)
      if tensor.is_cuda:
        tensor = tensor.cpu()
      value = tensor.numpy()
    else:
      raise NotImplementedError("Unsupported type: %s" % type)
  else:
    assert num_attributes == 0
    return None
  return name, value, is_tensor

def _data_expand(data, length):
  if isinstance(data, int):
    return [data for i in range(length)]
  else:
    return data

def _compute_pad(stride, dilation, input_size, filter, padding):
  stride = np.array(stride)
  dilation = np.array(dilation)
  input_size = np.array(input_size)
  filter = np.array(filter)
  effective_filter_size = (filter - 1) * dilation + 1
  if padding == "same":
    output_size = (input_size + stride - 1) // stride
  elif padding == "valid":
    output_size = (input_size + stride - effective_filter_size) // stride
  padding_needed = np.int64((output_size - 1) * stride + effective_filter_size - input_size)
  padding_needed = padding_needed.clip(min=0)

  padding_before = padding_needed // 2
  padding_after = padding_needed - padding_before
  pad = [i for i in padding_before] + [i for i in padding_after]
  return pad

class BaseNode():
  def __init__(self, info):
    self.name = str(info["name"])
    self.op_type = str(info["op_type"])
    self.inputs = list(info["inputs"])
    self.outputs = list(info["outputs"])

class PytorchNode(BaseNode):
  def __init__(self, node):
    info = dict()
    info["name"] = node.output().debugName()
    info["op_type"] = node.kind()
    info["inputs"] = [inp.debugName() for inp in node.inputs()]
    info["outputs"] = [outp.debugName() for outp in node.outputs()]
    super().__init__(info)
    self.node_proto = node


class PytorchConverter(BaseConverter):
  MLIRImporterTypeStr = {
    "float64": "f64",
    "float32": "F32",
    "float16": "F16",
    "int8": "INT8",
    "int16": "INT16",
    "int32": "INT32",
    "int64": "INT64",
    "uint8": "UINT8",
    "uint16": "UINT16",
    "uint32": "UINT32",
    "uint64": "UINT64",
    "bool": "BOOL",
  }

  def __init__(self,
               model_name: str,
               pytorch_file,
               input_shapes: list,
               output_names: list,
               preprocess_args=None):
    super().__init__()
    self.model_name = model_name
    self.weight_file = "{}_top_weight.npz".format(model_name)
    self.model = None
    self.mlir = None
    self.node_name_mapping = {}  # used in pytorch opt
    self.const_val = {}

    self.load_torch_model(pytorch_file, input_shapes, output_names)
    self.shape_infer()
    self.init_MLIRImporter()
    self.preprocess_args = preprocess_args
    self.converted_nodes = list()
    self.op_factory = {
      "aten::_convolution": lambda node: self.convert_conv_op(node),
      "aten::_convolution_mode": lambda node: self.convert_conv_mode_op(node),
      "aten::add": lambda node: self.convert_add_op(node),
      "aten::prelu": lambda node: self.convert_prelu_op(node),
    }
    self.check_op_names()

  def __del__(self):
    if self.mlir != None:
      del self.mlir
      self.mlir = None

  def shape_infer(self):
    in_tensors = [torch.rand(s) for s in self.input_shapes]
    with torch.no_grad():
      out_tensors = self.model(*in_tensors)
    output_tensors = []
    if isinstance(out_tensors, tuple):
      for o in out_tensors:
        output_tensors.append(o.numpy())
    else:
      output_tensors.append(out_tensors.numpy())
    self.output_types = [self.MLIRImporterTypeStr[o.dtype.name] for o in output_tensors]
    self.output_shapes=[o.shape for o in output_tensors]
    for idx, name in enumerate(self.output_names):
      self.addShape(name, self.output_shapes[idx])

  def get_all_op_names(self):
    """Return all operator names in the input graph"""
    self.nodes = list(self.graph.nodes())
    prim_blocks = ["prim::If", "prim::Loop"]
    for prim in prim_blocks:
      prim_nodes = self.graph.findAllNodes(prim, recurse=True)
      for prim_node in prim_nodes:
        for block in prim_node.blocks():
          self.nodes += block.nodes()
    return set(node.kind() for node in self.nodes)

  def check_op_names(self):
    op_names = self.get_all_op_names()
    known_ops = [
        "prim::Constant",
        "prim::GetAttr",
        "prim::ListConstruct",
        "prim::ListUnpack",
        "prim::TupleConstruct",
        "prim::TupleUnpack",
        "prim::RaiseException",
        "prim::If",
        "prim::Loop",
    ]
    known_ops += list(self.op_factory.keys())

    unknown_ops = []
    for op_name in op_names:
      if op_name not in known_ops:
        if not (op_name.endswith("_") and op_name[:-1] in known_ops):
          unknown_ops.append(op_name)
    if len(unknown_ops) != 0:
      raise RuntimeError(
        "The following operators are not implemented: {}".format(unknown_ops))

  def load_torch_model(self, pytorch_file, input_shapes: list, output_names: list):
    if isinstance(pytorch_file, str):
      self.model = torch.jit.load(pytorch_file, map_location=torch.device('cpu'))
    else:
      self.model = pytorch_file
    self.model.eval()
    self.graph = self.model.graph
    is_module = isinstance(self.model, torch.jit.ScriptModule)
    inputs = list(self.graph.inputs())
    inputs = inputs[1:] if is_module else inputs

    self.input_names = []
    for idx, inp in enumerate(inputs):
      self.input_names.append(inp.debugName())
      self.addShape(inp.debugName(), input_shapes[idx])

    self.output_names = []
    for outp in self.graph.outputs():
      self.output_names.append(outp.debugName())

    self.num_input = len(self.input_names)
    self.num_output = len(self.output_names)
    self.input_shapes = input_shapes
    self.input_types = [self.MLIRImporterTypeStr["float32"] for i in range(self.num_input)]

  def init_MLIRImporter(self):
    input_shapes = list()
    for _name in self.input_names:
      input_shapes.append(self.getShape(_name))
    output_shapes = list()
    for _name in self.output_names:
      output_shapes.append(self.getShape(_name))
    # init importer
    self.mlir = MLIRImporter(input_shapes, output_shapes, self.model_name, self.input_types)
    self.weight_file = self.mlir.weight_file

  def get_constant_list(self, node):
    data = []
    for input in node.inputs():
      if input.debugName() in self.const_val.keys():
        data.append(self.const_val[input.debugName()])
      else:
        raise KeyError("can not find const data")
    return node.output().debugName(), data

  def read_node(self, node):
    if node.kind() == 'prim::Constant':
      name, data, is_tensor = _get_constant(node)
      if not is_tensor:
        self.const_val[name] = data
      else:
        self.addWeight(name, data)
    elif node.kind() == 'prim::ListConstruct':
      name, data = self.get_constant_list(node)
      self.const_val[name] = data
    elif node.kind() == 'prim::ListUnpack':
      raise TypeError("can not find kind {}".format(node.kind()))
    elif node.kind() == 'prim::TupleConstruct':
      raise TypeError("can not find kind {}".format(node.kind()))
    elif node.kind() == 'prim::TupleUnpack':
      raise TypeError("can not find kind {}".format(node.kind()))
    elif node.kind() == 'prim::GetAttr':
      raise TypeError("can not find kind {}".format(node.kind()))
    elif node.kind() == 'prim::If':
      raise TypeError("can not find kind {}".format(node.kind()))
    elif node.kind() == 'prim::Loop':
      raise TypeError("can not find kind {}".format(node.kind()))
    else:
      return False
    return True

  def generate_mlir(self, mlir_file: str):
    """convert all to mlir"""
    # add input op
    for idx, _name in enumerate(self.input_names):
      input_op = self.mlir.create_input_op(_name, idx, **{})
      self.addOperand(_name, input_op)

    def NoneAndRaise(node):
      raise RuntimeError("{} Op not support now".format(node.op_type))

    self.converted_nodes.clear()
    for node in self.graph.nodes():
      if (self.read_node(node) is False):
        nd = PytorchNode(node)
        self.converted_nodes.append(nd)
    # checkout all type is supported
    unsupported = set()
    for n in self.converted_nodes:
      if n.op_type not in self.op_factory:
        unsupported.add(n.op_type)
    if unsupported:
      raise RuntimeError("Op not support:{}".format(unsupported))

    for n in self.converted_nodes:
      self.op_factory.get(n.op_type, lambda x: NoneAndRaise(x))(n)
    # add return op
    return_op = list()
    # Set output
    for idx, _name in enumerate(self.output_names):
      op = self.getOperand(_name)
      return_op.append(op)

    self.mlir.create_return_op(return_op)
    mlir_txt = self.mlir.print_module()
    with open(mlir_file, "w") as f:
      f.write(mlir_txt)
    self.WeightToNpz(self.weight_file)
    print("Save mlir file: {}".format(mlir_file))

  def convert_base_conv_op(self, pytorch_node: PytorchNode, mode = False):
    op = self.getOp(pytorch_node.inputs[0])
    strides = _data_expand(self.const_val[pytorch_node.inputs[3]], 2)
    pads = self.const_val[pytorch_node.inputs[4]]
    dilations = _data_expand(self.const_val[pytorch_node.inputs[5]], 2)
    group = self.const_val[pytorch_node.inputs[6 if mode else 8]]
    kernel_shape = self.getShape(pytorch_node.inputs[1])
    kernel_shape = kernel_shape[2:]
    if mode == True:
      input_size = self.getShape(pytorch_node.inputs[0])[2:]
      pads = _compute_pad(strides, dilations, input_size, kernel_shape, pads)
    else:
      transposed = self.const_val[pytorch_node.inputs[6]]
      output_padding = self.const_val[pytorch_node.inputs[7]]
      if isinstance(pads, int):
        pads = [pads for i in range(4)]
      elif len(pads) == 2:
        pads = [pads[0], pads[0], pads[1], pads[1]]

    operands = list()
    operands.append(op)
    filter_op = self.getOp(pytorch_node.inputs[1])
    operands.append(filter_op)
    if pytorch_node.inputs[2] not in self.const_val.keys() or self.const_val[pytorch_node.inputs[2]] is not None:
      bias_op = self.getOp(pytorch_node.inputs[2])
    else:
      bias_op = self.mlir.none_op
    operands.append(bias_op)
    p = {
        'name': pytorch_node.name,
        'kernel_shape': kernel_shape,
        'strides': strides,
        'dilations': dilations,
        'pads': pads,
        'group': group,
        'do_relu': False,
        'ins': [],
    }
    output_shape = self.getShape(pytorch_node.name)
    new_op = self.mlir.create_conv_op(operands, output_shape, **p)
    self.addOperand(pytorch_node.name, new_op)

  def convert_conv_op(self, pytorch_node: PytorchNode):
    self.convert_base_conv_op(pytorch_node)

  def convert_conv_mode_op(self, pytorch_node: PytorchNode):
    self.convert_base_conv_op(pytorch_node, True)

  def convert_add_op(self, pytorch_node: PytorchNode):
    op0 = self.getOp(pytorch_node.inputs[0])
    op1 = self.getOp(pytorch_node.inputs[1])
    scale = self.const_val[pytorch_node.inputs[2]]
    assert scale == 1
    p = {
        'name': pytorch_node.name,
        'do_relu': False,
    }
    output_shape = self.getShape(pytorch_node.name)
    new_op = self.mlir.create_add_op([op0, op1], output_shape, **p)
    self.addOperand(pytorch_node.name, new_op)

  def convert_prelu_op(self, pytorch_node: PytorchNode):
    op0 = self.getOp(pytorch_node.inputs[0])
    op1 = self.getOp(pytorch_node.inputs[1])
    output_shape = self.getShape(pytorch_node.name)
    new_op = self.mlir.create_prelu_op([op0, op1], output_shape, **{'name': pytorch_node.name})
    self.addOperand(pytorch_node.name, new_op)
