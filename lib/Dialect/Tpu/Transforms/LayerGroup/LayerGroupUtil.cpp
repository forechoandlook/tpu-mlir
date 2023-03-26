//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/LayerGroupUtil.h"
#include "tpu_mlir/Backend/Arch.h"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Module.h"

using namespace tpu_mlir::backend;

namespace tpu_mlir {
namespace tpu {

shape_secs_t get_group_max_secs(const LgInfo &lg_info) {
  int64_t n, c, d, h, w;
  module::getNCDHW(lg_info.group_ops[0]->getOperand(0), n, c, d, h, w,
                  lg_info.type);
  int64_t max_nsecs = n;
  if (isa<tpu::AddOp, tpu::SubOp, tpu::MulOp, tpu::DivOp, tpu::MaxOp, tpu::MinOp>(lg_info.group_ops[0])) {
    module::getNCDHW(lg_info.group_ops[0]->getOperand(1), n, c, d, h, w,
                    lg_info.type);
    max_nsecs = std::max(n, max_nsecs);
  }
  int64_t max_hsecs = llvm::maxIntN(64);
  // Need consider n_align if backend is BM1684
  int64_t n_align = 1;
  for (auto op : lg_info.group_ops) {
    auto lgOp = cast<LocalGenInterface>(op);
    for (auto v : get_output_values(op)) {
      module::getNCDHW(v, n, c, d, h, w, lg_info.type);
      if (Arch::ALIGN_4N) {
        auto stype = module::getStorageType(v);
        n_align = 32 / stype.getIntOrFloatBitWidth();
      }
      max_nsecs = std::min(max_nsecs, ceiling_func(n, n_align));
      if (succeeded(lgOp.AllowDataSplit(2, lg_info.type))) {
        max_hsecs = std::min(max_hsecs, h);
      } else {
        max_hsecs = 1;
      }
    }
  }

  return shape_secs_t{.nsecs = max_nsecs, .hsecs = max_hsecs};
}

shape_secs_t init_group_data_secs(const LgInfo &lg_info) {
  shape_secs_t shape_secs = {1, 1};
  if (lg_info.group_ops.size() == 1) {
    return shape_secs;
  }

  shape_secs_t max_shape_secs = get_group_max_secs(lg_info);
  // int64_t batch_size =
  // module::getShape(lg_info.group_ops[0]->getOperand(0))[0];
  int64_t in_n, in_c, in_d, in_h, in_w;
  int64_t out_n, out_c, out_d, out_h, out_w;
  for (auto op : lg_info.group_ops) {
    auto ins = get_input_values(op);
    auto outs = get_output_values(op);
    module::getNCDHW(ins[0], in_n, in_c, in_d, in_h, in_w, lg_info.type);
    module::getNCDHW(outs[0], out_n, out_c, out_d, out_h, out_w, lg_info.type);
    int64_t in0_lmem_bytes =
        Arch::get_tensor_lmem_bytes(ins[0], in_n, in_c, in_d, in_h, in_w);
    int64_t out0_lmem_bytes =
        Arch::get_tensor_lmem_bytes(outs[0], out_n, out_c, out_d, out_h, out_w);

    int64_t total_size = in0_lmem_bytes + out0_lmem_bytes;
    auto lg_op = cast<LocalGenInterface>(op);
    total_size += lg_op.getBufferSize(in0_lmem_bytes, out0_lmem_bytes, in_n,
                                      in_h, out_n, out_h, lg_info.type);
    for (size_t i = 1; i < ins.size(); ++i) {
      if(module::isWeight(ins[i])){
        total_size += Arch::get_weight_lmem_bytes(ins[i], lg_info.type, is_eu_align(ins[i]));
      } else {
        module::getNCDHW(ins[i], in_n, in_c, in_d, in_h, in_w, lg_info.type);
        total_size += Arch::get_tensor_lmem_bytes(ins[i], in_n, in_c, in_d, in_h, in_w);
      }
    }
    for (size_t i = 1; i < outs.size(); ++i) {
      module::getNCDHW(outs[i], out_n, out_c, out_d, out_h, out_w, lg_info.type);
      total_size +=
          Arch::get_tensor_lmem_bytes(outs[i], out_n, out_c, out_d, out_h, out_w);
    }

    // Need consider different backends
    int64_t total_secs = ceiling_func(total_size, Arch::LMEM_BYTES);
    shape_secs.nsecs =
        std::max(std::min(total_secs, max_shape_secs.nsecs), shape_secs.nsecs);
    total_secs = ceiling_func(total_secs, shape_secs.nsecs);
    shape_secs.hsecs = std::max(total_secs, shape_secs.hsecs);
  }

  return shape_secs;
}

static int64_t get_split_max_secs(BasicTimeStepPtr time_step) {
  int64_t timestep_num = time_step->get_timestep_num();
  std::vector<int64_t> lmem_req(timestep_num, 0);
  const MemBuff &lmem_buffer = time_step->get_lmem_buffer();

  auto update_lmem_req = [&lmem_req, &timestep_num](int64_t start_ts,
                                                    int64_t end_ts,
                                                    int64_t lmem_size) {
    if (start_ts <= end_ts) {
      for (int64_t ts = start_ts; ts <= end_ts; ++ts) {
        lmem_req[ts] += lmem_size;
      }
    } else {
      for (int64_t ts = 0; ts <= end_ts; ++ts) {
        lmem_req[ts] += lmem_size;
      }
      for (int64_t ts = start_ts; ts <= timestep_num - 1; ++ts) {
        lmem_req[ts] += lmem_size;
      }
    }
  };

  for (auto iter = lmem_buffer.begin(); iter != lmem_buffer.end(); ++iter) {
    int64_t start_ts = iter->second.start_ts;
    int64_t end_ts = iter->second.end_ts;
    update_lmem_req(start_ts, end_ts, (iter->second).size);
  }

  std::stable_sort(lmem_req.begin(), lmem_req.end(), std::greater<int64_t>());
  return ceiling_func(lmem_req[0], Arch::LMEM_BYTES);
}

void update_tensor_infos(const LgInfo &lg_info, TensorInfo &tensor_infos) {
  for (auto &iter : tensor_infos) {
    auto v = iter.first;
    iter.second.use_3ic_opt = use_3ic(v);
    iter.second.eu_align = is_eu_align(v);
    iter.second.need_bcast = need_bcast(v);
  }

  tensor_info_t ti(TIMESTEP_LOAD);
  int64_t n, c, d, h, w;
  for (auto op : lg_info.group_ops) {
    auto ins = get_input_values(op);
    for (auto in : ins) {
      if (auto src_op = dyn_cast_or_null<top::WeightOp>(in.getDefiningOp())) {
        ti.eu_align = is_eu_align(in);
        ti.need_bcast = need_bcast(in);
        module::getNCDHW(in, n, c, d, h, w, lg_info.type);
        ti.slice_info.n.clear();
        ti.slice_info.h.clear();
        ti.slice_info.n.push_back(std::make_pair((int64_t)0, (int64_t)n));
        ti.slice_info.h.push_back(std::make_pair((int64_t)0, (int64_t)h));
        tensor_infos[in] = ti;
      }
    }
  }
}

bool update_data_split(BasicTimeStepPtr time_step, const LgInfo &lg_info,
                       shape_secs_t &shape_secs) {
  shape_secs.nsecs = 1;
  shape_secs.hsecs = 1;
  bool status = false;
  auto &tensor_infos = time_step->get_tensor_infos();
  shape_secs_t max_shape_secs = get_group_max_secs(lg_info);
  for (int64_t nsec = 1; nsec <= max_shape_secs.nsecs; ++nsec) {
    shape_secs.nsecs = nsec;
    tensor_infos.clear();
    if (stripe_mine_max_slice(lg_info, shape_secs, tensor_infos) == false) {
      return false;
    }
    time_step->update_all_mem_buffer_size(lg_info);

    int64_t total_secs = get_split_max_secs(time_step);
    shape_secs.nsecs =
        std::max(shape_secs.nsecs, std::min(max_shape_secs.nsecs, total_secs));
    shape_secs.hsecs = ceiling_func(total_secs, shape_secs.nsecs);
    if (shape_secs.hsecs <= max_shape_secs.hsecs) {
      status = true;
      break;
    }
  }
  update_tensor_infos(lg_info, tensor_infos);
  return status;
}

bool strip_back_judge(Value v, const LgInfo &lg_info,
                      const std::multiset<Operation *> &op_set,
                      const std::set<Value, value_compare> &out_tensor_set) {
  auto users = v.getUsers();
  bool res = true;
  bool has_outer_group_user = false;
  for (auto op : users) {
    if (std::find(lg_info.group_ops.begin(), lg_info.group_ops.end(), op) !=
        lg_info.group_ops.end()) {
      if (op_set.count(op) == 0) {
        return false;
      }
    } else {
      has_outer_group_user = true;
    }
  }

  if (has_outer_group_user) {
    res = out_tensor_set.find(v) != out_tensor_set.end();
  }
  return res;
}

inline bool is_same_slice(const slice_pair_t &a, const slice_pair_t &b) {
  return (a.first == b.first) && (a.second == b.second);
}

bool is_same_slice_info(const slice_info_t &si0, const slice_info_t &si1) {
  if (si0.n.size() != si1.n.size() || si0.h.size() != si1.h.size()) {
    return false;
  }
  // check n
  for (size_t i = 0; i < si0.n.size(); ++i) {
    if (false == is_same_slice(si0.n[i], si1.n[i])) {
      return false;
    }
  }
  for (size_t i = 0; i < si0.h.size(); ++i) {
    if (false == is_same_slice(si0.h[i], si1.h[i])) {
      return false;
    }
  }
  // for (auto it : llvm::zip(si0.n, si1.n)) {
  //   if (false == is_same_slice(std::get<0>(it), std::get<1>(it))) {
  //     return false;
  //   }
  // }
  // // check h
  // for (auto it : llvm::zip(si0.h, si1.h)) {
  //   if (false == is_same_slice(std::get<0>(it), std::get<1>(it))) {
  //     return false;
  //   }
  // }
  return true;
}

bool is_broadcast_binary(Operation *op, Value in) {
  if (!isa<tpu::AddOp, tpu::SubOp, tpu::MulOp, tpu::MaxOp, tpu::MinOp>(op)) {
    return false;
  }
  auto other = in == op->getOperand(0) ? op->getOperand(1): op->getOperand(0);
  auto in_shape = in.getType().cast<RankedTensorType>().getShape();
  auto other_shape = other.getType().cast<RankedTensorType>().getShape();
  if (in_shape.size() != other_shape.size()) {
    return false;
  }
  for (int i = 0; i < in_shape.size(); ++i) {
    if (in_shape[i] != other_shape[i] && in_shape[i] == 1) {
      return true;
    }
  }
  return false;
}

slice_info_t get_out_slice_info(const shape_secs_t &shape_secs, int64_t n,
                                int64_t h) {
  slice_info_t slice_info;
  int64_t secs, idx, slice, step;
  // n slice info
  secs = shape_secs.nsecs;
  for (int64_t i = 0; i < secs; ++i) {
    step = n / secs + (n % secs > i);
    idx = n / secs * i + (n % secs > i ? i : n % secs);
    slice = (n - idx) > step ? step : (n - idx);
    // assert(idx < n);
    slice_info.n.emplace_back(slice_pair_t(idx, slice));
  }
  // h slice_info
  secs = shape_secs.hsecs;
  for (int64_t i = 0; i < secs; ++i) {
    step = h / secs + (h % secs > i);
    idx = h / secs * i + (h % secs > i ? i : h % secs);
    slice = (h - idx) > step ? step : (h - idx);
    // assert(idx < h);
    slice_info.h.emplace_back(slice_pair_t(idx, slice));
  }

  return slice_info;
}

bool get_backward_slice_info(slice_info_t &in_si, const slice_info_t &out_si,
                             Operation *op, Value in,
                             const shape_secs_t &shape_secs,
                             group_type_t group_type) {
  int64_t n, c, d, h, w;
  module::getNCDHW(in, n, c, d, h, w, group_type);

  auto lg_op = cast<LocalGenInterface>(op);
  int64_t idx = 0, slice = 0;
  bool is_broadcast_tensor = is_broadcast_binary(op, in);
  if (shape_secs.nsecs == 1) {
    in_si.n.emplace_back(slice_pair_t(0, n));
  } else {
    for (auto &s : out_si.n) {
      auto ret = lg_op.BackwardN(idx, slice, s.first, s.second);
      if (is_broadcast_tensor && n == 1) {
        idx = 0;
        slice = 1;
      } else {
        if (failed(ret) || slice == 0) {
          return false;
        }
      }
      in_si.n.emplace_back(slice_pair_t(idx, slice));
    }
  }

  int64_t pre_end_idx = 0;
  idx = slice = 0;
  if (shape_secs.hsecs == 1) {
    in_si.h.emplace_back(slice_pair_t(0, h));
  } else {
    for (int i = 0; i < out_si.h.size(); i++) {
      auto &s = out_si.h[i];
      auto ret = lg_op.BackwardH(idx, slice, s.first, s.second);
      if (is_broadcast_tensor && h == 1) {
        idx = 0;
        slice = 1;
      } else {
        bool end_reached = idx + slice == pre_end_idx;
        if (failed(ret) || slice == 0 || (idx == 0 && i > 0) || end_reached) {
          return false;
        }
      }
      pre_end_idx = idx + slice;
      in_si.h.emplace_back(slice_pair_t(idx, slice));
    }
  }
  return true;
}

bool check_hsecs(Value value, slice_info_t &si, group_type_t group_type) {
  assert(si.h.size() > 0);
  int64_t n, c, d, h, w;
  module::getNCDHW(value, n, c, d, h, w, group_type);
  int64_t total_h = 0;
  for (auto &it : si.h) {
    total_h += it.second;
  }
  if (total_h * 2 > h * 3) { // h increase 1.5 times
    return false;
  }
  return true;
}

static bool
backward_update_slice(const LgInfo &lg_info, const shape_secs_t &shape_secs,
                      Value out, std::list<Value> &tensor_branchs,
                      TensorInfo &tensor_infos,
                      std::multiset<Operation *> &op_set,
                      const std::set<Value, value_compare> &out_tensor_set) {

  // Don't backward when this out tensor is the input of the group
  if (std::find(lg_info.group_ins.begin(), lg_info.group_ins.end(), out) !=
      lg_info.group_ins.end()) {
    // return check_hsecs(out, tensor_infos[out].slice_info, lg_info.type);
    return true;
  }
  auto op = out.getDefiningOp();
  op_set.insert(op);

  slice_info_t &out_si = tensor_infos[out].slice_info;

  for (auto in : op->getOperands()) {
    auto pre_op = in.getDefiningOp();
    if (pre_op != nullptr && isa<top::WeightOp, top::NoneOp>(pre_op)) {
      continue;
    }
    slice_info_t si;
    auto ret =
        get_backward_slice_info(si, out_si, op, in, shape_secs, lg_info.type);
    if (ret == false) {
      return false;
    }
    auto iter = tensor_infos.find(in);
    if (iter != tensor_infos.end()) {
      if (false == is_same_slice_info(si, iter->second.slice_info)) {
        return false;
      }
    } else {
      tensor_infos[in] = tensor_info_t(si);
    }
    if (strip_back_judge(in, lg_info, op_set, out_tensor_set)) {
      tensor_branchs.push_back(in);
    }
  }
  return true;
}

bool stripe_mine_max_slice(const LgInfo &lg_info,
                           const shape_secs_t &shape_secs,
                           TensorInfo &tensor_infos) {
  if (lg_info.group_ops.size() == 1) {
    return true;
  }
  tensor_infos.clear();

  int64_t n, c, d, h, w;
  int64_t max_nslice = 0, max_hslice = 0;
  std::list<Value> tensor_branchs;
  std::multiset<Operation *> op_set;
  std::set<Value, value_compare> out_tensor_set;
  slice_info_t si;
  for (auto out : lg_info.group_outs) {
    module::getNCDHW(out, n, c, d, h, w, lg_info.type);
    max_nslice = std::max(max_nslice, ceiling_func(n, shape_secs.nsecs));
    if (Arch::ALIGN_4N) {
      auto stype = module::getStorageType(out);
      int64_t align_n = 32 / stype.getIntOrFloatBitWidth();
      max_nslice = align_up(max_nslice, align_n);
    }
    max_hslice = (h + shape_secs.hsecs - 1) / shape_secs.hsecs;
    si.n.clear();
    si.h.clear();
    si.n.emplace_back(slice_pair_t(0, max_nslice));
    si.h.emplace_back(slice_pair_t(0, max_hslice));
    tensor_infos[out] = tensor_info_t(si);

    out_tensor_set.insert(out);
    if (strip_back_judge(out, lg_info, op_set, out_tensor_set)) {
      tensor_branchs.push_back(out);
    }
  }

  bool ret = false;
  while (!tensor_branchs.empty()) {
    auto out_tensor = tensor_branchs.front();
    tensor_branchs.pop_front();
    ret = backward_update_slice(lg_info, shape_secs, out_tensor, tensor_branchs,
                                tensor_infos, op_set, out_tensor_set);
    if (!ret) {
      return false;
    }
  }

  //  if (check_tensor_slice(tensor_infos) == false) {
  //    return false;
  //  }

  return true;
}

bool stripe_mine_idx_slice(const LgInfo &lg_info,
                           const shape_secs_t &shape_secs,
                           TensorInfo &tensor_infos) {
  if (lg_info.group_ops.size() == 1) {
    return true;
  }
  tensor_infos.clear();

  int64_t n, c, d, h, w;
  std::list<Value> tensor_branchs;
  std::multiset<Operation *> op_set;
  std::set<Value, value_compare> out_tensor_set;
  for (auto out : lg_info.group_outs) {
    module::getNCDHW(out, n, c, d, h, w, lg_info.type);
    auto si = get_out_slice_info(shape_secs, n, h);

    tensor_infos[out] = tensor_info_t(si);
    out_tensor_set.insert(out);
    if (strip_back_judge(out, lg_info, op_set, out_tensor_set)) {
      tensor_branchs.push_back(out);
    }
  }

  bool ret = false;
  while (!tensor_branchs.empty()) {
    auto out_tensor = tensor_branchs.front();
    tensor_branchs.pop_front();
    ret = backward_update_slice(lg_info, shape_secs, out_tensor, tensor_branchs,
                                tensor_infos, op_set, out_tensor_set);
    if (!ret) {
      return false;
    }
  }

  return true;
}

void get_max_slice_nh(const slice_info_t &slice_info, int64_t &max_nslice,
                      int64_t &max_hslice) {
  max_nslice = 0;
  max_hslice = 0;
  for (auto &slice : slice_info.n) {
    max_nslice = std::max(max_nslice, slice.second);
  }
  for (auto &slice : slice_info.h) {
    max_hslice = std::max(max_hslice, slice.second);
  }
}

int64_t get_buffer_size(Value v, const tensor_info_t &ti,
                        group_type_t group_type) {
  int64_t buf_size = 0;
  int64_t n, c, d, h, w;
  module::getNCDHW(v, n, c, d, h, w, group_type);
  if (module::isWeight(v)) {
    if (group_type == GROUP_SMALL_C) {
      buf_size = Arch::get_tensor_lmem_bytes(v, n, c, d, h, w, ti.eu_align);
    } else {
      buf_size = Arch::get_weight_lmem_bytes(v, group_type, ti.eu_align);
    }
  } else {
    int64_t nslice, hslice;
    auto &si = ti.slice_info;
    get_max_slice_nh(si, nslice, hslice);
    buf_size =
        Arch::get_tensor_lmem_bytes(v, nslice, hslice, group_type, ti.eu_align);
  }
  return buf_size;
}

void set_fake_local_layer_param(Operation *op, int64_t nidx, int64_t nslice,
                                int64_t hidx, int64_t hslice) {
  auto ctx = op->getContext();
  auto builder = OpBuilder(ctx);
  int64_t group_type = 0;
  module::isOpInGroup(op, &group_type);
  auto lg_attr = LayerGroupAttr::get(
      ctx, 0, 0, 0, 0, true, builder.getDenseI64ArrayAttr({hidx}),
      builder.getDenseI64ArrayAttr({hslice}),
      builder.getDenseI64ArrayAttr({nidx}),
      builder.getDenseI64ArrayAttr({nslice}), 0, 0, group_type);
  op->setAttr(LocalGenInterface::kLayerGroupAttrName, lg_attr);
}

void delete_fake_local_layer_param(Operation *op) {
  op->removeAttr(LocalGenInterface::kLayerGroupAttrName);
}

void generate_fake_global_addr(Operation *op) {
  int64_t offset = Arch::LMEM_BANK_BYTES;
  int64_t i = 0;
  auto ins = get_input_values(op);
  auto outs = get_output_values(op);
  for (auto in : ins) {
    module::setAddress(in, offset * i);
    ++i;
  }
  for (auto out : outs) {
    module::setAddress(out, offset * i);
    ++i;
  }
}

void delete_fake_global_addr(Operation *op) {

  auto ins = get_input_values(op);
  auto outs = get_output_values(op);
  for (auto in : ins) {
    auto type = in.getType().cast<RankedTensorType>();
    Builder builder(in.getContext());
    auto new_type =
        RankedTensorType::get(type.getShape(), type.getElementType());
    in.setType(new_type);
  }
  for (auto out : outs) {
    auto type = out.getType().cast<RankedTensorType>();
    Builder builder(out.getContext());
    auto new_type =
        RankedTensorType::get(type.getShape(), type.getElementType());
    out.setType(new_type);
  }
}

bool is_eu_align_cv18xx(Value opd) {
  auto op = *opd.getUsers().begin();
  if (module::isWeight(opd)) {
    if (isa<tpu::LutOp>(op)) {
      if (opd == op->getOperand(1)) {
        return false;
      }
    } else if (isa<tpu::LutBF16Op>(op)) {
      if (opd == op->getOperand(1) || opd == op->getOperand(2)) {
        return false;
      }
    } else if (isa<tpu::ScaleLutOp>(op)) {
      if (opd == op->getOperand(1)) {
        return false;
      }
    } else if (isa<tpu::ScaleOp>(op)) {
      return false;
    } else if (auto castOp = dyn_cast<tpu::Conv2DOp>(op)) {
      auto attr = castOp.parseParam();
      if (opd == op->getOperand(1) && attr.is_dw) {
        // if is dw conv, filter need to align.
        return true;
      }
      if (module::isUniformQuantized(castOp.getOutput()) &&
          opd == op->getOperand(2) && !attr.is_dw && attr.groups > 1) {
        // if is int8 group conv, bias need to align.
        return true;
      }
      return false;
    } else if (auto castOp = dyn_cast<tpu::DeconvOp>(op)) {
      auto attr = castOp.parseParam();
      if (opd == op->getOperand(1) && attr.is_dw) {
        // if is dw conv, filter need to align.
        return true;
      }
      if (module::isUniformQuantized(castOp.getOutput()) &&
          opd == op->getOperand(2) && !attr.is_dw && attr.g > 1) {
        // if is int8 group conv, bias need to align.
        return true;
      }
      return false;
    } else if (isa<tpu::LayerNormOp>(op)) {
      return false;
    } else {
      return true; // prelu concat
    }
  }
  return true;
}

bool is_eu_align_bm1686(Value opd) {
  auto op = *opd.getUsers().begin();
  if (module::isWeight(opd)) {
    if (isa<tpu::Conv1DOp, tpu::Conv2DOp, tpu::Conv3DOp, tpu::DeconvOp>(op)) {
      if ((opd == op->getOperand(1) || opd == op->getOperand(2))) {
        return false;
      }
    } else if (isa<tpu::RequantIntAxisOp>(op)) {
      if ((opd == op->getOperand(1))) {
        return false;
      }
    } else if (isa<tpu::PReluOp, tpu::ScaleOp>(op)) {
      return false;
    } else {
      return true;
    }
  }
  return true;
}

bool is_eu_align_common(Value opd) {
  auto op = *opd.getUsers().begin();
  if (module::isWeight(opd)) {
    if (isa<tpu::Conv1DOp, tpu::Conv2DOp, tpu::Conv3DOp, tpu::DeconvOp>(op)) {
      if ((opd == op->getOperand(1) || opd == op->getOperand(2))) {
        return false;
      }
    } else if (isa<tpu::PReluOp, tpu::ScaleOp>(op)) {
      return false;
    } else {
      return true;
    }
  }
  return true;
}

bool is_eu_align(Value opd) {
  // Eu align rule may be different in different platforms
  if (module::isBM1686()) {
    return is_eu_align_bm1686(opd);
  } else if (module::isCV18xx()) {
    return is_eu_align_cv18xx(opd);
  } else {
    return is_eu_align_common(opd);
  }
}

bool need_bcast(Value opd) {
  if (opd.hasOneUse() == false) {
    return false;
  }
  auto use_op = *opd.getUsers().begin();
  if (auto cast_op = dyn_cast<tpu::LutOp>(use_op)) {
    return opd == cast_op.getTable();
  } else if (auto cast_op = dyn_cast<tpu::LutBF16Op>(use_op)) {
    return opd == cast_op.getTable() || opd == cast_op.getMantissa();
  } else if (auto cast_op = dyn_cast<tpu::LayerNormOp>(use_op)) {
    return module::isCV18xx() && isa<top::WeightOp>(opd.getDefiningOp());
  } else {
    return false;
  }
}

int64_t use_3ic(Value opd) {
  for (auto use_op : opd.getUsers()) {
    if (auto cast_op = dyn_cast<tpu::Conv2DOp>(*use_op)) {
      if (opd == cast_op.getInput()) {
        return cast_op.getUse_3icOptimize();
      }
    }
  }
  return 0;
}

std::vector<Value> get_input_values(Operation *op) {
  auto value_vec = std::vector<Value>();
  for (auto in : op->getOperands()) {
    if (in.getType().isa<NoneType>()) {
      continue;
    }
    value_vec.push_back(in);
  }
  return std::move(value_vec);
}

std::vector<Value> get_output_values(Operation *op) {
  auto value_vec = std::vector<Value>();
  for (auto out : op->getResults()) {
    if (out.getType().isa<NoneType>()) {
      continue;
    }
    value_vec.push_back(out);
  }
  return std::move(value_vec);
}

} // namespace tpu
} // namespace tpu_mlir
