# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import TYPE_CHECKING, Any, Optional, Tuple, Union, cast

import copy
import os
import re
import time
import logging 
import collections
import importlib

import torch
from torch import Tensor
import torch.distributed as dist
from torch.nn import ModuleList
import torch.nn.functional as F
import random

from ..impls import communicate as C
from ..impls.fast_dispatch import fast_encode, fast_decode, extract_critical
from ..impls.overlap import a2a_ffn_overlap_forward
from . import losses


class MOELayer(torch.nn.Module):
    """Tutel optimized MOELayer
    """
    @staticmethod
    def global_expert_count(num_local_experts, group=None):
        if not isinstance(num_local_experts, int):
            num_local_experts = -int(1 / (num_local_experts + 1e-5))
        world_size = C.get_world_size(group)
        if num_local_experts == 0:
            raise Exception("Invalid value of num_local_experts: %d" % num_local_experts)
        if num_local_experts > 0:
            return num_local_experts * world_size
        assert world_size % -num_local_experts == 0, f"Excepting {-num_local_experts} devices to share an expert param, while global device count is {world_size}."
        return world_size // -num_local_experts

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        buff_name = prefix + '_num_global_experts'
        if buff_name not in state_dict:
            logging.warning(f"\033[31mYou are loading a legacy format of checkpoint with at least one Tutel MoE layer inside, which wouldn't support new Tutel feature allowing the number of experts per checkpoint file to mutate.\033[0m")
            logging.warning(f"\033[31m  The next time you overwrite it with new checkpoint, the recording format will be updated automatically.\033[0m")
            logging.warning(f"\033[31m  However, the new format won't be compatible with early Tutel versions, unless you force loading it with `model.load_state_dict(.., strict=False)`.\033[0m")
            state_dict[buff_name] = self._num_global_experts
        else:
            state_experts, expect_experts = int(state_dict[buff_name]), self.num_global_experts
            assert state_experts == expect_experts, "Failed to load state from checkpoint: the number of global experts mismatch (%s <- %s)" % (expect_experts, state_experts)

        for name, param in self.experts.named_parameters():
            buff_name = prefix + 'experts.' + name
            assert buff_name in state_dict, "Could not find parameter `%s` in state_dict." % buff_name
            if state_dict[buff_name].numel() == param.numel():
                state_dict[buff_name] = state_dict[buff_name].view(param.shape)
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return super().state_dict(destination, prefix, keep_vars)

    @property
    def num_global_experts(self):
        return int(self._num_global_experts)

    def __init__(
        self,
        gate_type,
        model_dim: int,
        experts=None,
        scan_expert_func=None,
        result_func=None,
        group=None,
        seeds=None,
        a2a_ffn_overlap_degree=1,
        is_postscore=True,
        batch_prioritized_routing=False,
        normalize_gate=True,
        is_gshard_loss=True,
        parallel_type='adaptive:1',
        use_2dh=False,
        idea=None,
        **kwargs
    ):
        super().__init__()
        assert model_dim % 2 == 0, "Model_dim (%s) must be even value, while this Model_dim mod 2 > 0." % model_dim
        group = group or dist.group.WORLD

        if 'pad_samples' in kwargs:
            logging.warning(f"`pad_samples` option in Tutel Moe-layer has been deprecated, as Tutel always assumes `pad_samples=False` for better efficiency.")
            kwargs.pop('pad_samples')
        for k in kwargs:
            raise Exception('Unrecognized argument provided to Tutel Moe-layer: %s' % k)

        self.l_aux = 0
        self.group = group
        self.result_func = result_func
        self.skip_moe = (int(os.environ.get('SKIP_MOE', '0')) != 0)

        self.num_local_experts = experts.pop('count_per_node', 1)
        self.register_buffer('_num_global_experts', torch.tensor(MOELayer.global_expert_count(self.num_local_experts, self.group)))

        self.world_size = C.get_world_size(self.group)
        if self.num_global_experts < self.world_size:
            self.sharded_count = self.world_size // self.num_global_experts
            self.num_local_experts = 1
        else:
            self.sharded_count = 1

        self.auto_parallel, self.adaptive_degree, self.use_model_parallel = False, self.sharded_count, True
        self.valid_rs = [0] + [i for i in range(1, self.sharded_count + 1) if self.sharded_count % i == 0]

        if parallel_type.startswith('adaptive:'):
            self.adaptive_degree = int(parallel_type[parallel_type.index(':') + 1:])
            self.adaptive_degree = min(max(self.adaptive_degree, 0), self.sharded_count)
            if self.adaptive_degree not in self.valid_rs:
                raise Exception("Unexpected value of adaptive_degree: %d, expecting a candidate within %s." % (self.adaptive_degree, self.valid_rs))
        elif self.sharded_count == 1:
            pass
        elif parallel_type in ('data', 'model'):
            self.adaptive_degree = 1 if parallel_type == 'data' else self.sharded_count
        elif parallel_type == 'auto':
            self.adaptive_degree = 1
        else:
            raise Exception('Unrecognized parallel type specified: %s' % parallel_type)

        self.model_dim = model_dim

        self.is_postscore = is_postscore
        self.batch_prioritized_routing = batch_prioritized_routing
        if int(os.environ.get('BATCH_PRIO', 0)) != 0:
            self.batch_prioritized_routing = True
        self.normalize_gate = normalize_gate
        self.is_gshard_loss = is_gshard_loss

        self.a2a_ffn_overlap_degree = a2a_ffn_overlap_degree
        self.use_2dh = use_2dh

        if seeds is not None and seeds[1] is not None:
            torch.manual_seed(seeds[1])

        experts_type = experts.pop('type')
        if experts_type == 'custom':
            self.experts = cast(ModuleList, experts['module'])
        else:
            assert re.match(r'[a-zA-Z0-9\_]+', experts_type), "Expert type must only include digits, letters and underline characters."
            try:
                fused_experts = importlib.import_module(f'...experts.{experts_type}', __name__)
            except ModuleNotFoundError:
                raise Exception('Builtin expert type is not recognized: %s' % experts_type)

            if experts_type == 'ffn':
                assert 'fused_custom_fn' not in experts, "`fused_custom_fn` option for Tutel Moe-layer has been deprecated, please follows helloworld_from_scratch.py for custom construction instead."
                assert 'implicit_dropout_p' not in experts, "`implicit_dropout_p` option for Tutel Moe-layer has been deprecated, please use torch.nn.Dropout(p=implicit_dropout_p) on custom activation_fn (for fc1_dropout) and after Tutel Moe-layer (for fc2_dropout) instead."

            self.experts = fused_experts.ExpertModule(**experts)

        self.experts.update(self)

        if scan_expert_func is not None:
            for n, p in self.experts.named_parameters():
                scan_expert_func(n, p)
        for n, p in self.experts.named_parameters():
            setattr(p, '_tutel_expert', True)

        if isinstance(gate_type, str):
            assert re.match(r'^Top[0-9]+Gate$', gate_type), "Unrecognized gate_type: %s" % gate_type
            top_k = int(gate_type[3:-4])
            logging.warning(f"gate_type value `{gate_type}` in Tutel Moe-layer has been deprecated, please use gate_type = {{'type': 'top', 'k': {top_k}}} instead.")
            gate_type = {'type': 'top', 'k': top_k}

        if not isinstance(gate_type, list):
            gate_type = [gate_type]

        self.gates = []
        for gi, single_gate_type in enumerate(gate_type):
            gate_type = single_gate_type['type']
            single_gate_type.pop('type')
            assert re.match(r'[a-zA-Z0-9\_]+', gate_type), "Gate type must only include digits, letters and underline characters."

            if seeds is not None and seeds[0] is not None:
                torch.manual_seed(seeds[0] + gi)
            try:
                single_gate = importlib.import_module(f'...gates.{gate_type}', __name__)
            except ModuleNotFoundError:
                raise Exception("Unrecognized gate_type: %s" % gate_type)

            gate_module = single_gate.Gate(model_dim=self.model_dim, num_global_experts=self.num_global_experts, **single_gate_type)
            if not hasattr(gate_module, 'gate_noise'):
                gate_module.gate_noise = single_gate_type.get('gate_noise', 0.0)
            if not hasattr(gate_module, 'capacity_factor'):
                gate_module.capacity_factor = single_gate_type.get('capacity_factor', float(os.environ.get('CAP_FACTOR', 1.0)))

            self.gates += [gate_module]

        self.gates = ModuleList(self.gates)

        if seeds is not None and len(seeds) > 2 and seeds[2] is not None:
            torch.manual_seed(seeds[2])
            
        
        # split expert vars
        # self.scores_mat = None
        # self.scores_accelerate_mat = None
        # self.experts.split_experts()
        # self.count = 0
        # self.gate_interval = 3 # define static frquency to update gating matrix
        
        # self.expert_id=None # define which expert could be use now
        # self.scores=torch.zeros([self.num_local_experts],dtype=torch.float).to('cuda') # equal to routing matrix
        # self.change_rates = [[] for _ in range(self.num_local_experts)] # save the historical expert change rate
        # self.accum_threshold = 0.3 # when the sum of the historical expert change rate accumuating to this threshold, load operation will be triggered
        # self.max_history_len = 5 # define the maximum of the historical save length
        # self.update_frequency=3 # define dynamic frquency to update gating matrix
        # self.idea=idea # define the which gating strategy being used
        # self.expert_pool_indices = [0, 1]

    def extra_repr(self):
        return 'Top-K(s) = %s, Total-Experts = %d [managed by %d device(s)],' % (
            [f'k={x.top_k}, noise={x.gate_noise}' for x in self.gates],
            self.num_global_experts,
            self.world_size,
        )

    def get_parameter_iterator(self, param_type):
        if param_type == 'gate':
            return self.gates.named_parameters()
        elif param_type == 'local_experts':
            return self.experts.named_parameters()
        else:
            raise Exception("Specified parameter type is not recognized: %s. Valid `param_type` includes: gate, local_experts." % param_type)

    def expert_local(self, x, reserve_shape):
        y = self.experts(x.view(x.size(0), x.size(1), *reserve_shape), self)
        self.protected_shape = y.shape
        return y.reshape(y.size(0), y.size(1), -1)
    
    def calculate_gating_matrix(self,input):
        # calculate the gate matrix first
        original_shape, original_dtype  = input.shape, input.dtype
        # input.shape  torch.Size([1, 197, 384])
        x = input.reshape(-1, original_shape[-reserve_dims:].numel())
        # x.shape  torch.Size([197, 384])
        gctx = self.gates[gate_index]
        logits = gctx(x) # torch.Size([197, 4])
    
        if self.training and gctx.gate_noise > 0:
            logits_w_noise = logits + gctx.gate_noise * torch.randn_like(logits) / self.num_global_experts
        else:
            logits_w_noise = logits
        
        scores = F.softmax(logits_w_noise, dim=1)    # (197, expert_num)
        scores = torch.mean(scores, dim=0)
        top2_k_logits, top2_k_indices = scores.topk(2, dim=-1)
        return top2_k_indices
    
    def forward_fengqt(self, input: Tensor, gate_index=0, capacity_factor=None, top_k=None, a2a_ffn_overlap_degree=None, reserve_dims=1, inequivalent_tokens=True, adaptive_r=None):
        # idea 1: 取前两个最高的expert进行缓存
        if self.skip_moe:
            result_output = input
            result_output.l_aux = None
            return self.result_func(result_output) if self.result_func is not None else result_output
        if not self.change_frequency:
            if self.count>=3 and self.expert_id is None:
                top2_k_indices=self.calculate_gating_matrix(input)
                # idea1: 直接去计算gate的前两个expert的权重，然后load进gpu中
                if self.idea=='direct_load_top2':
                    self.experts.move_experts(top2_k_indices)
                    self.expert_id = top2_k_indices[0]
                    self.count += 1
                # idea2: 使用dist来判断当前与上一个matrix的相似度，小于一定阈值就判定大概率top1和top2有变化，提前load top2进去,否则就只load top1到gpu
                elif self.idea=='use_similarity':
                    dist = torch.norm(scores - self.scores, p=2)
                    similarity=1/(1+dist)
                    if similarity<0.9: # 相似度足够小，load进top2
                        self.experts.move_experts(top2_k_indices)
                    else: # 相似度不够，只load top1
                        self.experts.move_experts(top2_k_indices[0])
                    self.scores=scores
                elif self.idea=='expert_change_rate':
                    # idea2：每个epcho中，对于一维矩阵中的元素，需要去判断它的变化率是否累加到一个正数值，如果累加到后，就可以触发load top2
                    for i in range(self.num_experts):
                        change_rate = scores[i] - self.scores[i]
                        self.change_rates[i].append(change_rate)
                        if len(self.change_rates[i]) > self.max_history_len:
                            self.change_rates[i].pop(0)
                    top2=top2_k_indices[1]
                    # 对top2的expert的历史变化率进行求和
                    if sum(self.change_rates[top2]) > self.accum_threshold:
                        self.experts.move_experts(top2_k_indices)
                    else:
                        self.experts.move_experts(top2_k_indices[0])
                self.count=0
        else:
            if self.count>=self.update_frequency and self.expert_id is None:
                top2_k_indices=self.calculate_gating_matrix(input)
                # idea3: 动态改变更新的频率,更好地设定update的频率，尽量做到每次update就可以使得gates的top1能够更新，
                # 或者说具备提前load top1和2进gpu，如果是渐变切换domain大概率是一个expert先占据top2，进而提升到top1
                # 当然也有突变的情况，expert还没有提前load进去但是当前阶段就直接提升到top1，这属于极少部分情况
                if self.idea=='update frequency':
                    for i in range(self.num_experts):
                        change_rate = scores[i] - self.scores[i]
                        self.change_rates[i].append(change_rate)
                        if len(self.change_rates[i]) > self.max_history_len:
                            self.change_rates[i].pop(0)
                    top2=top2_k_indices[1]
                    # 对top2的expert的历史变化率进行求和,accum threshold是平均每一步的增长
                    if sum(self.change_rates[top2])/len(self.change_rates[top2]) > self.accum_threshold/self.numberOfLoadTop2:
                        self.experts.move_experts(top2_k_indices)
                        
                        # 间隔越大才能触发load top2，意味着后续的更新gate的频率可以减慢
                        if self.numberOfLoadTop2>3:
                            self.update_frequency=self.numberOfLoadTop2*2
                        elif self.numberOfLoadTop2<=3:
                            self.update_frequency=self.numberOfLoadTop2
                        self.numberOfLoadTop2=0
                    else:
                        self.experts.move_experts(top2_k_indices[0])
                        self.numberOfLoadTop2+=1 # 还没有触发load top2的操作
                self.count=0
        self.count+=1                    
        # expert forward
        return self.experts.split_forward(input, self.expert_id)

    def forward_kongr(self, input: Tensor, gate_index=0, capacity_factor=None, top_k=None, a2a_ffn_overlap_degree=None, reserve_dims=1, inequivalent_tokens=True, adaptive_r=None):
        if self.skip_moe:
            result_output = input
            result_output.l_aux = None
            return self.result_func(result_output) if self.result_func is not None else result_output


        # gate calculation
        original_shape, original_dtype  = input.shape, input.dtype
        # print(input.shape) # torch.Size([64, 197, 384])
        x = input.reshape(-1, original_shape[-reserve_dims:].numel())
        # print(x.shape) # torch.Size([12608, 384])
        gctx = self.gates[gate_index]
        logits = gctx(x) # torch.Size([12608, 4])
        # print(logits.shape)
        if self.training and gctx.gate_noise > 0:
            logits_w_noise = logits + gctx.gate_noise * torch.randn_like(logits) / self.num_global_experts
        else:
            logits_w_noise = logits
        scores = F.softmax(logits_w_noise, dim=1)    # (197, expert_num)
        scores = torch.mean(scores, dim=0)
        socres_sorted_indices = torch.argsort(scores, descending=True).tolist()
        # print(socres_sorted_indices)
        
        self.count += 1
        if self.count > self.gate_interval:
            # expert caching update (每隔self.gate_interval帧)
            # 把score第二大的expert caching起来
            second_best_expert_id = socres_sorted_indices[1]
            
            if socres_sorted_indices.index(self.expert_pool_indices[0]) < socres_sorted_indices.index(self.expert_pool_indices[1]):
                self.expert_pool_indices[1] = second_best_expert_id
                expert_id = self.expert_pool_indices[0]
            else:
                self.expert_pool_indices[0] = second_best_expert_id
                expert_id = self.expert_pool_indices[1]
                
            self.experts.move_experts(self.expert_pool_indices)
        
            self.count = 0
        else:
            # expert selection（每帧）
            # 从现有的两个expert中选出最好的那个使用
            if socres_sorted_indices.index(self.expert_pool_indices[0]) < socres_sorted_indices.index(self.expert_pool_indices[1]):
                expert_id = self.expert_pool_indices[0]
            else:
                expert_id = self.expert_pool_indices[1]
        

        # expert forward
        return self.experts.split_forward(input, expert_id)


    def forward(self, input: Tensor, gate_index=0, capacity_factor=None, top_k=None, a2a_ffn_overlap_degree=None, reserve_dims=1, inequivalent_tokens=True, adaptive_r=None):
        if self.skip_moe:
            result_output = input
            result_output.l_aux = None
            return self.result_func(result_output) if self.result_func is not None else result_output

        original_shape, original_dtype  = input.shape, input.dtype
        assert len(original_shape) >= 2, "Input data must be at least 2D tensor: (s)amples, .., (m)odel_dim"

        x = input.reshape(-1, original_shape[-reserve_dims:].numel())
        for p in self.experts.parameters():
            x = x.to(p.dtype)
            break
        gctx = self.gates[gate_index]
        if a2a_ffn_overlap_degree is not None:
            self.a2a_ffn_overlap_degree = a2a_ffn_overlap_degree
        a2a_ffn_overlap_degree = self.a2a_ffn_overlap_degree

        def routing():
            logits = gctx(x)

            if self.training and gctx.gate_noise > 0:
                logits_w_noise = logits + gctx.gate_noise * torch.randn_like(logits) / self.num_global_experts
            else:
                logits_w_noise = logits

            scores = F.softmax(logits_w_noise, dim=1)    # shape is (token_num, expert_num)
            if self.is_gshard_loss:
                _loss_fn = lambda gates, topk_ids: losses.gshard_loss(gates, topk_ids)
            else:
                _loss_fn = lambda gates, topk_ids: losses.load_importance_loss(
                    F.softmax(logits, dim=1), logits_w_noise.gather(index=topk_ids, dim=1),
                    self.num_global_experts, gctx.gate_noise)   
            return logits.dtype, extract_critical(scores,
                top_k = gctx.top_k if top_k is None else top_k,
                loss_fn = _loss_fn,
                capacity_factor = gctx.capacity_factor if capacity_factor is None else capacity_factor,
                batch_prioritized_routing = self.batch_prioritized_routing,
                normalize_gate = self.normalize_gate,
                group = self.group,
                alignment = self.sharded_count * a2a_ffn_overlap_degree,
                inequivalent_tokens = inequivalent_tokens,
            )


        if x.is_cuda:
            with torch.cuda.amp.autocast(enabled=False):
                logits_dtype, (crit, l_aux) = routing()
        else:
            logits_dtype, (crit, l_aux) = routing()

        y = fast_encode(x.to(logits_dtype), crit, self.is_postscore).to(x.dtype)

        if adaptive_r is not None:
            self.adaptive_degree = adaptive_r

        if self.adaptive_degree == 0:
            y = self.expert_local(y, original_shape[-reserve_dims:])
        else:
            if self.auto_parallel:
                self.use_model_parallel = (y.numel() * (self.sharded_count - 1) * 2 < sum([x.numel() for x in self.experts.parameters()]))

            if self.num_global_experts < self.world_size:
                if self.use_model_parallel:
                    y = y.repeat(1, self.adaptive_degree, 1).view(self.world_size, -1, y.size(2))
                else:
                    y = y.view(self.world_size, -1, y.size(2))

            if a2a_ffn_overlap_degree > 1 and y.is_cuda:
                def expert_fn(expert_input):
                    return self.expert_local(expert_input, original_shape[-reserve_dims:])
                y = a2a_ffn_overlap_forward(y, expert_fn=expert_fn, a2a_ffn_overlap_degree=a2a_ffn_overlap_degree, use_2dh=self.use_2dh, group=self.group)
            else:
                y = C.all_to_all(y, 1, 0, use_2dh=self.use_2dh, group=self.group)
                y = self.expert_local(y, original_shape[-reserve_dims:])
                y = C.all_to_all(y, 0, 1, use_2dh=self.use_2dh, group=self.group)

            if self.num_global_experts < self.world_size:
                if self.use_model_parallel:
                    y = torch.sum(y.view(self.num_global_experts, self.adaptive_degree, -1, y.size(2)), dim=1)
                else:
                    y = y.view(self.num_global_experts, -1, y.size(2))

        y = fast_decode(y.to(logits_dtype), crit, self.is_postscore)

        y = y.view(list(original_shape[:-reserve_dims]) + list(self.protected_shape[-reserve_dims:])).to(original_dtype)
        self.l_aux = y.l_aux = l_aux
        return self.result_func(y) if self.result_func is not None else y

moe_layer = MOELayer
