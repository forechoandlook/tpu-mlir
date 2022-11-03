#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

import argparse
import pymlir
import numpy as np
from pathlib import Path
from calibration.mix_precision import MixPrecSearcher

if __name__ == '__main__':
    print("SOPHGO Toolchain {}".format(pymlir.module().version))
    # yapf: disable
    parser = argparse.ArgumentParser(description="Generate quantization table")
    parser.add_argument('mlir_file', help='fp32 mlir file')
    parser.add_argument('--dataset', help='dataset path for mix precision searching')
    parser.add_argument("--data_list", help="specify a file with inputs's absolute path for mix precision searching")
    parser.add_argument('--input_num', type=int, default=10,
                        help='num of inputs for quantization searching')
    parser.add_argument('--calibration_table', required=True,
                        help='calibration table generated by calibration or tune tool')
    parser.add_argument('--chip', required=True, type=str,
                        choices=['bm1684x', 'bm1684', 'cv183x', 'cv182x', 'cv181x'],
                        help='chip platform name')
    parser.add_argument('--num_layers', type=int, default=10,
                        help='number of layers for mix quantization')
    parser.add_argument('--loss_table', default='full_loss_table.txt',
                        help="output all loss of layers if each layer is quantized to f16")
    parser.add_argument('-o', '--quantize_table', required=True,
                        help='output searched bf16 layer table')
    # yapf: enable
    args = parser.parse_args()
    searcher = MixPrecSearcher(args)
    sort_bf16_layers = searcher.run()

    with open(args.loss_table, "w") as f:
        for idx, layer in enumerate(sort_bf16_layers):
            loss_msg = "No.{:<4}: Layer: {:<50}\t\tLoss: {}".format(idx, layer[0], layer[1])
            f.write("{}\n".format(loss_msg))
            print(loss_msg)

    with open(args.quantize_table, "w") as f:
        sort_bf16_layers = sort_bf16_layers[:args.num_layers]
        for i in sort_bf16_layers:
            f.write("{}\n".format(i[0]))
    print("Output bf16 table to {}".format(args.quantize_table))