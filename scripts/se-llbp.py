# Copyright (c) 2025 Technical University of Munich
# All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# Install dependencies

"""
This script further shows an example of booting an ARM based full system Ubuntu
disk image. The script boots a full system Ubuntu image and starts the function container.
The function is invoked using a test client.

The workflow has two steps
1. Use the "setup" mode to boot the full system from scratch using the KVM core. The
   script will perform functional warming and then take a checkpoint of the system.
2. Use the "eval" mode to start from the previously taken checkpoint and perform
   the actual measurements using a detailed core model.

Usage
-----

```
scons build/<ALL|X86|ARM>/gem5.opt -j<NUM_CPUS>
./build/<ALL|ARM|gem5.opt fs-fdp.py
    --mode <setup/eval> --workload <benchmark>
    --kernel <path-to-vmlinux> --disk <path-to-disk-image>
    [--cpu <cpu-type>] [--fdp]
```

"""
import m5
from pathlib import Path
from typing import Iterator
import argparse


from m5.objects import (
    TAGE_SC_L_64KB,
    TAGE_SC_L_TAGE_64KB,
    ConditionalPredictor,
    BranchPredictor,
    LLBP,
    LLBPX,
    LLBP_TAGE_64KB,
)

from gem5.isas import ISA
from gem5.simulate.simulator import Simulator
from gem5.utils.requires import requires
from gem5.resources.resource import BinaryResource, obtain_resource
from gem5.components.boards.abstract_board import AbstractBoard
from gem5.components.boards.simple_board import SimpleBoard

from gem5.components.cachehierarchies.classic.private_l1_private_l2_cache_hierarchy import PrivateL1PrivateL2CacheHierarchy
from gem5.components.memory import DualChannelDDR4_2400
from gem5.components.processors.cpu_types import CPUTypes
from gem5.components.processors.simple_processor import SimpleProcessor
from gem5.simulate.simulator import Simulator


isa_choices = {
    "X86": ISA.X86,
    "Arm": ISA.ARM,
    "RiscV": ISA.RISCV,
}

workloads = {
    "Arm": "arm-hello64-static",
    "X86": "x86-hello64-static",
    "RISCV": "riscv-hello",
}

cpu_types = {
    "atomic": CPUTypes.ATOMIC,
    "timing": CPUTypes.TIMING,
    "o3": CPUTypes.O3,
}

parser = argparse.ArgumentParser(
    description="An example configuration script to run a hello world binary with LLBP-X"
)

# The only positional argument accepted is the benchmark name in this script.

parser.add_argument(
    "--isa",
    type=str,
    default="Arm",
    help="The ISA to simulate.",
    choices=isa_choices.keys(),
)

parser.add_argument(
    "--cpu-type",
    type=str,
    default="o3",
    help="The CPU model to use.",
    choices=cpu_types.keys(),
)
parser.add_argument(
    "--bp",
    type=str,
    default="TSL64k",
    help="BP model.",
)


args = parser.parse_args()


# This check ensures the gem5 binary is compiled to the correct ISA target.
# If not, an exception will be thrown.
requires(isa_required=isa_choices[args.isa])



# Here we setup the processor. For booting we take the KVM core and
# for the evaluation we can take ATOMIC, TIMING or O3

processor = SimpleProcessor(
    cpu_type=cpu_types[args.cpu_type],
    isa=isa_choices[args.isa],
    num_cores=1,
)


def predictor_map(predictor_name: str) -> ConditionalPredictor:
    match predictor_name:
        case "TSL64k":
            cbp = TAGE_SC_L_64KB(
                tage=TAGE_SC_L_TAGE_64KB(),
            )
        case "TSL512k":
            cbp = TAGE_SC_L_64KB(
                tage=TAGE_SC_L_TAGE_64KB(
                    logTagTableSize = 13
                ),
            )
        case "LLBP":
            cbp = LLBP(
                base=TAGE_SC_L_64KB(
                    tage=LLBP_TAGE_64KB(),
                ),
            )
        case "LLBPX":
            cbp = LLBPX(
                base=TAGE_SC_L_64KB(
                    tage=LLBP_TAGE_64KB(),
                ),
            )
        case _: raise ValueError(f"Unsupported BP: {args.bp}")
    return cbp




class BPU(BranchPredictor):
    instShiftAmt = 2 if args.isa == "Arm" else 1 if args.isa == "RISCV" else 0
    conditionalBranchPred = predictor_map(args.bp)

processor.cores[0].core.branchPred = BPU()


# The gem5 library simble board which can be used to run simple SE-mode
# simulations.
board = SimpleBoard(
    clk_freq="3GHz",
    processor=processor,
    memory=DualChannelDDR4_2400(size="1GiB"),
    cache_hierarchy=PrivateL1PrivateL2CacheHierarchy(
        l1i_size="32KiB",
        l1d_size="32KiB",
        l2_size="1MiB",
    ),
)

print(f"Running BP: {args.bp} with {args.isa}")


# Here we set the workload. In this case we want to run a simple "Hello World!"
# program compiled to the specified ISA. The `Resource` class will automatically
# download the binary from the gem5 Resources cloud bucket if it's not already
# present.
board.set_se_binary_workload(
    obtain_resource(workloads[args.isa])
)

# Lastly we run the simulation.
simulator = Simulator(board=board)
simulator.run()

print("Simulation done.")