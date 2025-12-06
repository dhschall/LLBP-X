# LLBP-X: The Last-Level Branch Predictor Revisited Artifact

<p align="left">
    <a href="https://github.com/dhschall/LLBP-X/blob/main/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/badge/License-MIT-yellow.svg">
    </a>
    <a href="https://github.com/dhschall/LLBP-X/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/v/release/dhschall/LLBP-X">
    </a>
    <a href="https://doi.org/10.5281/zenodo.17807918"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.17807918.svg" alt="DOI"></a>
    <a href="https://github.com/dhschall/LLBP-X/actions/workflows/build-and-run.yml">
        <img alt="Build and test" src="https://github.com/dhschall/LLBP-X/actions/workflows/build-and-run.yml/badge.svg">
    </a>

</p>

> [!INFO]
> This repository based on the original [LLBP](https://github.com/dhschall/LLBP) repository and contains an extended version: LLBP-X.



The **Last-Level Branch Predictor (LLBP)** is a microarchitectural approach that improves branch prediction accuracy through additional high-capacity storage backing the baseline TAGE predictor. The key insight is that LLBP breaks branch predictor state into multiple program contexts which can be thought of as a call chain. Each context comprises only a small number of patterns and can be prefetched ahead of time. This enables LLBP to store a large number of patterns in a high-capacity structure and prefetch only the patterns for the upcoming contexts into a small, fast structure to overcome the long access latency of the high-capacity structure (LLBP).
LLBP was presented at [MICRO 2024](https://doi.org/10.1109/MICRO61859.2024.00042).

**LLBP-X** is an enhancement of the original LLBP design by introducing _dynamic context depth adaptation_, which yields a significantly better distribution of patterns for hard-to-predict branches, thereby reducing both pattern set contention and pattern duplication.
LLBP-X is presented at [HPCA 2026](https://conf.researchr.org/home/hpca-2026).

This repository contains the source code of the branch predictor models and infrastructure to evaluate LLBP-X's prediction accuracy.

The artifact consists of two main parts:
1. The branch predictor simulator based on the CBP framework to evaluate the prediction accuracy of LLBP-X and other branch predictor models on server traces. The aim of this framework is to provide a fast and easy way to evaluate different branch predictor configurations. It does *not* model the full CPU pipeline, only the branch predictor.
2. A gem5 implementation of LLBP-X to enable full-system simulations with gem5. (jump to [gem5 Simulation](#gem5-simulation) for details)


## CBP-based Branch Predictor Simulator

### Prerequisites

The infrastructure and following commands have been tested with the following system configuration:

* Ubuntu 22.04.2 LTS
* gcc 11.4.0
* cmake 3.22.1

> See the [CI pipeline](https://github.com/dhschall/LLBP-X/actions/workflows/build-and-run.yml) for other tested system configurations.



### Install Dependencies

```bash
# Install cmake
sudo apt install -y cmake libboost-all-dev build-essential pip parallel wget

# Python dependencies for plotting.
pip install -r analysis/requirements.txt

```


### Build the project

```bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Debug ..
cd ..

cmake --build ./build -j 8

```

### Server traces

The traces use to evaluate LLBP collected by running the server applications on gem5 in full-system mode. The OS of the disk image is Ubuntu 20.04 and the kernel version is 5.4.84. The traces are in the [ChampSim](https://github.com/ChampSim/ChampSim) format and contains both user and kernel space instructions. The traces are available on Zenodo at [10.5281/zenodo.13133242](https://doi.org/10.5281/zenodo.13133242).

The `download_traces.sh` script in the `scripts` folder will download all traces from Zenodo and stores them into the `traces` directory.:

```bash
./scripts/download_traces.sh
```


### Run the simulator

The simulator can be run with the following command and takes as inputs the trace file, the branch predictor model, the number of warmup instructions, and the number of simulation instructions.
The branch predictor model can be either `tage64kscl`, `tage512kscl`, `llbp`, `llbp-timing`, `llbpx` or `llbpx-timing`.
> The definition of the branch predictor models can be found in the `bpmodels/base_predictor.cc` file.

```bash
./build/predictor --model <predictor> -w <warmup instructions> -n <simulation instructions> <trace>
```

For convenience, the simulator contains a script to run the experiments on all evaluated benchmarks for a given branch predictor model (`./scripts/eval_benchmarks.sh <predictor>`).
The results in form of a stats file are stored in the `results` directory. Note, the simulator will print out some intermediate results after every 5M instructions which is useful to monitor the progress of the simulation.


### Plot results

The Jupyter notebook (`./analysis/mpki.ipynb`) can be used to parse the statistics file and plot the branch MPKI for different branch predictor models.

To reproduce a similar graph as in the paper (Figure 9), we provide a separate script (`./scripts/eval_all.sh`) that runs the experiments for all evaluated branch predictor models and benchmarks.

> *Note:* As we integrated the LLBP with ChampSim for the paper, the results might slightly differ from the presented numbers in the paper.

The script can be run as follows:

```bash
./scripts/eval_all.sh
```
Once the runs complete open they Jupyter notebook and hit run all cells.


## gem5 Simulation

For simulating LLBP-X in gem5, we provide an implementation of LLBP-X based on CBP model described above. For implementing LLBP-X in gem5, we tried to reuse as much of the existing TAGE-SC-L implementation as possible to reduce code redundancy and enable easy integration into gem5. However, due to differences in the code structure of gem5 and CBP that may lead to slight differences in the prediction accuracy of the models.

### Prerequisites
To simulate LLBP-X in gem5, you need to have gem5 cloned with all its prerequisites installed. The LLBP-X implementation has been tested with gem5 version v25.1.0.0. For further details on how to install gem5, please refer to the official [gem5 documentation](https://www.gem5.org/documentation/).


### Integrate and build LLBP-X in gem5
To integrate LLBP-X into gem5, copy the contents of the `gem5models` folder into the `src/cpu/pred` directory of your gem5 installation. This will add the LLBP-X implementation to gem5 and override some of the existing TAGE-SC-L files and the SConscript file to include the new files. Then rebuild gem5.
> Note you can use the provided `./scripts/setup_gem5.sh` script which will clone gem5, checkout a compatible commit, copy the LLBP-X files, and build gem5 for you.

```bash
cp -r gem5models/* <path-to-gem5>/src/cpu/pred/
cd <path-to-gem5>/
scons build/ARM/gem5.opt -j`nproc`
```

Alternatively, you can use the provided patch file `gem5_llbp_x.patch` to apply the changes to your gem5 installation:

```bash
cd <path-to-gem5>/
git apply ../scripts/gem5_llbp_x.patch
scons build/ARM/gem5.opt -j`nproc`
```

### Run LLBP-X in gem5

The `scripts` folder contains a simple configuration script (`se-llbp.py`) to run a hello world program in gem5's syscall emulation mode. You can run the simulation as follows:

```bash
./build/ARM/gem5.opt ./../scripts/se-llbp.py --bp=LLBPX
``` 
This will run the hello world program with LLBP-X as the branch predictor. Can change the `--bp` argument to `LLBPX`, `LLBP`, or `TSL64k` to simulate the other branch predictor models.

To quickly run all models and collect the branch mispredictions run:
```bash
./scripts/eval_all_gem5.sh
```
It will simulate all three models and print the branch mispredictions at the end.

## Citation
If you use our work, please cite paper:
```
@inproceedings{schall2026llbpx,
  title={The Last-Level Branch Predictor Revisited},
  author={Schall, David and  Ďuračková, Mária and Grot, Boris},
  booktitle={Proceedings of the 32nd IEEE International Symposium on High-Performance Computer Architecture (HPCA-32)},
  year={2026}
}
```

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact

David Schall - [GitHub](https://github.com/dhschall), [Website](https://dhschall.github.io/), [Mail](mailto:david.schall@tum.de)

## Acknowledgements
The authors thank the anonymous reviewers as well as the members of the Systems Research Group at the Technical University of Munich and the EASE Lab team at the University of Edinburgh for their valuable feedback on this work.
We are grateful to Caeden Whitaker, Mike Jennrich, and Matt Sinclair form the University of Wisconsin-Madison for helping with an initial gem5 implementation of LLBP, as well as Phillip Assmann from the Technical University of Munich for his significant effort in improving the model's correctness during his thesis work.