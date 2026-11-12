# SecSDQNEC-M

**Secure, middleware-enabled SDN and Double-DQN task offloading for IoT-UAV edge–cloud systems**

SecSDQNEC-M extends SDN-driven Double-DQN task offloading (SDQNEC) with a security middleware. IoT-UAVs send computation tasks to a middleware access point, and an SDN controller decides where each task is computed (UAV, middleware, Edge-1, Edge-2 or cloud) and over which OpenFlow path it travels. The framework keeps offloading efficient while defending against network-level and insider attacks:

| Attack | What the attacker does | How SecSDQNEC-M responds |
|---|---|---|
| DDoS on an SDN path | Floods one OpenFlow path with UDP traffic | Path autoencoder at the controller raises an alarm and the source is blocked |
| Compromised edge server | Accepts tasks, silently drops them and reports itself as idle | Beta-reputation trust falls and trust-aware action masking stops sending tasks there |
| Fake-task UAVs | Compromised UAVs flood oversized tasks | UAV autoencoder at the middleware raises an alarm and the UAV is blocked |

## Main components
- **SDN controller (OpenFlow 1.3):** Double-DQN offloading agent with trust-aware action masking, path anomaly detector, network monitor and response module.
- **Middleware access point:** task classifier, local anomaly detector (UAV and server autoencoders), beta-reputation trust engine, and UAV screening and blocking.
- **Neural networks** (Double DQN and autoencoders) implemented in plain C++ (`nn.h`), with no external machine-learning library.

## Repository structure
| Path | Purpose |
|---|---|
| `secsdqnec.cc` | ns-3 program: topology, middleware, SDN controller with DDQN, servers, attacks, detectors and metrics |
| `nn.h` | DQN/Double-DQN agent and autoencoder |
| `run_experiments.sh` | Main experiments: data collection, detector training, agent training, evaluation and plots |
| `run_extra.sh` | Conventional-IDS baseline, additional seeds, sensitivity analysis and detector comparison |
| `run_ablation2.sh` | Additional ablation: action masking vs. security reward, and masking threshold |
| `detector_ml.py` | Isolation Forest, One-Class SVM and LOF baselines on the same detection data |
| `ablation2_summary.py` | Tables and figures for `run_ablation2.sh` |
| `plots_ns3.py` | Figures and tables for the main experiments |
| `plots_extra.py` | Statistics (95% CI, Welch's t-test), figures and tables for the extra experiments |

## Requirements
- Linux (tested on Debian 13, ARM64), about 6 GB RAM and 10 GB disk
- [ns-3.40](https://www.nsnam.org/) with [OFSwitch13 5.2.3](https://github.com/ljerezchaves/ofswitch13)
- g++, CMake, Ninja, Git, Boost headers
- Python 3 with pandas, numpy, matplotlib, scipy and scikit-learn (plots and detector baselines only)

## Installation
```bash
# 1. Packages
sudo apt install -y g++ cmake ninja-build git patch libboost-dev \
     python3 python3-pandas python3-numpy python3-matplotlib python3-scipy python3-sklearn

# 2. ns-3.40 and OFSwitch13 5.2.3
git clone --branch ns-3.40 --depth 1 https://github.com/nsnam/ns-3-dev-git.git ~/ns-3.40
cd ~/ns-3.40
git clone --branch 5.2.3 --depth 1 https://github.com/ljerezchaves/ofswitch13.git contrib/ofswitch13
patch -p1 < contrib/ofswitch13/utils/ofswitch13-3_40.patch
patch -p1 < contrib/ofswitch13/utils/csma-full-duplex-3_40.patch

# 3. Configure and build ns-3 (20-60 min the first time)
./ns3 configure --build-profile=optimized --disable-examples --disable-tests --disable-python \
      --enable-modules="ofswitch13;wifi;mobility;internet;applications;flow-monitor;csma;point-to-point;energy"
./ns3 build

# 4. Add SecSDQNEC-M (all files go into one scratch folder)
mkdir -p scratch/secsdqnec
cp /path/to/SecSDQNEC-M/*.cc /path/to/SecSDQNEC-M/*.h /path/to/SecSDQNEC-M/*.sh /path/to/SecSDQNEC-M/*.py scratch/secsdqnec/
./ns3 build secsdqnec
```
On newer glibc versions the OFSwitch13 dependency may fail to compile because of a duplicate `strlcpy`. If so, change the line `#ifndef HAVE_STRLCPY` in `build/include/bofuss/util.h` to `#if !defined(HAVE_STRLCPY) && !defined(__cplusplus)` and build again.

## Running the experiments
All commands are run from the ns-3 folder. `JOBS` is the number of simulations run in parallel.
```bash
cd ~/ns-3.40

# quick test (about 1 minute)
./ns3 run "secsdqnec --mode=collect --policy=greedy --windows=2 --out=test --models=test"

# main experiments  -> results_v3/ and models_v3/
JOBS=4 bash scratch/secsdqnec/run_experiments.sh

# extra experiments -> results_extra/   (requires models_v3/)
JOBS=4 bash scratch/secsdqnec/run_extra.sh

# extra ablation    -> results_ablation2/ (requires models_v3/)
JOBS=4 bash scratch/secsdqnec/run_ablation2.sh
```
Each script writes CSV files, figures and a Markdown summary (`RESULTS.md`, `EXTRA_RESULTS.md`, `ABLATION2.md`) to its output folder. The graphs can be regenerated without re-running the simulations:
```bash
python3 scratch/secsdqnec/plots_ns3.py results_v3
python3 scratch/secsdqnec/plots_extra.py results_v3 results_extra
python3 scratch/secsdqnec/ablation2_summary.py results_ablation2
```

## Compared methods
| Group | Methods |
|---|---|
| Base-paper family | DQNEC, DDQNEC, SDQNEC |
| Conventional defence | SDQNEC + IDS (3-sigma threshold IDS with source blocking) |
| Heuristics | Greedy, Random |
| Ablations | without trust, without middleware, without security, without masking, without security reward |
| Proposed | SecSDQNEC-M |

Scenarios: no attack, DDoS on path, compromised edge, fake tasks, and mixed (all three attacks).

## Base paper
A. Ullah, I. Ullah, Q. M. u. Haq, S. Rubab, J. Baili and M. Attique Khan, "SDN-Driven Multi-Objective Task Offloading in IoT-Enabled UAVs in Edge-Cloud Computing Using Double DQN," *IEEE Transactions on Consumer Electronics*, vol. 71, no. 4, pp. 12124–12136, Nov. 2025, doi: 10.1109/TCE.2025.3597723.

## License
This code is released under the GNU General Public License v2.0 (see `LICENSE`), the same license as ns-3 and OFSwitch13, on which it is built.
