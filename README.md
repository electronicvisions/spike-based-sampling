# sbs

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3675211.svg)](https://doi.org/10.5281/zenodo.3675211)
[![Build Status](https://jenkins.bioai.eu/buildStatus/icon?job=bld_gerrit-model-nmsampling-sbs)](https://jenkins.bioai.eu/job/bld_gerrit-model-nmsampling-sbs/)

Spike-based-sampling, `sbs`, implements stochastic [LIF
sampling](https://arxiv.org/abs/1601.00909). It takes care of calibrating LIF
neurons for given neuron/input parameters and allows the evaluation of
arbitrary Boltzmann-distributions in static networks.

## Quickstart

If you want to jump right in, just see the [tutorial](examples/tutorial.py).

## Introduction

`sbs` clearly separates the abstract concept of stochastic LIF neurons and
Boltzmann machine (BM) from network communication code. Its two main conceptual
buildings blocks are [`LIFsampler`](sbs/samplers.py) as well as
[BoltzmannMachine](sbs/network.py).

### `LIFsampler`

The `LIFsampler` is described by a neuron model, its corresponding parameters
and a back- ground source configuration (typically one excitatory and one
inhibitory Poisson source with set rate and synaptic weight). Given this
configuration, it is able to automatically calibrate itself to find the weight
conversion factors as well as the membrane potential at which the activation
function is exactly 0.5. For re-usability, a complete `LIFsampler`'s
configuration is saved as JSON-file to allow for easy inspection. After
calibrating once, the `LIFsampler` can be created from file.

### `BoltzmannMachine`

The `BoltzmannMachine` on the other hand implements – as the name suggests – a
[BM](https://en.wikipedia.org/wiki/Boltzmann_machine) of inter- connected
heterogeneously configured `LIFsampler`s. The user can specify either theoretical
or biological weight/bias configuration. The `BoltzmannMachine` takes care of
automatically translating between the two, taking into account each
`LIFsampler`s possibly unique calibration data. The network can then be run to
gather spike samples from the corresponding biological network, from which a
sample-based approximation of the underlying probability distributions is
automatically computed. Renewing synapses with custom
[Tsodyks-Markram](http://www.scholarpedia.org/article/Short-term_synaptic_plasticity)
parameters are also possible. For smaller networks, theoretical distributions
can be computed as well as the Kullback-Leibler divergence (DKL) computed
between the two. Demanding computations regarding probability distributions or
state computations from spike trains are implemented using
[Cython](https://cython.org), a library that converts type-annotated Python to
C that is then pre-compiled and loaded as shared library during execution.

### `PyNN` on-demand

An important feature of sbs is that no `PyNN`-specific code is run until the
user explicitly requests it, e.g., via each objects `create()`-routine. This is
necessary due to `PyNN`'s inherent "statefulness". Even though a call to
`sim.end()`/`sim.setup(…)` is supposed to wipe the currently used simulator's
network state (according to the API-specification) it is not always the case.
This way, tasks such as computing theoretical probability distributions or
performing weight conversions of already calibrated `LIFsampler`s can be
accomplished without involving `PyNN` at all. Furthermore, tasks that involve
`PyNN` – e.g., calibration or the gathering of spikes given a BM-configuration
– can be offloaded into subprocesses in a fully transparent manner, allowing
for more than one of such tasks to be performed in a single run.

### On-demand computing via descriptors

Another feature introduced by `sbs` is the ability to compute attributes of a
class on demand. Each attribute computes the values of other attributes it
depends on automatically. This is accomplished by decorating each attribute by
`@DependsOn(...)` where the arguments are the names of the corresponding
dependencies. The whole class object then needs to be decorated with
`@HasDependencies`. Values are stored and reused once computed and only
discarded when one of the dependencies is changed.
For example, accessing the sampled Boltzmann probability distribution of the
Boltzmann-object for the first time after automatically computes the
distribution from the recorded spike data. Each subsequent access does not lead
to a new computation, the probability distribution is stored. If, however, new
spike data is gathered, the old distribution is discarded and recomputed once
needed. The same relationship exists between the theoretical distribution and
the weights set for the network.
The attribute is a simple function accepting up to one argument. Akin to the
properties concept of Python itself, it has to implement both get and set
operations. If the optional argument is `None` (get-operation), the function has
to compute its current value from its dependencies and return it. If the
optional argument is defined (set-operation), the function has the ability to
transform the value before returning what should be stored.

## Install

The installation process is your plain old `setuptools`-workflow.

Global install:
```shell
python setup.py install
```

Local install:
```shell
python setup.py install --user
```

Install to specific `<folder>`:
```shell
python setup.py install --prefix=<folder>
```

## Requirements
* Python 2 (upgrade to Python 3 happening soon™)
* PyNN 0.8
* For [NEST](https://github.com/nest/nest-simulator), the speed-up improvements
  are only tested with versions up to `2.14.0`!


## Authors

`sbs` was foolishly written by:

* Oliver Breitwieser, Kirchhoff Institute for Physics, Heidelberg University

The following people contributed to the code:

* Andreas Baumbach, Kirchhoff Institute for Physics, Heidelberg University
* Agnes Korcsak-Gorzo, Forschungszentrum Jülich
* Johann Klähn, Kirchhoff Institute for Physics, Heidelberg University
* Max Brixner, Kirchhoff Institute for Physics, Heidelberg University

`sbs` is based on a proof-of-concept prototype developed for the original [LIF
sampling](https://arxiv.org/abs/1601.00909) paper by Mihai Petrovici, who
guided the development of `sbs` in terms of theory.

## License 
`sbs` is licensed under LGPLv3. See [LICENSE](LICENSE) for more information.

