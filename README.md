<h1 align="center">TPSC for the Kane-Mele-Hubbard model</h1>
<p align="center">
A Python code for the Two-Particle-Self-Consistent (TPSC) approach on the Kane-Mele-Hubbard model that can calculate the spin Hall conductivity for finite temperatures as described in [<a href="https://arxiv.org/abs/2307.15652">Lessnich et al. (2023)</a>].
</p>

## Table of contents

- [Get started](#get-started)
- [Licence and Citation](#licence-and-citation)



## Get started

You can clone this directory via

```bash
git clone https://github.com/Dominik-Lessnich/TPSC_KMH.git
```

The code makes use of the [<a href="https://github.com/SpM-lab/sparse-ir"> sparse-ir </a>] library (version 0.92.1) that can be installed with pip via

```bash
pip install sparse-ir[xprec]
```

Further it makes use of the standard Python libraries [<a href="https://numpy.org/"> Numpy </a>] (version 1.23.1), [<a href="https://matplotlib.org/"> Matplotlib </a>](version 3.5.2), [<a href="https://scipy.org/"> Scipy </a>] (version 1.8.1).

The file Example_KaneMeleHubbard.py contains an example file with input parameters that can be modified. 
I recomend executing the cells one by one in Visual Studio Code to explore that file.
Alternatively, it can be run in a terminal via

```bash
python3 Example_KaneMeleHubbard.py
```

With the given parameters the code runs for about a minute on a laptop.
The code solves the TPSC self-consistency equations and further calculates spin correlation lengths, spin Hall conductivity (with and without vertex corrections) and the band gap renormalization.
It also plots the TPSC self-energy, Green's function and susceptibilities in k-space and in matrix form for the individual orbital and spin indices.



## Licence and Citation

This software is released under the GNU General Public License v3.0.

If you find this code usefull and use parts of it in your research please consider citing the following paper [<a href="https://arxiv.org/abs/2307.15652">Lessnich et al., arXiv:2307.15652 (2023)</a>].

For questions please contact 

- [dominik.lessnich@usherbrooke.ca]
- [andre-marie.tremblay@usherbrooke.ca]