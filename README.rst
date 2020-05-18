PennyLane PyQuest Plugin
#########################

This template repository provides the boilerplate and file structure required to easily create your
own PennyLane plugin.

See the `PennyLane Developer API documentation <https://pennylane.readthedocs.io/en/latest/API/overview.html>`_
for more details on developing a PennyLane plugin.

`PyQuest-cffi <https://pyquest.readthedocs.io>`_ is a Python library that connects to the high-performance mixed
state simulator Quest.

`PennyLane <https://pennylane.readthedocs.io>`_ is a machine learning library for optimization
and automatic differentiation of hybrid quantum-classical computations.

Installation
============

PennyLane-PyQuest requires both PennyLane and PyQuest-cffi. It can be installed via ``pip``:

.. code-block:: bash

    $ pip install git+https://www.github.com/johannesjmeyer/pennylane-pyquest


Getting started
===============

Once PennyLane-PyQuest is installed, the provided PyQuest-cffi devices can be accessed straight
away in PennyLane.

You can instantiate these devices for PennyLane as follows:

.. code-block:: python

    import pennylane as qml
    dev1 = qml.device('pyquest.pure', wires=2)
    dev2 = qml.device('pyquest.mixed', wires=2)

These devices can then be used just like other devices for the definition and evaluation of
QNodes within PennyLane. For more details, see the
`plugin usage guide <https://plugin-name.readthedocs.io/en/latest/usage.html>`_ and refer
to the PennyLane documentation.


Contributing
============

We welcome contributions - simply fork the PennyLane-PyQuest repository, and then make a
`pull request <https://help.github.com/articles/about-pull-requests/>`_ containing your contribution.
All contributers to PennyLane-SF will be listed as authors on the releases.

We also encourage bug reports, suggestions for new features and enhancements, and even links to cool
projects or applications built on PennyLane and PyQuest-cffi.


Authors
=======

Johannes Jakob Meyer.

If you are doing research using PennyLane, please cite our papers:

    Bergholm et al.,
    *PennyLane: Automatic differentiation of hybrid quantum-classical computations.* 2018.
    `arXiv:1811.04968 <https://arxiv.org/abs/1811.04968>`_


Support
=======

- **Source Code:** https://github.com/johannesjmeyer/pennylane-pyquest
- **Issue Tracker:** https://github.com/johannesjmeyer/pennylane-pyquest/issues

If you are having issues, please let us know by posting the issue on our GitHub issue tracker.


License
=======

PennyLane-PyQuest is **free** and **open source**, released under the Apache License, Version 2.0.
