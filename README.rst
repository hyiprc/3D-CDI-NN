3D-CDI-NN - Neural network for coherent diffraction image inversion
===================================================================

3D-CDI-NN is a deep neural network model plus automatic
differentiation developed for retrieving phase information from 3D
coherent diffraction images. The model is implemented using Tensorflow
and the training dataset is generated using physics-based atomistic
simulations. Custom codes are written to handle the resampling of
diffraction images to oversampling ratios appropriate for the neural
network model.

.. image:: ./docs/_static/output_compare.png
    :alt: Comparing output from phase retrieval, 3D-CDI-NN prediction, and AD refined 3D-CDI-NN prediction

Reference:

"Rapid 3D nanoscale coherent imaging via physics-aware deep learning"
Applied Physics Reviews 8, 021407 (2021)
<https://doi.org/10.1063/5.0031486>


----

3D-CDI-NN is free software/open source, and is distributed under the
BSD license. It contains third-party code, see below for the license
information on third-party code:

+--------------+------------------------------------------------------------------------+
| Python       | <https://docs.python.org/3/license.html>                               |
+--------------+------------------------------------------------------------------------+
| NumPy        | <https://github.com/numpy/numpy/blob/master/LICENSE.txt>               |
+--------------+------------------------------------------------------------------------+
| SciPy        | <https://scipy.org/scipylib/license.html>                              |
+--------------+------------------------------------------------------------------------+
| scikit-image | <https://github.com/scikit-image/scikit-image/blob/master/LICENSE.txt> |
+--------------+------------------------------------------------------------------------+
| TensorFlow   | <https://github.com/tensorflow/tensorflow/blob/master/LICENSE>         |
+--------------+------------------------------------------------------------------------+
