#!/bin/python
#-----------------------------------------------------------------------------
# File Name : neurostar.py
# Author: Emre Neftci
#
# Creation Date : Fri 22 Jun 2018 12:56:52 PM PDT
# Last Modified :
#
# Copyright : (c) UC Regents, Emre Neftci
# Licence : GPLv2
#-----------------------------------------------------------------------------
from graph import *


class Op(object):
    default_output = None
    # Properties attribute
    __props__ = ()

    # itypes and otypes attributes are
    # compulsory if make_node method is not defined.
    # They're the type of input and output respectively
    itypes = None
    otypes = None

    # Compulsory if itypes and otypes are not defined
    def make_node(self, *inputs):
        pass

    # Python implementation:
    def perform(self, node, inputs_storage, output_storage):
        pass

    # Other type of implementation
    # C implementation: [see theano web site for other functions]
    def c_code(self, node, inputs, outputs, sub):
        pass

    # optional:
    check_input = True

    def __init__(self, *args):
        pass

    def grad(self, inputs, g):
        pass

    def infer_shape(self, node, input_shapes):
        pass

    def get_params(self, node):
        if hasattr(self, 'params_type'):
            # and isinstance(self.params_type, theano.gof.ParamsType)
            wrapper = self.params_type
            if not all(hasattr(self, field) for field in wrapper.fields):
                # Let's print missing attributes for debugging.
                not_found = tuple(
                    field for field in wrapper.fields if not hasattr(self, field))
                raise AttributeError('%s: missing attributes %s for ParamsType.' % (
                    type(self).__name__, not_found))
            # ParamsType.get_params() will apply filtering to attributes.
            return self.params_type.get_params(self)
        raise Exception('get_params')

    def __call__(self, *inputs, **kwargs):
        """
        Optional: return some or all output[s] of `make_node`.

        It is called by code such as:

        .. python::

           x = tensor.matrix()

           # tensor.exp is an Op instance, calls
           # Op.__call__(self=<instance of exp>, inputs=(x,))
           y = tensor.exp(x)

        This class implements a convenience function (for graph-building) which
        uses `default_output`, but subclasses are free to override this function
        and ignore `default_output`.

        Parameters
        ----------
        inputs
            The Op's inputs, forwarded to the call to `make_node()`.
        kwargs
            Additional keyword arguments to be forwarded to
            `make_node()` *except* for optional argument `return_list` (which
            defaults to False). If `return_list` is True, then the returned
            value is always a list. Otherwise it is either a single Variable
            when the output of `make_node()` contains a single element, or this
            output (unchanged) when it contains multiple elements.

        """
        return_list = kwargs.pop('return_list', False)
        node = self.make_node(*inputs, **kwargs)

        if self.default_output is not None:
            rval = node.outputs[self.default_output]
            if return_list:
                rval = [rval]
            return rval
        else:
            if return_list:
                return list(node.outputs)
            elif len(node.outputs) == 1:
                return node.outputs[0]
            else:
                return node.outputs


class Add(Op):
    __props__ = ()

    def __str__(self):
        return "" + self.__class__.__name__

    def make_node(self, x, y):
        # Note: using x_.type() is dangerous, as it copies x's broadcasting
        # behaviour
        out = TensorVariable(x.type)
        return Apply(self, [x, y], [out])

    def perform(self, node, inputs, output_storage):
        x = inputs[0]
        y = inputs[1]
        z = output_storage[0]
        z[0] = x + y

    def infer_shape(self, node, i0_shapes):
        return i0_shapes

    def grad(self, inputs, output_grads):
        return [output_grads[0], output_grads[0]]


class Dot(Op):
    # See doc in instance of this Op or function after this class definition.
    __props__ = ()

    def __str__(self):
        return "Sparse" + self.__class__.__name__

    def infer_shape(self, node, shapes):
        xshp, yshp = shapes
        x, y = node.inputs
        if x.ndim == 2 and y.ndim == 2:
            return [(xshp[0], yshp[1])]
        if x.ndim == 1 and y.ndim == 2:
            return [(yshp[1],)]
        if x.ndim == 2 and y.ndim == 1:
            return [(xshp[0],)]
        if x.ndim == 1 and y.ndim == 1:
            return [()]
        raise NotImplementedError()

    def make_node(self, x, y):
        dtype_out = x.dtype

        # Sparse dot product should have at least one sparse variable
        # as input. If the other one is not sparse, it has to be converted
        # into a tensor.

#         if y.ndim == 1 or x.ndim == 1:
#             bz = (False,)
#         else:
#             bz = (False, False)
        return Apply(self, [x, y], [TensorVariable(dtype_out)])

    def perform(self, node, inputs, out):
        x, y = inputs
        out = out[0]

        rval = x * y

        rval = rval.toarray()

        out[0] = rval

    def grad(self, inputs, gout):
        (x, y) = inputs
        (gz,) = gout
        rval = []

        rval.append(Dot()(gz, y.T))
        rval.append(Dot()(x.T, gz))

        return rval


class SynapticDenseConnection(Dot):
    # This should derive from Dot. It is like Dot, but it has a neuromorphic
    # implementation and restricts the types of input and output variables.
    # The first input variable must be a Variable (without any inputs) and the
    # second is a SpikeListType
    pass

add = Add()
dot = Dot()


class Sigmoid(Op):
    __props__ = ()

    def __str__(self):
        return "ElementWise" + self.__class__.__name__

    def make_node(self, a):
        # Note: using x_.type() is dangerous, as it copies x's broadcasting
        # behavior
        out = TensorVariable(a.type)
        return Apply(self, [a], [out])

    def perform(self, node, inputs, output_storage):
        x = inputs[0]
        z = output_storage[0]
        z[0] = 1. / (1 + np.exp(-x))

    def infer_shape(self, node, i0_shapes):
        return i0_shapes

    def grad(self, inputs, output_grads):
        x = inputs[0]
        out = 1. / (1 + np.exp(-x))
        return [dot(output_grads[0], out * (1 - out))]

# TODO
# Implement gradient function
# Add spike stream data type
# Extract a simple compiler/scheduler/linker from theano for python implementation
# Understand how alternate implementations are used

sigmoid = Sigmoid()

if __name__ == "__main__":
    W = TensorVariable(type=np.float32)
    v = TensorVariable(type=np.float32)
    b = TensorVariable(type=np.float32)
    u1 = sigmoid(add(dot(W, v), b))
    u2 = sigmoid(add(dot(W, v), b))
    x = sigmoid(add(u1, u2))
