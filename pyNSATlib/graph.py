#!/bin/python
#-----------------------------------------------------------------------------
# File Name : graph.py
# Author: Theano
#
# Creation Date : Wed 27 Jun 2018 01:44:18 PM PDT
# Last Modified : 
#----------------------------------------------------------------------------- 

"""
Node classes (`Apply`, `Variable`) and expression graph algorithms.
"""
from __future__ import absolute_import, print_function, division

from collections import deque
import contextlib
from copy import copy
from itertools import count

import warnings

from six import string_types, integer_types, iteritems
from six import iteritems, integer_types, string_types, with_metaclass
import numpy as np

from theano.misc.ordered_set import OrderedSet

__docformat__ = "restructuredtext en"

# Lazy imports to avoid circular dependencies.
is_same_graph_with_merge = None
equal_computations = None

NoParams = object()

def asarray(data, dtype):
    return data


class MetaObject(type):
    def __new__(cls, name, bases, dct):
        props = dct.get('__props__', None)
        if props is not None:
            if not isinstance(props, tuple):
                raise TypeError("__props__ has to be a tuple")
            if not all(isinstance(p, string_types) for p in props):
                raise TypeError("elements of __props__ have to be strings")

            def _props(self):
                """
                Tuple of properties of all attributes
                """
                return tuple(getattr(self, a) for a in props)
            dct['_props'] = _props

            def _props_dict(self):
                """This return a dict of all ``__props__`` key-> value.

                This is useful in optimization to swap op that should have the
                same props. This help detect error that the new op have at
                least all the original props.

                """
                return dict([(a, getattr(self, a))
                             for a in props])
            dct['_props_dict'] = _props_dict

            if '__hash__' not in dct:
                def __hash__(self):
                    return hash((type(self),
                                 tuple(getattr(self, a) for a in props)))
                dct['__hash__'] = __hash__

            if '__eq__' not in dct:
                def __eq__(self, other):
                    return (type(self) == type(other) and
                            tuple(getattr(self, a) for a in props) ==
                            tuple(getattr(other, a) for a in props))
                dct['__eq__'] = __eq__

            if '__str__' not in dct:
                if len(props) == 0:
                    def __str__(self):
                        return "%s" % (self.__class__.__name__,)
                else:
                    def __str__(self):
                        return "%s{%s}" % (
                            self.__class__.__name__,
                            ", ".join("%s=%r" % (p, getattr(self, p))
                                      for p in props))
                dct['__str__'] = __str__

        return type.__new__(cls, name, bases, dct)

class object2(with_metaclass(MetaObject, object)):
    __slots__ = []

    def __ne__(self, other):
        return not self == other


class scratchpad(object):
    def clear(self):
        self.__dict__.clear()

    def __update__(self, other):
        self.__dict__.update(other.__dict__)
        return self

    def __str__(self):
        return "scratchpad" + str(self.__dict__)

    def __repr__(self):
        return "scratchpad" + str(self.__dict__)

    def info(self):
        print("<scratchpad instance at %i>" % id(self))
        for k, v in iteritems(self.__dict__):
            print("  %s: %s" % (k, v))

def add_tag_trace(thing, user_line=None):
    """
    Add tag.trace to an node or variable.

    The argument is returned after being affected (inplace).

    Parameters
    ----------
    thing
        The object where we add .tag.trace.
    user_line
        The max number of user line to keep.

    Notes
    -----
    We also use config.traceback.limit for the maximum number of stack level
    we look.

    """
    if user_line is None:
        user_line = 0 
        #user_line = config.traceback.limit

    if user_line == -1:
        user_line = None
    skips = ["theano/tensor/", "theano\\tensor\\",
             "theano/compile/", "theano\\compile\\",
             "theano/gof/", "theano\\gof\\",
             "theano/scalar/basic.py", "theano\\scalar\\basic.py",
             "theano/sandbox/", "theano\\sandbox\\",
             "theano/scan_module/", "theano\\scan_module\\",
             "theano/sparse/", "theano\\sparse\\",
             "theano/typed_list/", "theano\\typed_list\\"]

    #if config.traceback.compile_limit > 0:
    #    skips = []
    skips=[]

    tr = simple_extract_stack(limit=user_line, skips=skips)
    # Different python version use different sementic for
    # limit. python 2.7 include the call to extrack_stack. The -1 get
    # rid of it.

    if tr:
        thing.tag.trace = [tr]
    else:
        thing.tag.trace = tr
    return thing


class Node(object2):
    """
    A Node in a theano graph.

    Graphs contain two kinds of Nodes -- Variable and Apply.
    Edges in the graph are not explicitly represented.
    Instead each Node keeps track of its parents via
    Variable.owner / Apply.inputs and its children
    via Variable.clients / Apply.outputs.

    """

    def get_parents(self):
        """
        Return a list of the parents of this node.
        Should return a copy--i.e., modifying the return
        value should not modify the graph structure.

        """
        raise NotImplementedError()


class Apply(Node):
    """
    An :term:`Apply` instance is a node in an expression graph which represents
    the application of an `Op` to some input `Variable` nodes, producing some
    output `Variable` nodes.

    This class is typically instantiated by an Op's make_node() function, which
    is typically called by that Op's __call__() function.

    An Apply instance serves as a simple structure with three important
    attributes:

    - :literal:`inputs` :  a list of `Variable` nodes that represent the
      arguments of the expression,

    - :literal:`outputs` : a list of `Variable` nodes that represent the
      variable of the expression, and

    - :literal:`op` : an `Op` instance that determines the nature of the
      expression being applied.

    The driver `compile.function` uses Apply's inputs attribute together with
    Variable's owner attribute to search the expression graph and determine
    which inputs are necessary to compute the function's outputs.

    A `Linker` uses the Apply instance's `op` field to compute the variables.

    Comparing with the Python language, an `Apply` instance is theano's version
    of a function call (or expression instance) whereas `Op` is theano's version
    of a function definition.

    Parameters
    ----------
    op : `Op` instance
    inputs : list of Variable instances
    outputs : list of Variable instances

    Notes
    -----
    The owner field of each output in the outputs list will be set to self.

    If an output element has an owner that is neither None nor self, then a
    ValueError exception will be raised.

    """

    def __init__(self, op, inputs, outputs):
        self.op = op
        self.inputs = []
        self.tag = scratchpad()

        if not isinstance(inputs, (list, tuple)):
            raise TypeError("The inputs of an Apply must be a list or tuple")

        if not isinstance(outputs, (list, tuple)):
            raise TypeError("The output of an Apply must be a list or tuple")

        # filter inputs to make sure each element is a Variable
        for input in inputs:
            if isinstance(input, Variable):
                self.inputs.append(input)
            else:
                raise TypeError("The 'inputs' argument to Apply must contain Variable instances, not %s" % input)
        self.outputs = []
        # filter outputs to make sure each element is a Variable
        for i, output in enumerate(outputs):
            if isinstance(output, Variable):
                if output.owner is None:
                    output.owner = self
                    output.index = i
                elif output.owner is not self or output.index != i:
                    raise ValueError("All output variables passed to Apply must belong to it.")
                self.outputs.append(output)
            else:
                raise TypeError("The 'outputs' argument to Apply must contain Variable instances with no owner, not %s" % output)

    def run_params(self):
        """
        Returns the params for the node, or NoParams if no params is set.

        """
        try:
            return self.op.get_params(self)
        except Exception():
            return NoParams

    def __getstate__(self):
        d = self.__dict__
        # ufunc don't pickle/unpickle well
        if hasattr(self.tag, 'ufunc'):
            d = copy(self.__dict__)
            t = d["tag"]
            del t.ufunc
            d["tag"] = t
        return d

    def default_output(self):
        """
        Returns the default output for this node.

        Returns
        -------
        Variable instance
            An element of self.outputs, typically self.outputs[0].

        Notes
        -----
        May raise AttributeError self.op.default_output is out of range, or if
        there are multiple outputs and self.op.default_output does not exist.

        """
        do = getattr(self.op, 'default_output', None)
        if do is None:
            if len(self.outputs) == 1:
                return self.outputs[0]
            else:
                raise AttributeError(
                    "%s.default_output should be an output index." % self.op)
        elif not isinstance(do, integer_types):
            raise AttributeError("%s.default_output should be an int or long" %
                                 self.op)
        elif do < 0 or do >= len(self.outputs):
            raise AttributeError("%s.default_output is out of range." %
                                 self.op)
        return self.outputs[do]

    out = property(default_output,
                   doc="alias for self.default_output()")
    """
    Alias for self.default_output().

    """

    def __str__(self):
        return op_as_string(self.inputs, self)

    def __repr__(self):
        return str(self)

    def __asapply__(self):
        return self

    def clone(self):
        """
        Duplicate this Apply instance with inputs = self.inputs.

        Returns
        -------
        object
            A new Apply instance (or subclass instance) with new outputs.

        Notes
        -----
        Tags are copied from self to the returned instance.

        """
        cp = self.__class__(self.op, self.inputs,
                            [output.clone() for output in self.outputs])
        cp.tag = copy(self.tag)
        return cp

    def clone_with_new_inputs(self, inputs, strict=True):
        """
        Duplicate this Apply instance in a new graph.

        Parameters
        ----------
        inputs
            List of Variable instances to use as inputs.
        strict : bool
            If True, the type fields of all the inputs must be equal
            to the current ones (or compatible, for instance Tensor /
            GpuArray of the same dtype and broadcastable patterns,
            in which case they will be converted into current Type), and
            returned outputs are guaranteed to have the same types as
            self.outputs.  If False, then there's no guarantee that the
            clone's outputs will have the same types as self.outputs,
            and cloning may not even be possible (it depends on the Op).

        Returns
        -------
        object
            An Apply instance with the same op but different outputs.

        """
        assert isinstance(inputs, (list, tuple))
        remake_node = False
        new_inputs = inputs[:]
        for i, (curr, new) in enumerate(zip(self.inputs, new_inputs)):
            if not curr.type == new.type:
                if strict:
                    # If compatible, casts new into curr.type
                    new_inputs[i] = curr.type.filter_variable(new)
                else:
                    remake_node = True
        if remake_node:
            new_node = self.op.make_node(*new_inputs)
            new_node.tag = copy(self.tag).__update__(new_node.tag)
        else:
            new_node = self.clone()
            new_node.inputs = new_inputs
        return new_node

    def get_parents(self):
        return list(self.inputs)

    # convenience properties
    nin = property(lambda self: len(self.inputs), doc='same as len(self.inputs)')
    """
    Property: Number of inputs.

    """
    nout = property(lambda self: len(self.outputs), doc='same as len(self.outputs)')
    """
    Property: Number of outputs.

    """
    params_type = property(lambda self: self.op.params_type, doc='type to use for the params')


class Variable(Node):
    """
    A :term:`Variable` is a node in an expression graph that represents a
    variable.

    The inputs and outputs of every `Apply` (theano.gof.Apply) are `Variable`
    instances. The input and output arguments to create a `function` are also
    `Variable` instances. A `Variable` is like a strongly-typed variable in
    some other languages; each `Variable` contains a reference to a `Type`
    instance that defines the kind of value the `Variable` can take in a
    computation.

    A `Variable` is a container for four important attributes:

    - :literal:`type` a `Type` instance defining the kind of value this
      `Variable` can have,

    - :literal:`owner` either None (for graph roots) or the `Apply` instance
      of which `self` is an output,

    - :literal:`index` the integer such that :literal:`owner.outputs[index] is
      this_variable` (ignored if `owner` is None),

    - :literal:`name` a string to use in pretty-printing and debugging.

    There are a few kinds of Variables to be aware of: A Variable which is the
    output of a symbolic computation has a reference to the Apply instance to
    which it belongs (property: owner) and the position of itself in the owner's
    output list (property: index).

    - `Variable` (this base type) is typically the output of a symbolic
      computation.

    - `Constant` (a subclass) which adds a default and un-replaceable
      :literal:`value`, and requires that owner is None.

    - `TensorVariable` subclass of Variable that represents a numpy.ndarray
       object.

    - `TensorSharedVariable` Shared version of TensorVariable.

    - `SparseVariable` subclass of Variable that represents
      a scipy.sparse.{csc,csr}_matrix object.

    - `GpuArrayVariable` subclass of Variable that represents our object on
      the GPU that is a subset of numpy.ndarray.

    - `RandomVariable`.

    A Variable which is the output of a symbolic computation will have an owner
    not equal to None.

    Using the Variables' owner field and the Apply nodes' inputs fields, one can
    navigate a graph from an output all the way to the inputs. The opposite
    direction is not possible until a FunctionGraph has annotated the Variables
    with the clients field, ie, before the compilation process has begun a
    Variable does not know which Apply nodes take it as input.

    Parameters
    ----------
    type : a Type instance
        The type governs the kind of data that can be associated with this
        variable.
    owner : None or Apply instance
        The Apply instance which computes the value for this variable.
    index : None or int
        The position of this Variable in owner.outputs.
    name : None or str
        A string for pretty-printing and debugging.

    Examples
    --------

    .. code-block:: python

        import theano
        from theano import tensor

        a = tensor.constant(1.5)        # declare a symbolic constant
        b = tensor.fscalar()            # declare a symbolic floating-point scalar

        c = a + b                       # create a simple expression

        f = theano.function([b], [c])   # this works because a has a value associated with it already

        assert 4.0 == f(2.5)            # bind 2.5 to an internal copy of b and evaluate an internal c

        theano.function([a], [c])       # compilation error because b (required by c) is undefined

        theano.function([a,b], [c])     # compilation error because a is constant, it can't be an input

        d = tensor.value(1.5)           # create a value similar to the constant 'a'
        e = d + b
        theano.function([d,b], [e])     # this works.  d's default value of 1.5 is ignored.

    The python variables :literal:`a,b,c` all refer to instances of type
    `Variable`. The `Variable` referred to by `a` is also an instance of
    `Constant`.

    `compile.function` uses each `Apply` instance's `inputs` attribute together
    with each Variable's `owner` field to determine which inputs are necessary
    to compute the function's outputs.

    """

    # __slots__ = ['type', 'owner', 'index', 'name']
    __count__ = count(0)

    def __init__(self, type, owner=None, index=None, name=None):
        super(Variable, self).__init__()

        self.tag = scratchpad()
        self.type = type
        if owner is not None and not isinstance(owner, Apply):
            raise TypeError("owner must be an Apply instance", owner)
        self.owner = owner
        if index is not None and not isinstance(index, integer_types):
            raise TypeError("index must be an int", index)
        self.index = index
        if name is not None and not isinstance(name, string_types):
            raise TypeError("name must be a string", name)
        self.name = name
        self.auto_name = 'auto_' + str(next(self.__count__))

        Variable.notify_construction_observers(self)

    def __str__(self):
        """Return a str representation of the Variable.

        """
        if self.name is not None:
            return self.name
        if self.owner is not None:
            op = self.owner.op
            if self.index == op.default_output:
                return str(self.owner.op) + ".out"
            else:
                return str(self.owner.op) + "." + str(self.index)
        else:
            return "<%s>" % str(self.type)

    def __repr__(self, firstPass=True):
        """Return a repr of the Variable.

        Return a printable name or description of the Variable. If
        config.print_test_value is True it will also print the test_value if
        any.
        """
        to_print = [str(self)]
        if firstPass:
            try:
                to_print.append(self.__repr_test_value__())
            except AttributeError:
                pass
        return '\n'.join(to_print)

    def clone(self):
        """
        Return a new Variable like self.

        Returns
        -------
        Variable instance
            A new Variable instance (or subclass instance) with no owner or
            index.

        Notes
        -----
        Tags are copied to the returned instance.

        Name is copied to the returned instance.

        """
        # return copy(self)
        cp = self.__class__(self.type, None, None, self.name)
        cp.tag = copy(self.tag)
        return cp

    def __lt__(self, other):
        raise NotImplementedError('Subclasses of Variable must provide __lt__',
                                  self.__class__.__name__)

    def __le__(self, other):
        raise NotImplementedError('Subclasses of Variable must provide __le__',
                                  self.__class__.__name__)

    def __gt__(self, other):
        raise NotImplementedError('Subclasses of Variable must provide __gt__',
                                  self.__class__.__name__)

    def __ge__(self, other):
        raise NotImplementedError('Subclasses of Variable must provide __ge__',
                                  self.__class__.__name__)

    def get_parents(self):
        if self.owner is not None:
            return [self.owner]
        return []

    def eval(self, inputs_to_values=None):
        """
        Evaluates this variable.

        Parameters
        ----------
        inputs_to_values
            A dictionary mapping theano Variables to values.

        Examples
        --------

        >>> import numpy as np
        >>> import theano.tensor as T
        >>> x = T.dscalar('x')
        >>> y = T.dscalar('y')
        >>> z = x + y
        >>> np.allclose(z.eval({x : 16.3, y : 12.1}), 28.4)
        True

        We passed :func:`eval` a dictionary mapping symbolic theano
        variables to the values to substitute for them, and it returned
        the numerical value of the expression.

        Notes
        -----

        `eval` will be slow the first time you call it on a variable --
        it needs to call :func:`function` to compile the expression behind
        the scenes. Subsequent calls to :func:`eval` on that same variable
        will be fast, because the variable caches the compiled function.

        This way of computing has more overhead than a normal Theano
        function, so don't use it too much in real scripts.
        """

        if inputs_to_values is None:
            inputs_to_values = {}

        if not hasattr(self, '_fn_cache'):
            self._fn_cache = dict()

        inputs = tuple(sorted(inputs_to_values.keys(), key=id))
        if inputs not in self._fn_cache:
            self._fn_cache[inputs] = theano.function(inputs, self)
        args = [inputs_to_values[param] for param in inputs]

        rval = self._fn_cache[inputs](*args)

        return rval

    def __getstate__(self):
        d = self.__dict__.copy()
        d.pop("_fn_cache", None)
        return d

    #  refer to doc in nodes_constructed.
    construction_observers = []

    @classmethod
    def append_construction_observer(cls, observer):
        cls.construction_observers.append(observer)

    @classmethod
    def remove_construction_observer(cls, observer):
        cls.construction_observers.remove(observer)

    @classmethod
    def notify_construction_observers(cls, instance):
        for observer in cls.construction_observers:
            observer(instance)


class Constant(Variable):
    """
    A :term:`Constant` is a `Variable` with a `value` field that cannot be
    changed at runtime.

    Constant nodes make eligible numerous optimizations: constant inlining in
    C code, constant folding, etc.

    Notes
    -----
    The data field is filtered by what is provided in the constructor for the
    Constant's type field.

    WRITEME

    """

    # __slots__ = ['data']
    def __init__(self, type, data, name=None):
        Variable.__init__(self, type, None, None, name)
        self.data = type.filter(data)
        tils.add_tag_trace(self)

    def equals(self, other):
        # this does what __eq__ should do, but Variable and Apply should always be hashable by id
        return isinstance(other, Constant) and self.signature() == other.signature()

    def signature(self):
        return (self.type, self.data)

    def merge_signature(self):
        return self.signature()

    def __str__(self):
        if self.name is not None:
            return self.name
        else:
            name = str(self.data)
            if len(name) > 20:
                name = name[:10] + '...' + name[-10:]
            return 'Constant{%s}' % name

    def clone(self):
        """
        We clone this object, but we don't clone the data to lower memory
        requirement. We suppose that the data will never change.

        """
        cp = self.__class__(self.type, self.data, self.name)
        cp.tag = copy(self.tag)
        return cp

    def __set_owner(self, value):
        """
        WRITEME

        Raises
        ------
        ValueError
            If `value` is not `None`.

        """
        if value is not None:
            raise ValueError("Constant instances cannot have an owner.")

    owner = property(lambda self: None, __set_owner)
    value = property(lambda self: self.data, doc='read-only data access method')

    # index is not defined, because the `owner` attribute must necessarily be None


def stack_search(start, expand, mode='bfs', build_inv=False):
    """
    Search through a graph, either breadth- or depth-first.

    Parameters
    ----------
    start : deque
        Search from these nodes.
    expand : callable
        When we get to a node, add expand(node) to the list of nodes to visit.
        This function should return a list, or None.
    mode : string
        'bfs' or 'dfs' for breath first search or depth first search.

    Returns
    -------
    list of `Variable` or `Apply` instances (depends on `expend`)
        The list of nodes in order of traversal.

    Notes
    -----
    A node will appear at most once in the return value, even if it
    appears multiple times in the start parameter.

    :postcondition: every element of start is transferred to the returned list.
    :postcondition: start is empty.

    """

    if mode not in ('bfs', 'dfs'):
        raise ValueError('mode should be bfs or dfs', mode)
    rval_set = set()
    rval_list = list()
    if mode == 'bfs':
        start_pop = start.popleft
    else:
        start_pop = start.pop
    expand_inv = {}  # var: clients
    while start:
        l = start_pop()
        if id(l) not in rval_set:
            rval_list.append(l)
            rval_set.add(id(l))
            expand_l = expand(l)
            if expand_l:
                if build_inv:
                    for r in expand_l:
                        expand_inv.setdefault(r, []).append(l)
                start.extend(expand_l)
    assert len(rval_list) == len(rval_set)
    if build_inv:
        return rval_list, expand_inv
    return rval_list


def ancestors(variable_list, blockers=None):
    """
    Return the variables that contribute to those in variable_list (inclusive).

    Parameters
    ----------
    variable_list : list of `Variable` instances
        Output `Variable` instances from which to search backward through
        owners.

    Returns
    -------
    list of `Variable` instances
        All input nodes, in the order found by a left-recursive depth-first
        search started at the nodes in `variable_list`.

    """
    def expand(r):
        if r.owner and (not blockers or r not in blockers):
            return reversed(r.owner.inputs)
    dfs_variables = stack_search(deque(variable_list), expand, 'dfs')
    return dfs_variables


def inputs(variable_list, blockers=None):
    """
    Return the inputs required to compute the given Variables.

    Parameters
    ----------
    variable_list : list of `Variable` instances
        Output `Variable` instances from which to search backward through
        owners.

    Returns
    -------
    list of `Variable` instances
        Input nodes with no owner, in the order found by a left-recursive
        depth-first search started at the nodes in `variable_list`.

    """
    vlist = ancestors(variable_list, blockers)
    rval = [r for r in vlist if r.owner is None]
    return rval


def variables_and_orphans(i, o):
    """
    Extract list of variables between i and o nodes via
    dfs traversal and chooses the orphans among them

    Parameters
    ----------
    i : list
         Input variables.
    o : list
         Output variables.

    """
    def expand(r):
        if r.owner and r not in i:
            l = list(r.owner.inputs) + list(r.owner.outputs)
            l.reverse()
            return l
    variables = stack_search(deque(o), expand, 'dfs')
    orphans = [r for r in variables if r.owner is None and r not in i]
    return variables, orphans


def ops(i, o):
    """
    Set of Ops contained within the subgraph between i and o

    Parameters
    ----------
    i : list
        Input variables.
    o : list
        Output variables.

    Returns
    -------
    object
        The set of ops that are contained within the subgraph that lies
        between i and o, including the owners of the variables in o and
        intermediary ops between i and o, but not the owners of the variables
        in i.

    """
    ops = set()
    variables, orphans = variables_and_orphans(i, o)
    for r in variables:
        if r not in i and r not in orphans:
            if r.owner is not None:
                ops.add(r.owner)
    return ops


def variables(i, o):
    """
    Extracts list of variables within input and output nodes via dfs travesal

    Parameters
    ----------
    i : list
        Input variables.
    o : list
        Output variables.

    Returns
    -------
    object
        The set of Variables that are involved in the subgraph that lies
        between i and o. This includes i, o, orphans(i, o) and all values of
        all intermediary steps from i to o.

    """
    return variables_and_orphans(i, o)[0]


def orphans(i, o):
    """
    Extracts list of variables within input and output nodes
    via dfs travesal and returns the orphans among them

    Parameters
    ----------
    i : list
        Input Variables.
    o : list
        Output Variables.

    Returns
    -------
    object
        The set of Variables which one or more Variables in o depend on but are
        neither in i nor in the subgraph that lies between i and o.

    Examples
    --------
    orphans([x], [(x+y).out]) => [y]

    """
    return variables_and_orphans(i, o)[1]


def clone(i, o, copy_inputs=True, copy_orphans=None):
    """Copies the subgraph contained between i and o.

    Parameters
    ----------
    i : list
        Input Variables.
    o : list
        Output Variables.
    copy_inputs : bool
        If True, the inputs will be copied (defaults to True).
    copy_orphans:
        When None, use the copy_inputs value,
        When True, new orphans nodes are created.
        When False, original orphans nodes are reused in the new graph.

    Returns
    -------
    object
        The inputs and outputs of that copy.

    Note
    ----

    A constant, if in the ``i`` list is not an orpha. So it will be
    copied depending of the ``copy_inputs`` parameter. Otherwise it
    will be copied depending of the ``copy_orphans`` parameter.

    """
    if copy_orphans is None:
        copy_orphans = copy_inputs
    equiv = clone_get_equiv(i, o, copy_inputs, copy_orphans)
    return [equiv[input] for input in i], [equiv[output] for output in o]


def clone_get_equiv(inputs, outputs, copy_inputs=True, copy_orphans=True,
                    memo=None):
    """
    Return a dictionary that maps from Variable and Apply nodes in the
    original graph to a new node (a clone) in a new graph.

    This function works by recursively cloning inputs... rebuilding a directed
    graph from the inputs up to eventually building new outputs.

    Parameters
    ----------
    inputs : a list of Variables
    outputs : a list of Variables
    copy_inputs : bool
        True means to create the cloned graph from new input
        nodes (the bottom of a feed-upward graph).
        False means to clone a graph that is rooted at the original input
        nodes.
    copy_orphans:
        When True, new constant nodes are created. When False, original
        constant nodes are reused in the new graph.
    memo : None or dict
        Optionally start with a partly-filled dictionary for the return value.
        If a dictionary is passed, this function will work in-place on that
        dictionary and return it.

    """
    if memo is None:
        memo = {}

    # clone the inputs if necessary
    for input in inputs:
        if copy_inputs:
            cpy = input.clone()
            cpy.owner = None
            cpy.index = None
            memo.setdefault(input, cpy)
        else:
            memo.setdefault(input, input)

    # go through the inputs -> outputs graph cloning as we go
    for apply in io_toposort(inputs, outputs):
        for input in apply.inputs:
            if input not in memo:
                if copy_orphans:
                    cpy = input.clone()
                    memo[input] = cpy
                else:
                    memo[input] = input

        new_apply = apply.clone_with_new_inputs([memo[i] for i in apply.inputs])
        memo.setdefault(apply, new_apply)
        for output, new_output in zip(apply.outputs, new_apply.outputs):
            memo.setdefault(output, new_output)

    # finish up by cloning any remaining outputs (it can happen)
    for output in outputs:
        if output not in memo:
            memo[output] = output.clone()

    return memo


def general_toposort(outputs, deps, debug_print=False,
                     compute_deps_cache=None, deps_cache=None,
                     clients=None):
    """
    WRITEME

    Parameters
    ----------
    deps
        A python function that takes a node as input and returns its dependence.
    compute_deps_cache : optional
        If provided deps_cache should also be provided. This is a function like
        deps, but that also cache its results in a dict passed as deps_cache.
    deps_cache : dict
        Must be used with compute_deps_cache.
    clients : dict
        If a dict is passed it will be filled with a mapping of node
        -> clients for each node in the subgraph.

    Notes
    -----
        deps(i) should behave like a pure function (no funny business with
        internal state).

        deps(i) will be cached by this function (to be fast).

        The order of the return value list is determined by the order of nodes
        returned by the deps() function.

        deps should be provided or can be None and the caller provides
        compute_deps_cache and deps_cache. The second option removes a Python
        function call, and allows for more specialized code, so it can be
        faster.

    """
    if compute_deps_cache is None:
        deps_cache = {}

        def compute_deps_cache(io):
            if io not in deps_cache:
                d = deps(io)
                if d:
                    if not isinstance(d, (list, OrderedSet)):
                        raise TypeError(
                            "Non-deterministic collections here make"
                            " toposort non-deterministic.")
                    deps_cache[io] = list(d)
                else:
                    deps_cache[io] = d
                return d
            else:
                return deps_cache[io]
    assert deps_cache is not None

    assert isinstance(outputs, (tuple, list, deque))

    reachable, _clients = stack_search(deque(outputs), compute_deps_cache,
                                       'dfs', True)
    if clients is not None:
        clients.update(_clients)
    sources = deque([r for r in reachable if not deps_cache.get(r, None)])

    rset = set()
    rlist = []
    while sources:
        node = sources.popleft()
        if node not in rset:
            rlist.append(node)
            rset.add(node)
            for client in _clients.get(node, []):
                d = [a for a in deps_cache[client] if a is not node]
                deps_cache[client] = d
                if not d:
                    sources.append(client)

    if len(rlist) != len(reachable):
        if debug_print:
            print('')
            print(reachable)
            print(rlist)
        raise ValueError('graph contains cycles')

    return rlist


def io_toposort(inputs, outputs, orderings=None, clients=None):
    """
    Perform topological sort from input and output nodes

    Parameters
    ----------
    inputs : list or tuple of Variable instances
    outputs : list or tuple of Apply instances
    orderings : dict
        Key: Apply instance. Value: list of Apply instance.
        It is important that the value be a container with a deterministic
        iteration order. No sets allowed!
    clients : dict
        If a dict is provided it will be filled with mappings of
        node->clients for each node in the subgraph that is sorted

    """
    if not orderings and clients is None:  # ordering can be None or empty dict
        # Specialized function that is faster when more then ~10 nodes
        # when no ordering.

        # Do a new stack implementation with the vm algo.
        # This will change the order returned.
        computed = set(inputs)
        todo = [o.owner for o in reversed(outputs) if o.owner]
        order = []
        while todo:
            cur = todo.pop()
            # We suppose that all outputs are always computed
            if cur.outputs[0] in computed:
                continue
            if all([i in computed or i.owner is None for i in cur.inputs]):
                computed.update(cur.outputs)
                order.append(cur)
            else:
                todo.append(cur)
                todo.extend(i.owner for i in cur.inputs if i.owner)
        return order

    compute_deps = None
    compute_deps_cache = None
    iset = set(inputs)
    deps_cache = {}

    if not orderings:  # ordering can be None or empty dict
        # Specialized function that is faster when no ordering.
        # Also include the cache in the function itself for speed up.

        def compute_deps_cache(obj):
            if obj in deps_cache:
                return deps_cache[obj]
            rval = []
            if obj not in iset:
                if isinstance(obj, Variable):
                    if obj.owner:
                        rval = [obj.owner]
                elif isinstance(obj, Apply):
                    rval = list(obj.inputs)
                if rval:
                    if not isinstance(rval, (list, OrderedSet)):
                        raise TypeError(
                            "Non-deterministic collections here make"
                            " toposort non-deterministic.")
                    deps_cache[obj] = list(rval)
                else:
                    deps_cache[obj] = rval
            else:
                deps_cache[obj] = rval
            return rval
    else:

        # the inputs are used only here in the function that decides what
        # 'predecessors' to explore
        def compute_deps(obj):
            rval = []
            if obj not in iset:
                if isinstance(obj, Variable):
                    if obj.owner:
                        rval = [obj.owner]
                elif isinstance(obj, Apply):
                    rval = list(obj.inputs)
                rval.extend(orderings.get(obj, []))
            else:
                assert not orderings.get(obj, None)
            return rval

    topo = general_toposort(outputs, deps=compute_deps,
                            compute_deps_cache=compute_deps_cache,
                            deps_cache=deps_cache, clients=clients)
    return [o for o in topo if isinstance(o, Apply)]


default_leaf_formatter = str


def default_node_formatter(op, argstrings):
    return "%s(%s)" % (op.op, ", ".join(argstrings))


def io_connection_pattern(inputs, outputs):
    """
    Returns the connection pattern of a subgraph defined by given
    inputs and outputs.

    """
    inner_nodes = io_toposort(inputs, outputs)

    # Initialize 'connect_pattern_by_var' by establishing each input as
    # connected only to itself
    connect_pattern_by_var = {}
    nb_inputs = len(inputs)

    for i in range(nb_inputs):
        input = inputs[i]
        inp_connection_pattern = [i == j for j in range(nb_inputs)]
        connect_pattern_by_var[input] = inp_connection_pattern

    # Iterate through the nodes used to produce the outputs from the
    # inputs and, for every node, infer their connection pattern to
    # every input from the connection patterns of their parents.
    for n in inner_nodes:

        # Get the connection pattern of the inner node's op. If the op
        # does not define a connection_pattern method, assume that
        # every node output is connected to every node input
        try:
            op_connection_pattern = n.op.connection_pattern(n)
        except AttributeError:
            op_connection_pattern = ([[True] * len(n.outputs)] *
                                     len(n.inputs))

        # For every output of the inner node, figure out which inputs it
        # is connected to by combining the connection pattern of the inner
        # node and the connection patterns of the inner node's inputs.
        for out_idx in range(len(n.outputs)):
            out = n.outputs[out_idx]
            out_connection_pattern = [False] * nb_inputs

            for inp_idx in range(len(n.inputs)):
                inp = n.inputs[inp_idx]

                if inp in connect_pattern_by_var:
                    inp_connection_pattern = connect_pattern_by_var[inp]

                    # If the node output is connected to the node input, it
                    # means it is connected to every inner input that the
                    # node inputs is connected to
                    if op_connection_pattern[inp_idx][out_idx]:
                        out_connection_pattern = [out_connection_pattern[i] or
                                                  inp_connection_pattern[i]
                                                  for i in range(nb_inputs)]

            # Store the connection pattern of the node output
            connect_pattern_by_var[out] = out_connection_pattern

    # Obtain the global connection pattern by combining the
    # connnection patterns of the individual outputs
    global_connection_pattern = [[] for o in range(len(inputs))]
    for out in outputs:
        out_connection_pattern = connect_pattern_by_var.get(out)
        if out_connection_pattern is None:
            # the output is completely isolated from inputs
            out_connection_pattern = [False] * len(inputs)
        for i in range(len(inputs)):
            global_connection_pattern[i].append(out_connection_pattern[i])

    return global_connection_pattern


def op_as_string(i, op,
                 leaf_formatter=default_leaf_formatter,
                 node_formatter=default_node_formatter):
    """
    Op to return a string representation of the subgraph
    between i and o
    """
    strs = as_string(i, op.inputs, leaf_formatter, node_formatter)
    return node_formatter(op, strs)


def as_string(i, o,
              leaf_formatter=default_leaf_formatter,
              node_formatter=default_node_formatter):
    """
    Returns a string representation of the subgraph between i and o

    Parameters
    ----------
    i : list
        Input `Variable` s.
    o : list
        Output `Variable` s.
    leaf_formatter : callable
        Takes a `Variable`  and returns a string to describe it.
    node_formatter : callable
        Takes an `Op`  and the list of strings corresponding to its arguments
        and returns a string to describe it.

    Returns
    -------
    str
        Returns a string representation of the subgraph between i and o. If the
        same op is used by several other ops, the first occurrence will be
        marked as :literal:`*n -> description` and all subsequent occurrences
        will be marked as :literal:`*n`, where n is an id number (ids are
        attributed in an unspecified order and only exist for viewing
        convenience).

    """
    i = set(i)

    orph = orphans(i, o)

    multi = set()
    seen = set()
    for output in o:
        op = output.owner
        if op in seen:
            multi.add(op)
        else:
            seen.add(op)
    for op in ops(i, o):
        for input in op.inputs:
            op2 = input.owner
            if input in i or input in orph or op2 is None:
                continue
            if op2 in seen:
                multi.add(op2)
            else:
                seen.add(input.owner)
    multi = [x for x in multi]
    done = set()

    def multi_index(x):
        return multi.index(x) + 1

    def describe(r):
        if r.owner is not None and r not in i and r not in orph:
            op = r.owner
            idx = op.outputs.index(r)
            if len(op.outputs) == 1:
                idxs = ""
            else:
                idxs = "::%i" % idx
            if op in done:
                return "*%i%s" % (multi_index(op), idxs)
            else:
                done.add(op)
                s = node_formatter(op, [describe(input) for input in op.inputs])
                if op in multi:
                    return "*%i -> %s" % (multi_index(op), s)
                else:
                    return s
        else:
            return leaf_formatter(r)

    return [describe(output) for output in o]


def view_roots(r):
    """
    Utility function that returns the leaves of a search through
    consecutive view_map()s.

    WRITEME

    """
    owner = r.owner
    if owner is not None:
        try:
            view_map = owner.op.view_map
            view_map = dict((owner.outputs[o], i)
                            for o, i in iteritems(view_map))
        except AttributeError:
            return [r]
        if r in view_map:
            answer = []
            for i in view_map[r]:
                answer += view_roots(owner.inputs[i])
            return answer
        else:
            return [r]
    else:
        return [r]


def list_of_nodes(inputs, outputs):
    """
    Return the apply nodes of the graph between inputs and outputs.

    """
    return stack_search(
        deque([o.owner for o in outputs]),
        lambda o: [inp.owner for inp in o.inputs
                   if inp.owner and
                   not any(i in inp.owner.outputs for i in inputs)])


def is_in_ancestors(l_node, f_node):
    r"""
    Goes up in the graph and returns True if the apply node f_node is found.

    Use a stack implementation as the vm algo.
    We suppose all nodes are not lazy
    (i.e. for IfElse we suppose all inputs are computed)
    """
    computed = set()
    todo = [l_node]
    while todo:
        cur = todo.pop()
        if cur.outputs[0] in computed:
            continue
        if all([i in computed or i.owner is None for i in cur.inputs]):
            computed.update(cur.outputs)
            if cur is f_node:
                return True
        else:
            todo.append(cur)
            todo.extend(i.owner for i in cur.inputs if i.owner)
    return False


@contextlib.contextmanager
def nodes_constructed():
    """
    A contextmanager that is used in inherit_stack_trace and keeps track
    of all the newly created varaible nodes inside an optimization. A list
    of new_nodes is instantiated but will be filled in a lazy manner (when
    Variable.notify_construction_observers is called).


    `observer` is the entity that updates the new_nodes list.
    construction_observers is a list inside Variable class and contains
    a list of observer functions. The observer functions inside
    construction_observers are only called when a variable node is
    instantiated (where Variable.notify_construction_observers is called).
    When the observer function is called, a new variable node is added to
    the new_nodes list.


    Parameters
    ----------
    new_nodes
        A list of all the variable nodes that are created inside the optimization.

    yields
        new_nodes list.
    """
    new_nodes = []

    def observer(node):
        new_nodes.append(node)
    Variable.append_construction_observer(observer)
    yield new_nodes
    Variable.remove_construction_observer(observer)

class PureType(object):
    """
    Interface specification for variable type instances.

    A :term:`Type` instance is mainly responsible for two things:

    - creating `Variable` instances (conventionally, `__call__` does this), and

    - filtering a value assigned to a `Variable` so that the value
      conforms to restrictions imposed by the type (also known as
      casting, this is done by `filter`).

    """

    # the type that will be created by call to make_variable.
    Variable = Variable

    # the type that will be created by call to make_constant
    Constant = Constant

    def filter(self, data, strict=False, allow_downcast=None):
        """
        Required: Return data or an appropriately wrapped/converted data.

        Subclass implementation should raise a TypeError exception if
        the data is not of an acceptable type.

        If strict is True, the data returned must be the same as the
        data passed as an argument. If it is False, and allow_downcast
        is True, filter may cast it to an appropriate type. If
        allow_downcast is False, filter may only upcast it, not lose
        precision. If allow_downcast is None (default), the behaviour can be
        Type-dependent, but for now it means only Python floats can be
        downcasted, and only to floatX scalars.

        Raises
        ------
        MethodNotDefined
            Subclass doesn't implement this function.

        """
        raise MethodNotDefined("filter", type(self), self.__class__.__name__)

    # If filter_inplace is defined, it will be called instead of
    # filter() This is to allow reusing the old allocated memory. As
    # of this writing this is used only when we transfer new data to a
    # shared variable on the gpu.

    # def filter_inplace(value, storage, strict=False, allow_downcast=None)

    def filter_variable(self, other, allow_convert=True):
        """
        Convert a symbolic variable into this Type, if compatible.

        For the moment, the only Types compatible with one another are
        TensorType and GpuArrayType, provided they have the same
        number of dimensions, same broadcasting pattern, and same
        dtype.

        If Types are not compatible, a TypeError should be raised.

        """
        if not isinstance(other, graph.Variable):
            # The value is not a Variable: we cast it into
            # a Constant of the appropriate Type.
            other = self.Constant(type=self, data=other)

        if other.type != self and allow_convert:
            other2 = self.convert_variable(other)
            if other2 is not None:
                return other2

        if other.type != self:
            raise TypeError(
                'Cannot convert Type %(othertype)s '
                '(of Variable %(other)s) into Type %(self)s. '
                'You can try to manually convert %(other)s into a %(self)s.'
                % dict(othertype=other.type, other=other, self=self))
        return other

    def convert_variable(self, var):
        """
        Patch variable so that its type will match self, if possible.

        If the variable can't be converted, this should return None.

        The conversion can only happen if the following implication is
        true for all possible `val`.

          self.is_valid_value(val) => var.type.is_valid_value(val)

        For the majority of types this means that you can only have
        non-broadcastable dimensions become broadcastable and not the
        inverse.

        The default is to not convert anything which is always safe.

        """
        return None

    def is_valid_value(self, a):
        """
        Required: Return True for any python object `a` that would be a
        legal value for a Variable of this Type.

        """
        try:
            self.filter(a, strict=True)
            return True
        except (TypeError, ValueError):
            return False

    def value_validity_msg(self, a):
        """
        Optional: Return a message explaining the output of
        is_valid_value.

        """
        return "none"

    def make_variable(self, name=None):
        """
        Return a new `Variable` instance of Type `self`.

        Parameters
        ----------
        name : None or str
            A pretty string for printing and debugging.

        """
        return self.Variable(self, name=name)

    def make_constant(self, value, name=None):
        return self.Constant(type=self, data=value, name=name)

    def __call__(self, name=None):
        """
        Return a new `Variable` instance of Type `self`.

        Parameters
        ----------
        name : None or str
            A pretty string for printing and debugging.

        """
        return utils.add_tag_trace(self.make_variable(name))

    def values_eq(self, a, b):
        """
        Return True if a and b can be considered exactly equal.

        a and b are assumed to be valid values of this Type.

        """
        return a == b

    def values_eq_approx(self, a, b):
        """
        Return True if a and b can be considered approximately equal.

        This function is used by theano debugging tools to decide
        whether two values are equivalent, admitting a certain amount
        of numerical instability. For example, for floating-point
        numbers this function should be an approximate comparison.

        By default, this does an exact comparison.

        Parameters
        ----------
        a
            A potential value for a Variable of this Type.

        b
            A potential value for a Variable of this Type.

        Returns
        -------
        bool

        """
        return self.values_eq(a, b)

#    def get_shape_info(self, obj):
        """
        Optional function. See TensorType().get_shape_info for definition.

        """

#    def get_size(self, shape_info):
        """
        Optional function. See TensorType().get_size for definition.

        """

class TensorType(object2, PureType):
    """
    Symbolic `Type` representing a numpy.ndarray value.

    Initialize self.dtype and self.broadcastable.

    Parameters
    ----------
    dtype: str
        Corresponding to numpy dtype (e.g., 'int64')
        The value (ndarray) associated to a `Variable` of this `Type` will
        have this dtype.
    broadcastable: tuple, list, or array of boolean values
        This argument serves two purposes. First, the True elements of this
        list indicate the dimensions where the shape of an associated value
        must be 1. Secondly, the length of this list is the number of
        dimensions that an associated value must have. See
        doc:`broadcasting` for an explanation of how this list is used.
    name : str
        Optional name for this type.

    """
    context_name = 'cpu'
    filter_checks_isfinite = False
    """
    When this is True, strict filtering rejects data containing NaN or
    Inf entries. (Used in `DebugMode`)
    """

    def __init__(self, dtype, broadcastable, name=None):
        self.dtype = str(dtype)
        if self.dtype == 'floatX':
            self.dtype = config.floatX
        # broadcastable is immutable, and all elements are either
        # True or False
        self.broadcastable = tuple(bool(b) for b in broadcastable)
        self.dtype_specs()  # error checking is done there
        self.name = name
        self.numpy_dtype = np.dtype(self.dtype)


    def clone(self, dtype=None, broadcastable=None):
        """
        Return a copy of the type optionally with a new dtype or
        broadcastable pattern.

        """
        if dtype is None:
            dtype = self.dtype
        if broadcastable is None:
            broadcastable = self.broadcastable
        return self.__class__(dtype, broadcastable, name=self.name)

    def filter(self, data, strict=False, allow_downcast=None):
        """
        Convert `data` to something which can be associated to a
        `TensorVariable`.

        This function is not meant to be called in user code. It is for
        `Linker` instances to use when running a compiled graph.

        """
        # Explicit error message when one accidentally uses a Variable as
        # input (typical mistake, especially with shared variables).
        if isinstance(data, Variable):
            raise TypeError(
                'Expected an array-like object, but found a Variable: '
                'maybe you are trying to call a function on a (possibly '
                'shared) variable instead of a numeric array?')

        if ((type(data) is np.ndarray) and
                (data.dtype == self.numpy_dtype)):
            if data.dtype.num != self.numpy_dtype.num:
                data = asarray(data, dtype=self.dtype)
            # -- now fall through to ndim check
        elif ((type(data) is np.memmap) and
              (data.dtype == self.numpy_dtype)):
            # numpy.memmap is a "safe" subclass of ndarray,
            # so we can use it wherever we expect a base ndarray.
            # however, casting it would defeat the purpose of not
            # loading the whole data into memory
            pass
        elif strict:
            # If any of the two conditions above was not met,
            # we raise a meaningful TypeError.
            if not (type(data) is np.ndarray):
                raise TypeError("%s expected a ndarray object." % self,
                                data, type(data))
            if data.dtype != self.numpy_dtype:
                raise TypeError(("%s expected a ndarray object with "
                                "dtype = %s (got %s).") %
                                (self, self.numpy_dtype, data.dtype))
            assert False, "This point should never be reached."
        else:
            if allow_downcast:
                # Convert to self.dtype, regardless of the type of data
                data = asarray(data, dtype=self.dtype)
                # TODO: consider to pad shape with ones to make it consistent
                # with self.broadcastable... like vector->row type thing
            else:
                if isinstance(data, np.ndarray):
                    # Check if self.dtype can accurately represent data
                    # (do not try to convert the data)
                    up_dtype = scal.upcast(self.dtype, data.dtype)
                    if up_dtype == self.dtype:
                        # Bug in the following line when data is a
                        # scalar array, see
                        # http://projects.scipy.org/numpy/ticket/1611
                        # data = data.astype(self.dtype)
                        data = asarray(data, dtype=self.dtype)
                    if up_dtype != self.dtype:
                        err_msg = (
                            '%s cannot store a value of dtype %s without '
                            'risking loss of precision. If you do not mind '
                            'this loss, you can: '
                            '1) explicitly cast your data to %s, or '
                            '2) set "allow_input_downcast=True" when calling '
                            '"function". Value: "%s"'
                            % (self, data.dtype, self.dtype, repr(data)))
                        raise TypeError(err_msg)
                elif allow_downcast is None and type(data) is float:
                    # Special case where we allow downcasting of Python float
                    # literals to floatX, even when floatX=='float32'
                    data = asarray(data, self.dtype)
                else:
                    # data has to be converted.
                    # Check that this conversion is lossless
                    converted_data = asarray(data, self.dtype)
                    # We use the `values_eq` static function from TensorType
                    # to handle NaN values.
                    if TensorType.values_eq(np.asarray(data),
                                            converted_data,
                                            force_same_dtype=False):
                        data = converted_data
                    else:
                        # Do not print a too long description of data
                        # (ndarray truncates it, but it's not sure for data)
                        str_data = str(data)
                        if len(str_data) > 80:
                            str_data = str_data[:75] + '(...)'

                        err_msg = (
                            '%s cannot store accurately value %s, '
                            'it would be represented as %s. '
                            'If you do not mind this precision loss, you can: '
                            '1) explicitly convert your data to a numpy array '
                            'of dtype %s, or '
                            '2) set "allow_input_downcast=True" when calling '
                            '"function".'
                            % (self, data, converted_data, self.dtype))
                        raise TypeError(err_msg, data)

        if self.ndim != data.ndim:
            raise TypeError("Wrong number of dimensions: expected %s,"
                            " got %s with shape %s." % (self.ndim, data.ndim,
                                                        data.shape))
        if not data.flags.aligned:
            try:
                msg = "object buffer" + str(data.data)
            except AttributeError:
                msg = ""
            raise TypeError("The numpy.ndarray object is not aligned."
                            " Theano C code does not support that.",
                            msg,
                            "object shape", data.shape,
                            "object strides", data.strides,
                            "object dtype", data.dtype)

        i = 0
        for b in self.broadcastable:
            if b and data.shape[i] != 1:
                raise TypeError("Non-unit value on shape on a broadcastable"
                                " dimension.", data.shape, self.broadcastable)
            i += 1
        if (self.filter_checks_isfinite and
                not np.all(np.isfinite(data))):
            raise ValueError("non-finite elements not allowed")
        return data

    def filter_variable(self, other, allow_convert=True):
        """
        Convert a symbolic Variable into a TensorType, if compatible.

        For the moment, only a TensorType and GpuArrayType will be
        converted, provided they have the same number of dimensions
        and dtype and have "compatible" broadcastable pattern.

        """
        if hasattr(other, '_as_TensorVariable'):
            other = other._as_TensorVariable()

        if not isinstance(other, Variable):
            # The value is not a Variable: we cast it into
            # a Constant of the appropriate Type.
            other = self.Constant(type=self, data=other)

        if other.type == self:
            return other

        if allow_convert:
            # Attempt safe broadcast conversion.
            other2 = self.convert_variable(other)
            if other2 is not None and other2.type == self:
                return other2

        raise TypeError(
            'Cannot convert Type %(othertype)s '
            '(of Variable %(other)s) into Type %(self)s. '
            'You can try to manually convert %(other)s into a %(self)s.' %
            dict(othertype=other.type,
                 other=other,
                 self=self))

    def value_validity_msg(self, a):
        try:
            self.filter(a, strict=True)
        except Exception as e:
            return str(e)
        return "value is valid"

    def dtype_specs(self):
        """
        Return a tuple (python type, c type, numpy typenum) that corresponds
        to self.dtype.

        This function is used internally as part of C code generation.

        """
        # TODO: add more type correspondances for e.g. int32, int64, float32,
        # complex64, etc.
        try:
            return {
                'float16': (float, 'npy_float16', 'NPY_FLOAT16'),
                'float32': (float, 'npy_float32', 'NPY_FLOAT32'),
                'float64': (float, 'npy_float64', 'NPY_FLOAT64'),
                'bool': (bool, 'npy_bool', 'NPY_BOOL'),
                'uint8': (int, 'npy_uint8', 'NPY_UINT8'),
                'int8': (int, 'npy_int8', 'NPY_INT8'),
                'uint16': (int, 'npy_uint16', 'NPY_UINT16'),
                'int16': (int, 'npy_int16', 'NPY_INT16'),
                'uint32': (int, 'npy_uint32', 'NPY_UINT32'),
                'int32': (int, 'npy_int32', 'NPY_INT32'),
                'uint64': (int, 'npy_uint64', 'NPY_UINT64'),
                'int64': (int, 'npy_int64', 'NPY_INT64'),
                'complex128': (complex, 'theano_complex128', 'NPY_COMPLEX128'),
                'complex64': (complex, 'theano_complex64', 'NPY_COMPLEX64')
            }[self.dtype]
        except KeyError:
            raise TypeError("Unsupported dtype for %s: %s"
                            % (self.__class__.__name__, self.dtype))

    def to_scalar_type(self):
        return scal.get_scalar_type(dtype=self.dtype)

    def __eq__(self, other):
        """
        Compare True iff other is the same kind of TensorType.

        """
        return type(self) == type(other) and other.dtype == self.dtype \
            and other.broadcastable == self.broadcastable

    def convert_variable(self, var):
        if (type(self) == type(var.type) and  # noqa
            self.dtype == var.type.dtype and
            self.ndim == var.type.ndim and
            all(sb == ob or ob for sb, ob in zip(self.broadcastable,
                                                 var.type.broadcastable))):
            #return theano.tensor.patternbroadcast(var, self.broadcastable)
            return var

    @staticmethod
    def may_share_memory(a, b):
        # This is a method of TensorType, so both a and b should be ndarrays
        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            return np.may_share_memory(a, b)
        else:
            return False

    @staticmethod
    def values_eq(a, b, force_same_dtype=True):
        # TODO: check to see if the shapes must match
        #      for now, we err on safe side...
        if a.shape != b.shape:
            return False
        if force_same_dtype and a.dtype != b.dtype:
            return False
        a_eq_b = (a == b)
        r = np.all(a_eq_b)
        if r:
            return True
        # maybe the trouble is that there are NaNs
        a_missing = np.isnan(a)
        if a_missing.any():
            b_missing = np.isnan(b)
            return np.all(a_eq_b + (a_missing == b_missing))
        else:
            return False

    @staticmethod
    def values_eq_approx(a, b, allow_remove_inf=False, allow_remove_nan=False,
                         rtol=None, atol=None):
        return values_eq_approx(a, b, allow_remove_inf, allow_remove_nan,
                                rtol, atol)

    def __hash__(self):
        """Hash equal for same kinds of TensorType"""
        return hashtype(self) ^ hash(self.dtype) ^ hash(self.broadcastable)

    ndim = property(lambda self: len(self.broadcastable),
                    doc="number of dimensions")
    """
    Number of dimensions.

    This read-only property is the preferred way to get the number of
    dimensions of a `TensorType`.

    """

    def make_variable(self, name=None):
        """
        Return a `TensorVariable` of this type.

        Parameters
        ----------
        name : str
            A pretty name to identify this `Variable` when printing and
            debugging

        """
        return self.Variable(self, name=name)

    def __str__(self):
        if self.name:
            return self.name
        else:
            b = self.broadcastable
            named_broadcastable = {(): 'scalar',
                                   (False,): 'vector',
                                   (False, True): 'col',
                                   (True, False): 'row',
                                   (False, False): 'matrix'}
            if b in named_broadcastable:
                bcast = named_broadcastable[b]
            else:
                if any(b):
                    bcast = str(b)
                else:
                    bcast = '%iD' % len(b)
            return "TensorType(%s, %s)" % (str(self.dtype), bcast)

    def value_zeros(self, shape):
        """
        Create an numpy ndarray full of 0 values.

        """
        return np.zeros(shape, dtype=self.dtype)

    def get_shape_info(self, obj):
        """
        Return the information needed to compute the memory size of ``obj``.

        The memory size is only the data, so this excludes the container.
        For an ndarray, this is the data, but not the ndarray object and
        other data structures such as shape and strides.

        ``get_shape_info()`` and ``get_size()`` work in tandem for the memory
        profiler.

        ``get_shape_info()`` is called during the execution of the function.
        So it is better that it is not too slow.

        ``get_size()`` will be called on the output of this function
        when printing the memory profile.

        Parameters
        ----------
        obj
            The object that this Type represents during execution.

        Returns
        -------
        object
            Python object that ``self.get_size()`` understands.

        """
        return obj.shape

    def get_size(self, shape_info):
        """
        Number of bytes taken by the object represented by shape_info.

        Parameters
        ----------
        shape_info
            The output of the call to get_shape_info().

        Returns
        -------
        int
            The number of bytes taken by the object described by ``shape_info``.

        """
        if shape_info:
            return np.prod(shape_info) * np.dtype(self.dtype).itemsize
        else:  # a scalar
            return np.dtype(self.dtype).itemsize

    def __repr__(self):
        return str(self)

## Node types
class _tensor_py_operators(object):
    # UNARY
    def __abs__(self):
        return theano.tensor.basic.abs_(self)

    def __neg__(self):
        return theano.tensor.basic.neg(self)

    # CASTS
    # REMOVED THESE BECAUSE PYTHON appears to require __int__ to return
    # an int. -JB 20081112
    # def __int__(self): return convert_to_int32(self)
    # def __float__(self): return convert_to_float64(self)
    # def __complex__(self): return convert_to_complex128(self)

    # COMPARISONS
    _is_nonzero = True

    def __lt__(self, other):
        rval = theano.tensor.basic.lt(self, other)
        rval._is_nonzero = False
        return rval

    def __le__(self, other):
        rval = theano.tensor.basic.le(self, other)
        rval._is_nonzero = False
        return rval

    def __gt__(self, other):
        rval = theano.tensor.basic.gt(self, other)
        rval._is_nonzero = False
        return rval

    def __ge__(self, other):
        rval = theano.tensor.basic.ge(self, other)
        rval._is_nonzero = False
        return rval

    def __nonzero__(self):
        # Python 2.x
        return self.__bool__()

    def __bool__(self):
        # This is meant to prohibit stuff like a < b < c, which is internally
        # implemented as (a < b) and (b < c). The trouble with this is the
        # side-effect that checking for a non-NULL a by typing "if a: ..."
        # uses the same __nonzero__ method.  We want these both to work, but
        # it seems impossible.  Currently, all vars evaluate to nonzero except
        # the return values of comparison operators, which raise this
        # exception.  If you can think of a better solution, go for it!
        #
        # __bool__ is Python 3.x data model. __nonzero__ is Python 2.x.
        if self._is_nonzero:
            return True
        else:
            raise TypeError(
                "Variables do not support boolean operations."
            )

    # BITWISE
    def __invert__(self):
        return theano.tensor.basic.invert(self)

    def __and__(self, other):
        return theano.tensor.basic.and_(self, other)

    def __or__(self, other):
        return theano.tensor.basic.or_(self, other)

    def __xor__(self, other):
        return theano.tensor.basic.xor(self, other)

    def __rand__(self, other):
        return theano.tensor.basic.and_(other, self)

    def __ror__(self, other):
        return theano.tensor.basic.or_(other, self)

    def __rxor__(self, other):
        return theano.tensor.basic.xor(other, self)

    # def __iand__(self, other):
    #    return _and_inplace(self, other)
    #
    # def __ior__(self, other):
    #    return _or_inplace(self, other)
    #
    # def __ixor__(self, other):
    #    return _xor_inplace(self, other)

    # ARITHMETIC - NORMAL
    def __add__(self, other):
        try:
            return theano.tensor.basic.add(self, other)
        # We should catch the minimum number of exception here.
        # Otherwise this will convert error when Theano flags
        # compute_test_value is used
        # Evidently, we need to catch NotImplementedError
        # TypeError from as_tensor_variable are caught in Elemwise.make_node
        # Oterwise TensorVariable * SparseVariable won't work!
        except (NotImplementedError, AsTensorError):
            # We must return NotImplemented and not an
            # NotImplementedError or raise an NotImplementedError.
            # That way python will give a good error message like this
            # `TypeError: unsupported operand type(s) for +:
            # 'TensorVariable' and 'TensorVariable'`
            return NotImplemented

    def __sub__(self, other):
        # See explanation in __add__ for the error catched
        # and the return value in that case
        try:
            return theano.tensor.basic.sub(self, other)
        except (NotImplementedError, AsTensorError):
            return NotImplemented

    def __mul__(self, other):
        # See explanation in __add__ for the error catched
        # and the return value in that case
        try:
            return theano.tensor.mul(self, other)
        except (NotImplementedError, AsTensorError):
            return NotImplemented

    def __div__(self, other):
        # See explanation in __add__ for the error catched
        # and the return value in that case
        try:
            return theano.tensor.basic.div_proxy(self, other)
        except IntegerDivisionError:
            # This is to raise the exception that occurs when trying to divide
            # two integer arrays (currently forbidden).
            raise
        except (NotImplementedError, AsTensorError):
            return NotImplemented
    __truediv__ = __div__

    def __pow__(self, other):
        # See explanation in __add__ for the error catched
        # and the return value in that case
        try:
            return theano.tensor.basic.pow(self, other)
        except (NotImplementedError, AsTensorError):
            return NotImplemented

    def __mod__(self, other):
        # See explanation in __add__ for the error catched
        # and the return value in that case
        try:
            return theano.tensor.basic.mod_check(self, other)
        except ComplexError:
            # This is to raise the exception that occurs when trying to compute
            # x % y with either x or y a complex number.
            raise
        except (NotImplementedError, AsTensorError):
            return NotImplemented

    def __divmod__(self, other):
        return theano.tensor.basic.divmod(self, other)

    def __truediv__(self, other):
        return theano.tensor.basic.true_div(self, other)

    def __floordiv__(self, other):
        return theano.tensor.basic.floor_div(self, other)

    def __rtruediv__(self, other):
        return theano.tensor.basic.true_div(other, self)

    def __rfloordiv__(self, other):
        return theano.tensor.basic.floor_div(other, self)

    # DO NOT USE THESE BECAUSE INPLACE OPS SHOULD BE INSERTED
    # BY OPTIMIZATIONS ONLY
    # ARITHMETIC - INPLACE
    # def __iadd__(self, other):
    #    return _add_inplace(self, other)
    # def __isub__(self, other):
    #    return _sub_inplace(self, other)
    #
    # def __imul__(self, other):
    #    return _mul_inplace(self, other)
    #
    # def __idiv__(self, other):
    #    return _div_inplace(self, other)
    #
    # def __ipow__(self, other):
    #    return _pow_inplace(self, other)

    # ARITHMETIC - RIGHT-OPERAND
    def __radd__(self, other):
        return theano.tensor.basic.add(other, self)

    def __rsub__(self, other):
        return theano.tensor.basic.sub(other, self)

    def __rmul__(self, other):
        return theano.tensor.basic.mul(other, self)

    def __rdiv__(self, other):
        return theano.tensor.basic.div_proxy(other, self)

    def __rmod__(self, other):
        return theano.tensor.basic.mod(other, self)

    def __rdivmod__(self, other):
        return theano.tensor.basic.divmod(other, self)

    def __rpow__(self, other):
        return theano.tensor.basic.pow(other, self)

    # TRANSPOSE
    T = property(lambda self: theano.tensor.basic.transpose(self))

    def transpose(self, *axes):
        """

        Returns
        -------
        object
            `tensor.transpose(self, axes)` or `tensor.transpose(self, axes[0])`.

        If only one `axes` argument is provided and it is iterable, then it is
        assumed to be the entire axes tuple, and passed intact to
        tensor.transpose.

        """
        if len(axes) == 0:
            return theano.tensor.basic.transpose(self)
        try:
            iter(axes[0])
            iterable = True
        except TypeError:
            iterable = False
        if len(axes) == 1 and iterable:
            return theano.tensor.basic.transpose(self, axes[0])
        else:
            return theano.tensor.basic.transpose(self, axes)

    shape = property(lambda self: theano.tensor.basic.shape(self))

    size = property(lambda self: self.shape[0] if self.ndim == 1 else
                    theano.tensor.basic.prod(self.shape))

    # We can't implement __len__ to provide a better error message.
    def any(self, axis=None, keepdims=False):
        return theano.tensor.basic.any(self, axis=axis, keepdims=keepdims)

    def all(self, axis=None, keepdims=False):
        return theano.tensor.basic.all(self, axis=axis, keepdims=keepdims)

    # Otherwise TensorVariable[:-1] does not work as Python 2.5.1 calls
    # __len__ before calling __getitem__. It also does not catch the raised
    # Exception!
    # def __len__(self):
    #     # We can't implement __len__ as Python requests that this
    #     # function returns an integer >=0
    #     raise Exception("Theano Variables can't work with len(Theano "
    #                     "Variable) due to Python restriction. You can use "
    #                     "TheanoVariable.shape[0] instead.")

    def reshape(self, shape, ndim=None):
        """Return a reshaped view/copy of this variable.

        Parameters
        ----------
        shape
            Something that can be converted to a symbolic vector of integers.
        ndim
            The length of the shape. Passing None here means for
            Theano to try and guess the length of `shape`.


        .. warning:: This has a different signature than numpy's
                     ndarray.reshape!
                     In numpy you do not need to wrap the shape arguments
                     in a tuple, in theano you do need to.

        """

        if ndim is not None:
            if not isinstance(ndim, integer_types):
                raise ValueError("Expected ndim to be an integer, is " +
                                 str(type(ndim)))

        return theano.tensor.basic.reshape(self, shape, ndim=ndim)

    def dimshuffle(self, *pattern):
        """
        Reorder the dimensions of this variable, optionally inserting
        broadcasted dimensions.

        Parameters
        ----------
        pattern
            List/tuple of int mixed with 'x' for broadcastable dimensions.

        Examples
        --------
        For example, to create a 3D view of a [2D] matrix, call
        ``dimshuffle([0,'x',1])``.  This will create a 3D view such that the
        middle dimension is an implicit broadcasted dimension.  To do the same
        thing on the transpose of that matrix, call ``dimshuffle([1, 'x', 0])``.

        Notes
        -----
        This function supports the pattern passed as a tuple, or as a
        variable-length argument (e.g. ``a.dimshuffle(pattern)`` is equivalent
        to ``a.dimshuffle(*pattern)`` where ``pattern`` is a list/tuple of ints
        mixed with 'x' characters).

        See Also
        --------
        DimShuffle

        """
        if (len(pattern) == 1) and (isinstance(pattern[0], (list, tuple))):
            pattern = pattern[0]
        op = theano.tensor.basic.DimShuffle(list(self.type.broadcastable),
                                            pattern)
        return op(self)

    def flatten(self, ndim=1):
        return theano.tensor.basic.flatten(self, ndim)

    def ravel(self):
        return theano.tensor.basic.flatten(self)

    def diagonal(self, offset=0, axis1=0, axis2=1):
        return theano.tensor.basic.diagonal(self, offset, axis1, axis2)

    # Transfer the data to another device
    def transfer(self, target):
        """
        If `target` is `'cpu'` this will transfer to a TensorType (if
        not already one).  Other types may define additional targets.

        Parameters
        ----------
        target : str
            The desired location of the output variable
        """
        return theano.tensor.transfer(self, target)

    # Elemwise
    def arccos(self):
        return theano.tensor.arccos(self)

    def arccosh(self):
        return theano.tensor.arccosh(self)

    def arcsin(self):
        return theano.tensor.arcsin(self)

    def arcsinh(self):
        return theano.tensor.arcsinh(self)

    def arctan(self):
        return theano.tensor.arctan(self)

    def arctanh(self):
        return theano.tensor.arctanh(self)

    def ceil(self):
        return theano.tensor.ceil(self)

    def cos(self):
        return theano.tensor.cos(self)

    def cosh(self):
        return theano.tensor.cosh(self)

    def deg2rad(self):
        return theano.tensor.deg2rad(self)

    def exp(self):
        return theano.tensor.exp(self)

    def exp2(self):
        return theano.tensor.exp2(self)

    def expm1(self):
        return theano.tensor.expm1(self)

    def floor(self):
        return theano.tensor.floor(self)

    def log(self):
        return theano.tensor.log(self)

    def log10(self):
        return theano.tensor.log10(self)

    def log1p(self):
        return theano.tensor.log1p(self)

    def log2(self):
        return theano.tensor.log2(self)

    def rad2deg(self):
        return theano.tensor.rad2deg(self)

    def sin(self):
        return theano.tensor.sin(self)

    def sinh(self):
        return theano.tensor.sinh(self)

    def sqrt(self):
        return theano.tensor.sqrt(self)

    def tan(self):
        return theano.tensor.tan(self)

    def tanh(self):
        return theano.tensor.tanh(self)

    def trunc(self):
        return theano.tensor.trunc(self)

    # CASTING
    def astype(self, dtype):
        return theano.tensor.cast(self, dtype)

    # SLICING/INDEXING
    def __getitem__(self, args):

        def includes_bool(args_el):
            if (isinstance(args_el, (np.bool_, bool)) or
                    (hasattr(args_el, 'dtype') and args_el.dtype == 'bool')):
                return True
            if (not isinstance(args_el, theano.tensor.Variable) and
                    isinstance(args_el, collections.Iterable)):
                for el in args_el:
                    if includes_bool(el):
                        return True
            return False

        if (isinstance(args, list) and
                any([isinstance(a, slice) for a in args])):
            pass
        elif not isinstance(args, tuple):
            args = args,

        # Count the dimensions, check for bools and find ellipses.
        ellipses = []
        index_dim_count = 0
        for i, arg in enumerate(args):
            if arg is np.newaxis:
                # no increase in index_dim_count
                pass
            elif arg is Ellipsis:
                # no increase in index_dim_count
                ellipses.append(i)
            elif (isinstance(arg, (np.ndarray, theano.tensor.Variable)) and
                    hasattr(arg, 'dtype') and arg.dtype == 'bool'):
                index_dim_count += arg.ndim
            else:
                # Python arrays can contain a mixture of bools and integers,
                # which requires complex rules to handle all special cases.
                # These rules differ slightly between NumPy versions.
                # Since earlier versions of Theano did not support any boolean
                # indexing, it is safe to throw an error if we encounter
                # any of these difficult cases.
                if includes_bool(arg):
                    raise TypeError('TensorType does not support Python bools '
                                    'for indexing, such as tensor[[True, False]]. '
                                    'To use a boolean mask, convert the mask to '
                                    'a NumPy array first, e.g., '
                                    'tensor[numpy.array([True, False])].')
                index_dim_count += 1

        # Check if the number of dimensions isn't too large.
        if self.ndim < index_dim_count:
            raise IndexError('too many indices for array')

        # Convert an Ellipsis if provided into an appropriate number of
        # slice(None).
        if len(ellipses) > 1:
            raise IndexError(
                "an index can only have a single Ellipsis (`...`)")
        elif len(ellipses) == 1:
            ellipsis_at = ellipses[0]
            args = list(args)
            args[ellipsis_at: ellipsis_at + 1] = (
                [slice(None)] * (self.ndim - index_dim_count))

        def is_empty_array(val):
            return ((isinstance(val, (tuple, list)) and len(val) == 0) or
                    (isinstance(val, np.ndarray) and val.size == 0))

        # Force input to be int64 datatype if input is an empty list or tuple
        # Else leave it as is if it is a real number
        args = tuple([np.array(inp, dtype=np.int64)
                      if(is_empty_array(inp)) else inp for inp in args])
        # Convert python literals to theano constants
        args = theano.tensor.subtensor.make_constant(args)
        # Determine if advanced indexing is needed or not
        # The logic is already in Subtensor.convert: if it succeeds,
        # standard indexing is used; if it fails with
        # AdvancedIndexingError, advanced indexing, or
        # AdvancedBooleanIndexingError, advanced indexing with boolean masks
        advanced = False
        advanced_boolean = False
        axis = None
        for i, arg in enumerate(args):
            try:
                if arg is not np.newaxis:
                    theano.tensor.subtensor.Subtensor.convert(arg)
            except theano.tensor.subtensor.AdvancedIndexingError:
                if advanced:
                    axis = None
                    break
                else:
                    advanced = True
                    axis = i
            except theano.tensor.subtensor.AdvancedBooleanIndexingError:
                advanced = False
                advanced_boolean = True
                break

        if advanced_boolean:
            return theano.tensor.subtensor.advanced_boolean_subtensor(self, *args)
        elif advanced:
            if (axis is not None and
                all(isinstance(a, slice) and
                    equal_slices(a, slice(None)) for a in args[:axis]) and
                all(isinstance(a, slice) and
                    equal_slices(a, slice(None)) for a in args[axis + 1:]) and
                (not hasattr(args[axis], 'dtype') or args[axis].dtype != 'bool') and
                isinstance(args[axis],
                           (np.ndarray, list,
                            TensorVariable, TensorConstant,
                            theano.tensor.sharedvar.TensorSharedVariable))):
                return self.take(args[axis], axis)
            else:
                return theano.tensor.subtensor.advanced_subtensor(self, *args)
        else:
            if np.newaxis in args:
                # None (aka np.newaxis) in numpy indexing means to add a
                # broadcastable dimension, which theano traditionally did with
                # the dimshuffle op.  The following code converts numpy-style
                # indexing on self to traditional [read: implemented] theano
                # indexing on a dimshuffled view of self.

                counter = 0
                pattern = []
                new_args = []
                for arg in args:
                    if arg == np.newaxis:
                        pattern.append('x')
                        new_args.append(slice(None, None, None))
                    else:
                        pattern.append(counter)
                        counter += 1
                        new_args.append(arg)
                view = self.dimshuffle(pattern)
                full_slices = True
                for arg in new_args:
                    # We can't do arg == slice(None, None, None) as in
                    # Python 2.7, this call __lt__ if we have a slice
                    # with some symbolic variable.
                    if not (isinstance(arg, slice) and
                            arg.start is None and
                            arg.stop is None and
                            arg.step is None):
                        full_slices = False
                if full_slices:
                    return view
                else:
                    return view.__getitem__(tuple(new_args))
            else:
                return theano.tensor.subtensor.Subtensor(args)(
                    self, *theano.tensor.subtensor.Subtensor.collapse(
                        args,
                        lambda entry: isinstance(entry, Variable)))

    def take(self, indices, axis=None, mode='raise'):
        return theano.tensor.subtensor.take(self, indices, axis, mode)

    # COPYING
    def copy(self, name=None):
        """Return a symbolic copy and optionally assign a name.

        Does not copy the tags.
        """
        copied_variable = theano.tensor.basic.tensor_copy(self)
        copied_variable.name = name
        return copied_variable

    def __iter__(self):
        try:
            for i in xrange(theano.tensor.basic.get_vector_length(self)):
                yield self[i]
        except TypeError:
            # This prevents accidental iteration via builtin.sum(self)
            raise TypeError(('TensorType does not support iteration. '
                             'Maybe you are using builtin.sum instead of '
                             'theano.tensor.sum? (Maybe .max?)'))

    # CONVENIENT ACCESS TO TYPE PROPERTIES
    ndim = property(lambda self: self.type.ndim)
    """The rank of this tensor."""

    broadcastable = property(lambda self: self.type.broadcastable)
    """
    The broadcastable signature of this tensor.

    See Also
    --------
    broadcasting

    """

    dtype = property(lambda self: self.type.dtype)
    """The dtype of this tensor."""

    # extra pseudo-operator symbols
    def __dot__(left, right):
        return theano.tensor.basic.dot(left, right)

    def __rdot__(right, left):
        return theano.tensor.basic.dot(left, right)

    dot = __dot__

    def sum(self, axis=None, dtype=None, keepdims=False, acc_dtype=None):
        """See `theano.tensor.sum`."""
        return theano.tensor.basic.sum(self, axis=axis,
                                       dtype=dtype, keepdims=keepdims,
                                       acc_dtype=acc_dtype)

    def prod(self, axis=None, dtype=None, keepdims=False, acc_dtype=None):
        """See `theano.tensor.prod`."""
        return theano.tensor.basic.prod(self, axis=axis,
                                        dtype=dtype, keepdims=keepdims,
                                        acc_dtype=acc_dtype)

    def norm(self, L, axis=None, keepdims=False):
        if L == 0:
            raise NotImplementedError()
        if np.isinf(L):
            raise NotImplementedError()
        # optimizations will/should catch cases like L=1, L=2
        y = theano.tensor.basic.pow(
            theano.tensor.basic.pow(
                theano.tensor.basic.abs_(self), L).sum(axis=axis), 1.0 / L)
        if keepdims:
            return theano.tensor.basic.makeKeepDims(self, y, axis)
        else:
            return y

    def mean(self, axis=None, dtype=None, keepdims=False, acc_dtype=None):
        """See `theano.tensor.mean`."""
        return theano.tensor.basic.mean(self, axis=axis,
                                        dtype=dtype, keepdims=keepdims,
                                        acc_dtype=acc_dtype)

    def var(self, axis=None, ddof=0, keepdims=False, corrected=False):
        """See `theano.tensor.var`."""
        return theano.tensor.basic.var(self, axis=axis, ddof=ddof,
                                       keepdims=keepdims, corrected=corrected)

    def std(self, axis=None, ddof=0, keepdims=False, corrected=False):
        """See `theano.tensor.std`."""
        return theano.tensor.basic.std(self, axis=axis, ddof=ddof,
                                       keepdims=keepdims, corrected=corrected)

    def min(self, axis=None, keepdims=False):
        """See `theano.tensor.min`."""
        return theano.tensor.basic.min(self, axis, keepdims=keepdims)

    def max(self, axis=None, keepdims=False):
        """See `theano.tensor.max`."""
        return theano.tensor.basic.max(self, axis, keepdims=keepdims)

    def argmin(self, axis=None, keepdims=False):
        """See `theano.tensor.argmin`."""
        return theano.tensor.basic.argmin(self, axis, keepdims=keepdims)

    def argmax(self, axis=None, keepdims=False):
        """See `theano.tensor.argmax`."""
        return theano.tensor.basic.argmax(self, axis, keepdims=keepdims)

    def nonzero(self, return_matrix=False):
        """See `theano.tensor.nonzero`."""
        return theano.tensor.basic.nonzero(self, return_matrix=return_matrix)

    def nonzero_values(self):
        """See `theano.tensor.nonzero_values`."""
        return theano.tensor.basic.nonzero_values(self)

    def sort(self, axis=-1, kind='quicksort', order=None):
        """See `theano.tensor.sort`."""
        return theano.tensor.sort(self, axis, kind, order)

    def argsort(self, axis=-1, kind='quicksort', order=None):
        """See `theano.tensor.argsort`."""
        return theano.tensor.argsort(self, axis, kind, order)

    def clip(self, a_min, a_max):
        "Clip (limit) the values in an array."
        return theano.tensor.basic.clip(self, a_min, a_max)

    def conj(self):
        """See `theano.tensor.conj`."""
        return theano.tensor.basic.conj(self)

    conjugate = conj

    def repeat(self, repeats, axis=None):
        """See `theano.tensor.repeat`."""
        return theano.tensor.extra_ops.repeat(self, repeats, axis)

    def round(self, mode=None):
        """See `theano.tensor.round`."""
        return theano.tensor.basic.round(self, mode)

    def trace(self):
        return theano.tensor.nlinalg.trace(self)

    # TO TRUMP NUMPY OPERATORS
    __array_priority__ = 1000

    def get_scalar_constant_value(self):
        return theano.tensor.basic.get_scalar_constant_value(self)

    def zeros_like(model, dtype=None):
        return theano.tensor.basic.zeros_like(model, dtype=dtype)

    def ones_like(model, dtype=None):
        return theano.tensor.basic.ones_like(model, dtype=dtype)

    def cumsum(self, axis=None):
        return theano.tensor.extra_ops.cumsum(self, axis)

    def cumprod(self, axis=None):
        return theano.tensor.extra_ops.cumprod(self, axis)

    def searchsorted(self, v, side='left', sorter=None):
        return theano.tensor.extra_ops.searchsorted(self, v, side, sorter)

    def ptp(self, axis=None):
        """See 'theano.tensor.ptp'."""

        return theano.tensor.ptp(self, axis)

    def swapaxes(self, axis1, axis2):
        """
        Return 'tensor.swapaxes(self, axis1, axis2).

        If a matrix is provided with the right axes, its transpose
        will be returned.

        """
        return theano.tensor.basic.swapaxes(self, axis1, axis2)

    def fill(self, value):
        """Fill inputted tensor with the assigned value."""
        return theano.tensor.basic.fill(self, value)

    def choose(self, choices, out=None, mode='raise'):
        """
        Construct an array from an index array and a set of arrays to choose
        from.

        """
        return theano.tensor.basic.choose(self, choices, out=None, mode='raise')

    def squeeze(self):
        """
        Remove broadcastable dimensions from the shape of an array.

        It returns the input array, but with the broadcastable dimensions
        removed. This is always `x` itself or a view into `x`.

        """
        return theano.tensor.extra_ops.squeeze(self)

    def compress(self, a, axis=None):
        """Return selected slices only."""
        return theano.tensor.extra_ops.compress(self, a, axis=axis)


class TensorVariable(_tensor_py_operators, Variable):
    """
    Subclass to add the tensor operators to the basic `Variable` class.

    """

    def __init__(self, type, owner=None, index=None, name=None):
        super(TensorVariable, self).__init__(type, owner=owner,
                                             index=index, name=name)

TensorType.Variable = TensorVariable



#def as_tensor_variable(x, name=None, ndim=None):
#    """Return `x`, transformed into a `TensorType`.
#
#    This function is often used by `make_node` methods of `Op` subclasses
#    to turn ndarrays, numbers, `Scalar` instances, `Apply` instances and
#    `TensorType` instances into valid input list elements.
#
#    Parameters
#    ----------
#    x : Apply instance, Variable instance, numpy.ndarray, or number
#        This thing will be transformed into a `Variable` in a sensible way. An
#        ndarray argument will not be copied, but a list of numbers will be
#        copied to make an ndarray.
#    name : str or None
#        If a new `Variable` instance is created, it will be named with this
#        string.
#    ndim : None or integer
#        Return a Variable with this many dimensions.
#
#    Raises
#    ------
#    ValueError
#        If an `Apply` with more than one output is fetched or
#        if `x` cannot be made into a Variable with `ndim` dimensions.
#    AsTensorError
#        If `x` cannot be converted to a TensorType Variable.
#
#    """
#    if hasattr(x, '_as_TensorVariable'):
#        return x._as_TensorVariable()  # TODO: pass name and ndim arguments
#
#    if isinstance(x, Apply):
#        # use Apply's default output mechanism
#        if (x.op.default_output is None) and (len(x.outputs) != 1):
#            raise ValueError(
#                "It is ambiguous which output of a multi-output Op has"
#                " to be fetched.", x)
#
#        x = x.default_output()
#    if isinstance(x, Variable):
#        if isinstance(x.type, scal.Scalar):
#            x = tensor_from_scalar(x)
#
#        if not isinstance(x.type, TensorType):
#            raise AsTensorError(
#                "Variable type field must be a TensorType.", x, x.type)
#
#        if ndim is None:
#            return x
#        else:
#            if (x.type.ndim > ndim):
#                # strip off leading broadcastable dimensions
#                first_non_broadcastable = [idx for idx in xrange(x.ndim)
#                                           if not x.broadcastable[idx]][0]
#                x = x.dimshuffle(list(range(x.ndim))[first_non_broadcastable:])
#                if x.ndim > ndim:
#                    raise ValueError(
#                        'TensorType could not be cast to have %i dimensions'
#                        % ndim, x.type
#                    )
#                return x
#            elif (x.type.ndim < ndim):
#                return shape_padleft(x, n_ones=(ndim - x.type.ndim))
#            else:
#                return x
#    if isinstance(x, (tuple, list)) and any(isinstance(xi, Variable) for xi in x):
#        try:
#            return stack(x)
#        except (TypeError, ValueError):
#            pass
#
#    if isinstance(x, bool):
#        raise AsTensorError(
#            "Cannot cast True or False as a tensor variable. Please use "
#            "np.array(True) or np.array(False) if you need these constants. "
#            "This error might be caused by using the == operator on "
#            "Variables. v == w does not do what you think it does, "
#            "use theano.tensor.eq(v, w) instead.")
#
#    try:
#        return Constant(x, name=name, ndim=ndim)
#    except TypeError:
#        try:
#            str_x = str(x)
#        except Exception:
#            str_x = repr(x)
#        raise Exception("Cannot convert %s to TensorType" % str_x, type(x))
