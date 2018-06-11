from .. import TensorBase, np
from .sparse_op import sparse_add, sparse_add_scalar, sparse_sub, sparse_sub_scalar, sparse_neg, sparse_mul, sparse_mul_scalar, sparse_div_scalar, sparse_pow_scalar, rsparse_pow_scalar, sparse_abs, sparse_eq, sparse_lt, sparse_le, sparse_gt, sparse_ge, sparse_ne

def type_check(self, other):
	scalar_check = False
	if isinstance(other, TensorBase):
		var = other
	elif np.isscalar(other):
		var = other
		scalar_check = True
	else:
		raise TypeError("input other is not numpy cupy ndarray nor tensor nor python variable, but to be: {}".format(type(other)))

	return var, scalar_check


def add(self, rhs):  # lhs + rhs
	"""Element-wise addition.
	Returns:
		tensor: Output tensor.
	"""
	var, scalar_check = self.type_check(rhs)
	if scalar_check==False:
		return sparse_add(self, rhs)
	else:
		return sparse_add_scalar(self, rhs)


def neg(self): #-x
	"""Element-wise negation.
	Returns:
		tensor: Output tensor.
	"""	
	return sparse_neg(self)


def sub(self, rhs):  # lhs - rhs
	"""Element-wise subtraction.
	Returns:
		tensor: Output tensor.
	"""
	var, scalar_check = self.type_check(rhs)
	if scalar_check==False:
		return sparse_sub(self, rhs)
	else:
		return sparse_sub_scalar(self, rhs)

def rsub(self, rhs): # rhs - lhs 
	"""Element-wise subtraction.
	Returns:
		tensor: Output tensor.
	"""
	return self.__neg__().__add__(rhs)

def mul(self, rhs):  # lhs * rhs
	"""Element-wise multiplication.
	Returns:
		tensor: Output tensor.
	"""
	var, scalar_check = self.type_check(rhs)
	if scalar_check==False:
		return sparse_mul(self, rhs)
	else:
		return sparse_mul_scalar(self, rhs)

def div(self, rhs):  # lhs / rhs
	"""Element-wise division.
	Returns:
		tensor: Output tensor.
	"""
	var, scalar_check = self.type_check(rhs)
	if scalar_check==False:
		raise ValueError("s_div is not implemented")
	else:
		return sparse_div_scalar(self, rhs)

def pow(self, rhs):  # lhs ** rhs
	"""Element-wise power function.
	Returns:
		tensor: Output tensor.
	"""
	var, scalar_check = self.type_check(rhs)
	if scalar_check==False:
		raise ValueError("s_pow is not implemented")
	else:
		return sparse_pow_scalar(self, rhs)

def rpow(self, rhs):  # rhs ** lhs
	"""Element-wise power function.
	Returns:
		tensor: Output tensor.
	"""
	return rsparse_pow_scalar(self, rhs)


def rdiv(self, rhs):  # rhs / lhs
	"""Element-wise division.
	Returns:
		tensor: Output tensor.
	"""
	raise ValueError("s_div is not implemented")

def absolute(self):  # abs(x)
	"""Element-wise absolute.
	Returns:
		tensor: Output tensor.
	"""
	return sparse_abs(self)

def eq(self, rhs):  # rhs == lhs
	"""Element-wise equal.
	Returns:
		tensor: Output tensor.
	"""
	var, scalar_check = self.type_check(rhs)
	if scalar_check==False:
		return sparse_eq(self, rhs)
	else:
		raise ValueError("Are you sure you want to compare scalar with a sparse matrix? Maybe try scatter_eq instead(scatter_eq TODO)")

def lt(self, rhs):  # rhs < lhs
	"""Element-wise less than.
	Returns:
		tensor: Output tensor.
	"""
	var, scalar_check = self.type_check(rhs)
	if scalar_check==False:
		return sparse_lt(self, rhs)
	else:
		raise ValueError("Are you sure you want to compare scalar with a sparse matrix? Maybe try scatter_eq instead(scatter_lt TODO)")


def le(self, rhs):  # rhs <= lhs
	"""Element-wise less equal.
	Returns:
		tensor: Output tensor.
	"""
	var, scalar_check = self.type_check(rhs)
	if scalar_check==False:
		return sparse_le(self, rhs)
	else:
		raise ValueError("Are you sure you want to compare scalar with a sparse matrix? Maybe try scatter_eq instead(scatter_le TODO)")


def gt(self, rhs):  # rhs > lhs
	"""Element-wise greater than.
	Returns:
		tensor: Output tensor.
	"""
	var, scalar_check = self.type_check(rhs)
	if scalar_check==False:
		return sparse_gt(self, rhs)
	else:
		raise ValueError("Are you sure you want to compare scalar with a sparse matrix? Maybe try scatter_eq instead(scatter_gt TODO)")

def ge(self, rhs):  # rhs >= lhs
	"""Element-wise greater than.
	Returns:
		tensor: Output tensor.
	"""
	var, scalar_check = self.type_check(rhs)
	if scalar_check==False:
		return sparse_ge(self, rhs)
	else:
		raise ValueError("Are you sure you want to compare scalar with a sparse matrix? Maybe try scatter_eq instead(scatter_ge TODO)")

def ne(self, rhs):  # rhs != lhs
	"""Element-wise not equal.
	Returns:
		tensor: Output tensor.
	"""
	var, scalar_check = self.type_check(rhs)
	if scalar_check==False:
		return sparse_ne(self, rhs)
	else:
		raise ValueError("Are you sure you want to compare scalar with a sparse matrix? Maybe try scatter_eq instead(scatter_ne TODO)")


def install_variable_arithmetics():
	TensorBase.type_check = type_check
	TensorBase.__neg__ = neg
	TensorBase.__abs__ = absolute
	TensorBase.__add__ = add
	TensorBase.__radd__ = add
	TensorBase.__sub__ = sub
	TensorBase.__rsub__ = rsub
	TensorBase.__mul__ = mul
	TensorBase.__rmul__ = mul
	TensorBase.__div__ = div
	TensorBase.__truediv__ = div
	TensorBase.__rdiv__ = rdiv
	TensorBase.__rtruediv__ = rdiv
	TensorBase.__pow__ = pow
	TensorBase.__rpow__ = rpow
	TensorBase.__gt__ = gt
	TensorBase.__lt__ = lt
	TensorBase.__ge__ = ge
	TensorBase.__le__ = le 
	TensorBase.__eq__ = eq 
	TensorBase.__ne__ = ne