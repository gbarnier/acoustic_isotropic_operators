#Module containing the definition of the operator necessary for the solver class
#It takes vector objects from the pyVector class
import pyVector as Vec
import time


class Operator:
	"""Abstract python operator class"""

	#Default class methods/functions
	def __init__(self):
		"""Generic class for operator"""
		return

	def __del__(self):
		"""Default destructor"""
		return

	def setDomainRange(self,domain,range):
		"""Function to set (cloning space) domain and range of the operator"""
		self.domain = domain.cloneSpace()
		self.range = range.cloneSpace()
		return

	def getDomain(self):
		"""Function to return operator domain"""
		return self.domain

	def getRange(self):
		"""Function to return operator range"""
		return self.range

	def checkDomainRange(self,model,data):
		"""Function to check model and data vector sizes"""
		if not self.domain.checkSame(model):
			raise ValueError("Provided model vector does not match operator domain")
		if not self.range.checkSame(data):
			raise ValueError("Provided data vector does not match operator range")
		return

	def dotTest(self,verb=False,maxError=.0001):
		"""Function to perform dot-product test
		   If passing the dot-product test, the function does not throw a Warning
		   Default relative error 10^-4
		"""
		if(verb): print("Dot-product test of forward and adjoint operators")
		if(verb): print("-------------------------------------------------")
		#Allocating temporary vectors for dot-product test
		d1=self.domain.cloneVector()
		d2=self.domain.cloneVector()
		r1=self.range.cloneVector()
		r2=self.range.cloneVector()

		#Randomize the input vectors
		d1.rand()
		r1.rand()

		#Applying forward and adjoint operators with add=False
		if(verb): print("Applying forward operator add=False")
		start = time.time()
		self.forward(False,d1,r2)
		end = time.time()
		if(verb): print("	Runs in: %s seconds"%(end-start))
		if(verb): print("Applying adjoint operator add=False")
		start = time.time()
		self.adjoint(False,d2,r1)
		end = time.time()
		if(verb): print("	Runs in: %s seconds"%(end-start))

		#Computing dot products
		dt1=d1.dot(d2)
		dt2=r1.dot(r2)

		#Dot-product testing
		if(verb): print("Dot products add=False: domain=%s range=%s "%(dt1,dt2))
		if(verb): print("Absolute error: %s"%(abs(dt1-dt2)))
		if(verb): print("Relative error: %s \n"%(abs((dt1-dt2)/dt2)))
		if (abs((dt1-dt2)/dt1) > maxError):
			#Deleting temporary vectors
			del d1,d2,r1,r2
			raise Warning("Dot products failure add=False; relative error greater than tolerance of %s"%(maxError))

		#Applying forward and adjoint operators with add=True
		if(verb): print("\nApplying forward operator add=True")
		start = time.time()
		self.forward(True,d1,r2)
		end = time.time()
		if(verb): print("	Runs in: %s seconds"%(end-start))
		if(verb): print("Applying adjoint operator add=True")
		start = time.time()
		self.adjoint(True,d2,r1)
		end = time.time()
		if(verb): print("	Runs in: %s seconds"%(end-start))

		#Computing dot products
		dt1=d1.dot(d2)
		dt2=r1.dot(r2)

		if(verb): print("Dot products add=True: domain=%s range=%s "%(dt1,dt2))
		if(verb): print("Absolute error: %s"%(abs(dt1-dt2)))
		if(verb): print("Relative error: %s \n"%(abs((dt1-dt2)/dt2)))
		if(abs((dt1-dt2)/dt1) > maxError):
			#Deleting temporary vectors
			del d1,d2,r1,r2
			raise Warning("Dot products failure add=True; relative error greater than tolerance of %s"%(maxError))

		if(verb): print("-------------------------------------------------")

		#Deleting temporary vectors
		del d1,d2,r1,r2
		return


	#Class methods/functions to be overridden
	def forward(self,add,model,data):
		"""Forward operator"""
		raise NotImplementedError("Forward must be overwritten")
		return

	def adjoint(self,add,model,data):
		"""Adjoint operator"""
		raise NotImplementedError("Adjoint must be overwritten")
		return


class scalingOp(Operator):
	"""Simple operator for testing Operator class"""
	def __init__(self,domain,scalar):
		assert(isinstance(scalar,float))
		self.setDomainRange(domain,domain)
		self.scalar=scalar
		return

	def forward(self,add,model,data):
		self.checkDomainRange(model,data)
		sc=0.
		if add: sc=1.
		data.scaleAdd(model,sc,self.scalar)
		return

	def adjoint(self,add,model,data):
		self.checkDomainRange(model,data)
		sc=0.
		if add: sc=1.
		model.scaleAdd(data,sc,self.scalar)
		return

class IdentityOp(Operator):
	"""Identity operator"""

	def __init__(self,domain):
		self.setDomainRange(domain,domain)
		return

	def forward(self,add,model,data):
		self.checkDomainRange(model,data)
		if add:
			data.scaleAdd(model)
		else:
			data.copy(model)
		return

	def adjoint(self,add,model,data):
		self.checkDomainRange(model,data)
		if add:
			model.scaleAdd(data)
		else:
			model.copy(data)
		return


class stackOperator(Operator):
	"""
		      Stack of operators class
	        		| d1 |   | A |
   			   Cm = |    | = |   | m
		            | d2 |   | B |
	"""

	def __init__(self,op1,op2,domain,range):
		"""Constructor for the stacked operator"""
		if(not isinstance(range,Vec.superVector)):
			raise ValueError("ERROR! Provided range vector not a superVector!")
		self.setDomainRange(domain,range)
		self.op1=op1 #A
		self.op2=op2 #B
		self.op1.setDomainRange(domain,range.vec1)
		self.op1.setDomainRange(domain,range.vec2)
		return

	def forward(self,add,model,data):
		"""Forward operator Cm"""
		self.checkDomainRange(model,data)
		# d1 = Am
		self.op1.forward(add,model,data.vec1)
		# d2 = Bm
		self.op2.forward(add,model,data.vec2)
		return


	def adjoint(self,add,model,data):
		"""Adjoint operator C'r = A'r1 + B'r2"""
		self.checkDomainRange(model,data)
		# m = A'd1
		self.op1.adjoint(add,model,data.vec1)
		# m += B'd2
		self.op2.adjoint(True,model,data.vec2)
		return
