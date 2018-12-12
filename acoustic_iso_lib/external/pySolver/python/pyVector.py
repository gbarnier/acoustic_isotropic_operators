#Module containing the definition of an abstract, in-core, and out-of-core vectors
import numpy as np
import re
import math
import time
import os
from copy import deepcopy
from shutil import copyfile
from sys import version_info
#other modules
import sys_util
import sep_util

#Sep library
import Hypercube
import pySepVector
import SepVector

#regex to read output of Solver_ops
re_dpr=re.compile("DOT RESULT(.*)")

#Vector class and derived classes
class vector:
	"""Abstract python vector class"""
	def __init__(self):
		"""Default constructor"""
		return

	def __del__(self):
		"""Default destructor"""
		return

	#Class vector operations

	def norm(self,N=2):
		"""Function to compute vector N-norm"""
		raise NotImplementedError("norm must be overwritten")
		return

	def zero(self):
		"""Function to zero out a vector"""
		raise NotImplementedError("zero must be overwritten")
		return

	def scale(self,sc):
		"""Function to scale a vector"""
		raise NotImplementedError("scale must be overwritten")
		return

	def rand(self):
		"""Function to randomize a vector"""
		raise NotImplementedError("rand must be overwritten")
		return

	def clone(self):
		"""Function to clone (deep copy) a vector"""
		raise NotImplementedError("clone must be overwritten")
		return

	def cloneSpace(self):
		"""Function to clone vector space"""
		raise NotImplementedError("cloneSpace must be overwritten")
		return

	def cloneVector(self):
		"""Function to clone/allocate vector from vector space"""
		raise NotImplementedError("cloneVector must be overwritten")
		return

	def checkSame(self):
		"""Function to check to make sure the vectors exist in the same space"""
		raise NotImplementedError("checkSame must be overwritten")
		return

	def writeVec(self,filename,mode='w'):
		"""Function to write vector to file"""
		raise NotImplementedError("writeVec must be overwritten")
		return

	#Combination of different vectors

	def copy(self,vec2):
		"""Function to copy vector"""
		raise NotImplementedError("copy must be overwritten")
		return

	def scaleAdd(self,vec2,sc1=1.0,sc2=1.0):
		"""Function to scale two vectors and add them to the first one"""
		raise NotImplementedError("scaleAdd must be overwritten")
		return

	def dot(self,vec2):
		"""Function to compute dot product between two vectors"""
		raise NotImplementedError("dot must be overwritten")
		return

	def multiply(self,vec2):
		"""Function to multiply element-wise two vectors"""
		raise NotImplementedError("multiply must be overwritten")
		return

	def isDifferent(self,vec2):
		"""Function to check if two vectors are identical"""
		raise NotImplementedError("isDifferent must be overwritten")
		return

#Set of vectors (useful to store results and same-Space vectors together)
class vectorSet:
	"""Class to store different vectors that live in the same Space"""

	def __init__(self):
		"""Default constructor"""
		self.Space = None #Space of the vectors
		self.vecSet = [] #List of vectors of the set
		return

	def __del__(self):
		"""Default destructor"""
		return

	def setSpace(self,vec_space):
		"""Method to set vector Space of the set"""
		#Checking type
		if(not isinstance(vec_space,vector)): raise TypeError("Input variable is not a vector")
		#Checking if space has already been assigned
		if(self.Space != None): raise AttributeError("Vector Space of the set already provided")
		self.Space = vec_space.cloneSpace() #setting space
		return

	def append(self,vec_in,copy=True):
		"""Method to add vector to the set"""
		#Check if setSpace was set
		if(self.Space == None): raise AttributeError("ERROR! Vector Set has no Space; call sepSpace first!")
		#Checking type
		if(not isinstance(vec_in,vector)): raise TypeError("ERROR! Input variable is not a vector")
		#Checking dimensionality
		if(not self.Space.checkSame(vec_in)): raise ValueError("ERROR! Provided vector not in the same Space of the vector set")
		if(copy):
			#Copying input vector
			self.vecSet.append(vec_in.clone())
		else:
			#Copying by reference
			self.vecSet.append(vec_in)
		return

	def writeSet(self,filename):
		"""Method to write to SEPlib file (by default it appends vectors to file)"""
		#Check input file to append or to write a new file
# 		if(os.path.isfile(filename)):
# 			ndims = sep_util.get_num_axes(filename)
# 			naxis= sep_util.get_axes(filename)
# 			if(self.Space.ndims < ndims): ndims -= 1
# 			#Checking for space matching
# 			if(self.Space.naxis != tuple(ii[0] for ii in naxis[:ndims])): raise ValueError("File %s does not conform set Space (i.e., number of axis elements). Cannot be used to append vectors"%(filename))
		#Writing binary
		for ivector in self.vecSet:
			ivector.writeVec(filename,mode='a')
		#Remove vectors within the set from memory
		del self.vecSet
		self.vecSet=[]
		return


class superVector(vector):
	"""Column-wise concatenation of vectors [vec1^T vec2^T]^T"""
	def __init__(self,vec1,vec2):
		"""SuperVector constructor"""
		self.vec1=vec1
		self.vec2=vec2
		return

	def __del__(self):
		"""SuperVector destructor"""
		del self.vec1
		del self.vec2
		return

	def norm(self,N=2):
		"""Function to compute vector N-norm"""
		norm = np.power(self.vec1.norm(N),N)
		norm += np.power(self.vec2.norm(N),N)
		return np.power(norm,1./N)

	def zero(self):
		"""Function to zero out a vector"""
		self.vec1.zero()
		self.vec2.zero()
		return

	def scale(self,sc):
		"""Function to scale a vector"""
		self.vec1.scale(sc)
		self.vec2.scale(sc)
		return

	def rand(self,snr=1.0):
		"""Function to randomize a vector"""
		self.vec1.rand(snr)
		self.vec2.rand(snr)
		return

	def clone(self):
		"""Function to clone (deep copy) a vector"""
		return superVector(self.vec1.clone(),self.vec2.clone())

	def cloneSpace(self):
		"""Function to clone vector space"""
		return superVector(self.vec1.cloneSpace(),self.vec2.cloneSpace())

	def cloneVector(self):
		"""Function to clone/allocate vector from vector space"""
		return superVector(self.vec1.cloneVector(),self.vec2.cloneVector())

	def checkSame(self,vec_in):
		"""Function to check to make sure the vectors exist in the same space"""
		#Checking type
		if(type(vec_in) is not superVector): raise TypeError("Input variable is not a superVector")
		checkspace1 = self.vec1.checkSame(vec_in.vec1)
		checkspace2 = self.vec2.checkSame(vec_in.vec2)
		#Checking space
		if(not checkspace1): print("WARNING! First vector component not in the same space vec1_component1 = %s; vec2_component1 = %s"%(self.vec1.naxis,vec_in.vec1.naxis))
		if(not checkspace2): print("WARNING! First vector component not in the same space vec1_component2 = %s; vec2_component2 = %s"%(self.vec2.naxis,vec_in.vec2.naxis))
		return (checkspace1 and checkspace2)

	def writeVec(self,filename,mode='w'):
		"""Function to write vector to file"""
		#Writing two files for the two components
		filename_comp1 = "".join(filename.split('.')[:-1])+"_comp1.H"
		filename_comp2 = "".join(filename.split('.')[:-1])+"_comp2.H"
		#Writing files
		self.vec1.writeVec(filename_comp1,mode)
		self.vec2.writeVec(filename_comp2,mode)
		return

	#Combination of different vectors
	def copy(self,vec_in):
		"""Function to copy vector from input vector"""
		#Checking type
		if(type(vec_in) is not superVector): raise TypeError("Input variable is not a superVector")
		#Checking dimensionality
		if(not self.checkSame(vec_in)): raise ValueError("ERROR! Dimensionality mismatching between given superVectors")
		self.vec1.copy(vec_in.vec1)
		self.vec2.copy(vec_in.vec2)
		return

	def scaleAdd(self,vec_in,sc1=1.0,sc2=1.0):
		"""Function to scale two vectors and add them to the first one"""
		#Checking type
		if(type(vec_in) is not superVector): raise TypeError("Input variable is not a superVector")
		#Checking dimensionality
		if(not self.checkSame(vec_in)): raise ValueError("ERROR! Dimensionality mismatching between given superVectors")
		self.vec1.scaleAdd(vec_in.vec1,sc1,sc2)
		self.vec2.scaleAdd(vec_in.vec2,sc1,sc2)
		return

	def dot(self,vec_in):
		"""Function to compute dot product between two vectors"""
		#Checking type
		if(type(vec_in) is not superVector): raise TypeError("Input variable is not a superVector")
		#Checking dimensionality
		if(not self.checkSame(vec_in)): raise ValueError("ERROR! Dimensionality mismatching between given superVectors")
		dot = self.vec1.dot(vec_in.vec1)
		dot += self.vec2.dot(vec_in.vec2)
		return dot

	def multiply(self,vec_in):
		"""Function to multiply element-wise two vectors"""
		#Checking type
		if(type(vec_in) is not superVector): raise TypeError("Input variable is not a superVector")
		#Checking dimensionality
		if(not self.checkSame(vec_in)): raise ValueError("ERROR! Dimensionality mismatching between given superVectors")
		self.vec1.multiply(vec_in.vec1)
		self.vec2.multiply(vec_in.vec2)
		return

	def isDifferent(self,vec_in):
		"""Function to check if two vectors are identical"""
		#Checking type
		if(type(vec_in) is not superVector): raise TypeError("Input variable is not a superVector")
		return (self.vec1.isDifferent(vec_in.vec1) and self.vec2.isDifferent(vec_in.vec2))


class vectorIC(vector):
	"""In-core python vector class"""
	def __init__(self,input):
		"""VectorIC constructor: arr=np.array
		   The naxis variable is a tuple that specifies the elements in each dimension starting from the fastest to the slowest memory wise
		   The array contained in this class are stored with C memory order (i.e., row-wise sorting)"""
		#Verify that input is a numpy array or header file or vectorOC
		if(isinstance(input,vectorOC)):
			#VectorOC passed to constructor
			self.arr,self.ax_info = sep_util.read_file(input.vecfile)
		elif(isinstance(input,str)):
			#Header file passed to constructor
			self.arr,self.ax_info = sep_util.read_file(input)
		elif(isinstance(input,np.ndarray)):
			#Numpy array passed to constructor
			if(np.isfortran(input)): raise TypeError("ERROR! Input array not a C contiguous array!")
			self.arr = np.array(input,copy=False)
			self.ax_info = None
		elif(isinstance(input,vectorSEP)):
			#VectorSEP passed to constructor
			self.arr = np.array(input.vec)
			self.ax_info = input.ax_info
		elif(isinstance(input,tuple)):
			#Tuple size passed to constructor
			self.arr = np.zeros(tuple(reversed(input)))
			self.ax_info = None
		else:
			#Not supported type
			raise ValueError("ERROR! Input variable not currently supported!")
		#Number of elements per axis (tuple). Checking also the memory order
		self.naxis = self.arr.shape #If fortran the first axis is the "fastest"
		if(not np.isfortran(self.arr)): self.naxis=tuple(reversed(self.naxis)) #If C last axis is the "fastest"
		if(self.naxis==()): self.naxis = (1,) #To fix problem with scalar within a vectorIC
		#Number of axes integer
		self.ndims = len(self.naxis)
		#Total number of elements
		self.size = self.arr.size
		return

	def __del__(self):
		"""VectorIC destructor"""
		del self.arr
		return

	def norm(self,N=2):
		"""Function to compute vector N-norm using Numpy"""
		return	np.linalg.norm(self.arr.flatten(),ord=N)

	def zero(self):
		"""Function to zero out a vector"""
		self.arr.fill(0)
		return

	def scale(self,sc):
		"""Function to scale a vector"""
		self.arr*=sc
		return

	def rand(self,snr=1.0):
		"""Fill vector with random number (~U[1,-1]) with a given SNR"""
		rms = np.sqrt(np.mean(np.square(self.arr)))
		amp_noise = 1.0
		if(rms != 0.): amp_noise = math.sqrt(3.0/snr)*rms #sqrt(3*Power_signal/SNR)
		del self.arr
		self.arr = amp_noise * (2.0 * np.random.random(tuple(reversed(self.naxis))) - 1.0)
		return

	def clone(self):
		"""Function to clone (deep copy) a vector"""
		return deepcopy(self)

	def cloneSpace(self):
		"""Function to clone vector space only (vector without actual vector array by using empty array of size 0)"""
		arr = np.empty(0)
		vec_space = vectorIC(arr)
		#Cloning space of input vector
		vec_space.naxis = self.naxis
		vec_space.ndims = self.ndims
		vec_space.size = self.size
		return vec_space

	def cloneVector(self):
		"""Function to clone/allocate vector from vector space"""
		vec_clone = self.clone() #Deep clone of vector
		#Checking if a vector space was provided
		if(vec_clone.arr.size == 0):
			vec_clone.arr = np.zeros(tuple(reversed(vec_clone.naxis)))
		return vec_clone

	def checkSame(self,vec2):
		"""Function to check dimensionality of vectors"""
		return self.naxis==vec2.naxis

	def writeVec(self,filename,mode='w'):
		"""Function to write vector to file"""
		#Check writing mode
		if(not mode in 'wa'): raise ValueError("Mode must be appending 'a' or writing 'w' ")
		#writing header/pointer file if not present and not append mode
		if(not (os.path.isfile(filename) and mode in 'a')):
			binfile = sep_util.datapath+filename.split('/')[-1]+'@'
			with open(filename,mode) as fid:
				#Writing axis info
				if(self.ax_info):
					for ii,ax_info in enumerate(self.ax_info):
						ax_id = ii + 1
						fid.write("n%s=%s o%s=%s d%s=%s label%s='%s'\n"%(ax_id,ax_info[0],ax_id,ax_info[1],ax_id,ax_info[2],ax_id,ax_info[3]))
				else:
					for ii,n_axis in enumerate(self.naxis):
						ax_id = ii + 1
						fid.write("n%s=%s o%s=0.0 d%s=1.0 \n"%(ax_id,n_axis,ax_id,ax_id))
				#Writing last axis for allowing appending (unless we are dealing with a scalar)
				if(self.naxis != (1,)):
					ax_id = self.ndims+1
					fid.write("n%s=%s o%s=0.0 d%s=1.0 \n"%(ax_id,1,ax_id,ax_id))
				fid.write("in='%s'\n"%(binfile))
			fid.close()
		else:
			binfile = sep_util.get_binary(filename)
			if(mode in 'a'):
				axes = sep_util.get_axes(filename)
				#Number of vectors already present in the file
				if(self.naxis == (1,)):
					n_vec = axes[0][0]
					append_dim = self.ndims
				else:
					n_vec = axes[self.ndims][0]
					append_dim = self.ndims+1
				with open(filename,mode) as fid:
					fid.write("n%s=%s o%s=0.0 d%s=1.0 \n"%(append_dim,n_vec+1,append_dim,append_dim))
				fid.close()
		#Writing binary file
		with open(binfile,mode+'b') as fid:
			#Writing big-ending floating point number
			if(np.isfortran(self.arr)): #Forcing column-wise binary writing
				self.arr.flatten('F').astype('>f').tofile(fid)
			else:
				self.arr.astype('>f').tofile(fid)
		fid.close()
		return

	def copy(self,vec2):
		"""Function to copy vector from input vector"""
		#Checking whether the input is a vector or not
		if(not isinstance(vec2,vectorIC)): raise TypeError("ERROR! Provided input vector not a vectorIC!")
		#Checking dimensionality
		if(not self.checkSame(vec2)): raise ValueError("ERROR! Dimensionality not equal: vec1 = %s; vec2 = %s"%(self.naxis,vec2.naxis))
		#Element-wise copy of the input array
		self.arr[:]=vec2.arr
		return

	def scaleAdd(self,vec2,sc1=1.0,sc2=1.0):
		"""Function to scale a vector"""
		#Checking whether the input is a vector or not
		if(not isinstance(vec2,vectorIC)): raise TypeError("ERROR! Provided input vector not a vectorIC!")
		#Checking dimensionality
		if(not self.checkSame(vec2)): raise ValueError("ERROR! Dimensionality not equal: vec1 = %s; vec2 = %s"%(self.naxis,vec2.naxis))
		#Performing scaling and addition
		self.arr=sc1*self.arr+sc2*vec2.arr
		return

	def dot(self,vec2):
		"""Function to compute dot product between two vectors"""
		#Checking whether the input is a vector or not
		if(not isinstance(vec2,vectorIC)): raise TypeError("ERROR! Provided input vector not a vectorIC!")
		#Checking size (must have same number of elements)
		if(self.size!=vec2.size): raise ValuError("ERROR! Vector size mismatching: vec1 = %s; vec2 = %s"%(self.size,vec2.size))
		#Checking dimensionality
		if(not self.checkSame(vec2)): raise ValueError("ERROR! Dimensionality not equal: vec1 = %s; vec2 = %s"%(self.naxis,vec2.naxis))
		return np.dot(self.arr.flatten(),vec2.arr.flatten())

	def multiply(self,vec2):
		"""Function to multiply element-wise two vectors"""
		#Checking whether the input is a vector or not
		if(not isinstance(vec2,vectorIC)): raise TypeError("ERROR! Provided input vector not a vectorIC!")
		#Checking size (must have same number of elements)
		if(self.size!=vec2.size): raise ValuError("ERROR! Vector size mismatching: vec1 = %s; vec2 = %s"%(self.size,vec2.size))
		#Checking dimensionality
		if(not self.checkSame(vec2)): raise ValueError("ERROR! Dimensionality not equal: vec1 = %s; vec2 = %s"%(self.naxis,vec2.naxis))
		#Performing element-wise multiplication
		self.arr=np.multiply(self.arr,vec2.arr)
		return

	def isDifferent(self,vec2):
		"""Function to check if two vectors are identical using built-in hash function"""
		#Checking whether the input is a vector or not
		if(not isinstance(vec2,vectorIC)): raise TypeError("ERROR! Provided input vector not a vectorIC!")
		#Using Hash table for python2 and numpy built-in function array_equal otherwise
		if(version_info[0]==2):
			#First make both array buffers read-only
			self.arr.flags.writeable = False
			vec2.arr.flags.writeable = False
			chcksum1=hash(self.arr.data)
			chcksum2=hash(vec2.arr.data)
			#Remake array buffers writable
			self.arr.flags.writeable = True
			vec2.arr.flags.writeable = True
			isDiff=(chcksum1!=chcksum2)
		else:
			isDiff=(not np.array_equal(self.arr,vec2.arr))
		return isDiff


class vectorOC(vector):
	"""Out-of-core python vector class"""
	def __init__(self,input):
		"""VectorOC constructor: input= numpy array, header file, vectorIC"""
		#Verify that input is a numpy array or header file or vectorOC
		if(isinstance(input,vectorIC)):
			#VectorIC passed to constructor
			#Placing temporary file into datapath folder
			tmp_vec = sep_util.datapath+"tmp_vectorOC"+str(int(time.time()*1000000))+".H"
			sep_util.write_file(tmp_vec,input.arr,input.ax_info)
			self.vecfile = tmp_vec #Assigning internal vector array
			#Removing header file? (Default behavior is to remove temporary file)
			self.remove_file = True
		elif(isinstance(input,np.ndarray)):
			#Numpy array passed to constructor
			tmp_vec = sep_util.datapath+"tmp_vectorOC"+str(int(time.time()*1000000))+".H"
			sep_util.write_file(tmp_vec,input)
			self.vecfile = tmp_vec #Assigning internal vector array
			#Removing header file? (Default behavior is to remove temporary file)
			self.remove_file = True
		elif(isinstance(input,vectorSEP)):
			#VectorSEP passed to constructor
			tmp_vec = sep_util.datapath+"tmp_vectorOC"+str(int(time.time()*1000000))+".H"
			arr = np.array(input.vec,copy=False)
			sep_util.write_file(tmp_vec,arr,input.ax_info)
			self.vecfile = tmp_vec #Assigning internal vector array
			#Removing header file? (Default behavior is to remove temporary file)
			self.remove_file = True
		elif(isinstance(input,str)):
			#Header file passed to constructor
			self.vecfile = input #Assigning internal vector array
			#Removing header file? (Default behavior is to preserve user file)
			self.remove_file = False
		else:
			#Not supported type
			raise ValueError("ERROR! Input variable not currently supported!")
		#Assigning binary file pointer
		self.binfile = sep_util.get_binary(self.vecfile)
		#Number of axes integer
		self.ndims = sep_util.get_num_axes(self.vecfile)
		#Number of elements per axis (tuple)
		axes_info = sep_util.get_axes(self.vecfile)
		axis_elements =tuple([ii[0] for ii in axes_info[:self.ndims]])
		self.naxis = axis_elements
		self.size = np.product(self.naxis)
		return

	def __del__(self):
		"""VectorOC destructor"""
		if(self.remove_file):
			#Removing both header and binary files (using os.system to make module compatible with python3.5)
			os.system("rm -f %s %s"%(self.vecfile,self.binfile))
		return

	def norm(self,N=2):
		"""Function to compute vector N-norm"""
		if(N != 2): raise NotImplementedError("Norm different than L2 not currently supported")
		#Running Solver_ops to compute norm value
		find = re_dpr.search(sys_util.RunShellCmd("Solver_ops file1=%s op=dot"%(self.vecfile),get_stat=False)[0])
		if find:
			return np.sqrt(float(find.group(1)))
		else:
			raise ValueError("ERROR! Trouble parsing dot product!")
		return

	def zero(self):
		"""Function to zero out a vector"""
		sys_util.RunShellCmd("head -c %s </dev/zero > %s"%(self.size*4,self.binfile),get_stat=False,get_output=False)
		# sys_util.RunShellCmd("Solver_ops file1=%s op=zero"%(self.vecfile),get_stat=False,get_output=False)
		return

	def scale(self,sc):
		"""Function to scale a vector"""
		import sys_util
		sys_util.RunShellCmd("Solver_ops file1=%s scale1_r=%s op=scale"%(self.vecfile,sc),get_stat=False,get_output=False)
		return

	def rand(self,snr=1.0):
		"""Fill vector with random number (~U[1,-1]) with a given SNR"""
		#Computing RMS amplitude of the vector
		rms=sys_util.RunShellCmd("Attr < %s want=rms param=1 maxsize=5000"%(self.vecfile),get_stat=False)[0]
		rms=float(rms.split("=")[1]) #Standard deviation of the signal
		amp_noise = 1.0
		if(rms != 0.): amp_noise = math.sqrt(3.0/snr)*rms #sqrt(3*Power_signal/SNR)
		#Filling file with random number with the proper scale
		sys_util.RunShellCmd("Noise file=%s rep=1 type=0 var=0.3333333333; Solver_ops file1=%s scale1_r=%s op=scale"%(self.vecfile,self.vecfile,amp_noise),get_stat=False,get_output=False)
		return

	def clone(self):
		"""Function to clone (deep copy) a vector and creating a copy of the associated header file"""
		#First performing a deep copy of the vector
		vec_clone = deepcopy(self)
		#Creating a temporary file with similar name but computer time at the end
		tmp_vec = self.vecfile.split(".H")[0].split("/")[-1] #Getting filename only
		#Placing temporary file into datapath folder
		tmp_vec = sep_util.datapath+tmp_vec+"_clone_"+str(int(time.time()*1000000))+".H"
		tmp_bin = tmp_vec+"@"
		#Copying header and binary files and setting pointers to new file
		copyfile(self.vecfile, tmp_vec) #Copying header
		copyfile(self.binfile, tmp_bin) #Copying binary
		vec_clone.vecfile = tmp_vec
		vec_clone.binfile = tmp_bin
		#"Fixing" header file
		with open(vec_clone.vecfile,"a") as fid:
			fid.write("in='%s\n'"%tmp_bin)
		#By default the clone file is going to be removed once the vector is deleted
		vec_clone.remove_file = True
		return vec_clone

	def cloneSpace(self):
		"""Function to clone vector space only (vector without actual vector binary file by using None values)"""
		vec_space = vectorOC(self.vecfile)
		#Removing header vector file
		vec_space.vecfile = None
		vec_space.binfile = None
		vec_space.remove_file = False
		return vec_space

	def cloneVector(self):
		"""Function to clone/allocate vector from vector space"""
		vec_clone = deepcopy(self) #Deep clone of vector
		#Checking if a vector space was provided
		if(vec_clone.vecfile == None):
			#Creating header and binary files from vector space
			#Placing temporary file into datapath folder
			tmp_vec = sep_util.datapath+"cloneVector_tmp_vector"+str(int(time.time()*1000000))+".H"
			axis_file = ""
			for iaxis,naxis in enumerate(vec_clone.naxis):
				axis_file += "n%s=%s "%(iaxis+1,naxis)
			#Creating temporary vector file
			cmd="Spike %s | Add scale=0.0 > %s"%(axis_file,tmp_vec)
			sys_util.RunShellCmd(cmd,get_stat=False,get_output=False)
			vec_clone.vecfile = tmp_vec
			vec_clone.binfile = sep_util.get_binary(vec_clone.vecfile)
			#Removing header file?
			vec_clone.remove_file = True
		return vec_clone

	def checkSame(self,vec2):
		"""Function to check dimensionality of vectors"""
		return self.naxis==vec2.naxis

	def writeVec(self,filename,mode='w'):
		"""Function to write vector to file"""
		#Check writing mode
		if(not mode in 'wa'): raise ValueError("Mode must be appending 'a' or writing 'w' ")
		#writing header/pointer file if not present and not append mode
		if(not (os.path.isfile(filename) and mode in 'a')):
			binfile = sep_util.datapath+filename.split('/')[-1]+'@'
			#Copying SEPlib header file
			copyfile(self.vecfile, filename)
			#Substituting binary file
			with open(filename,'a') as fid:
				fid.write("\nin='%s'\n"%(binfile))
			fid.close()
		else:
			binfile = sep_util.get_binary(filename)
			if(mode in 'a'):
				axes = sep_util.get_axes(filename)
				#Number of vectors already present in the file
				if(self.naxis == (1,)):
					n_vec = axes[0][0]
					append_dim = self.ndims
				else:
					n_vec = axes[self.ndims][0]
					append_dim = self.ndims+1
				with open(filename,mode) as fid:
					fid.write("n%s=%s o%s=0.0 d%s=1.0 \n"%(append_dim,n_vec+1,append_dim,append_dim))
				fid.close()
		#Writing or Copying binary file
		if(not (os.path.isfile(binfile) and mode in 'a')):
			copyfile(self.binfile,binfile)
		else:
			#Writing file if
			with open(binfile,mode+'b') as fid, open(self.binfile,'rb') as fid_toread:
				while True:
					data = fid_toread.read(sys_util.BUF_SIZE)
					if not data: break
					fid.write(data)
			fid.close(); fid_toread.close()
		return

	def copy(self,vec2):
		"""Function to copy vector from input vector"""
		#Checking whether the input is a vector or not
		if(not isinstance(vec2,vectorOC)): raise TypeError("ERROR! Provided input vector not a vectorOC!")
		#Checking dimensionality
		if(not self.checkSame(vec2)): raise ValueError("ERROR! Vector dimensionality mismatching: vec1 = %s; vec2 = %s"%(self.naxis,vec2.naxis))
		#Copy binary file of input vector
		copyfile(vec2.binfile, self.binfile) #Copying binary
		return

	def scaleAdd(self,vec2,sc1=1.0,sc2=1.0):
		"""Function to scale a vector"""
		#Checking whether the input is a vector or not
		if(not isinstance(vec2,vectorOC)): raise TypeError("ERROR! Provided input vector not a vectorOC!")
		#Checking dimensionality
		if(not self.checkSame(vec2)): raise ValueError("ERROR! Vector dimensionality mismatching: vec1 = %s; vec2 = %s"%(self.naxis,vec2.naxis))
		#Performing scaling and addition
		cmd="Solver_ops file1=%s scale1_r=%s file2=%s scale2_r=%s op=scale_addscale"%(self.vecfile,sc1,vec2.vecfile,sc2)
		sys_util.RunShellCmd(cmd,get_stat=False,get_output=False)
		return

	def dot(self,vec2):
		"""Function to compute dot product between two vectors"""
		#Checking whether the input is a vector or not
		if(not isinstance(vec2,vectorOC)): raise TypeError("ERROR! Provided input vector not a vectorOC!")
		#Checking size (must have same number of elements)
		if(self.size!=vec2.size): raise ValueError("ERROR! Vector size mismatching: vec1 = %s; vec2 = %s"%(self.size,vec2.size))
		#Checking dimensionality
		if(not self.checkSame(vec2)): raise ValueError("ERROR! Vector dimensionality mismatching: vec1 = %s; vec2 = %s"%(self.naxis,vec2.naxis))
		#Running Solver_ops to compute norm value
		cmd="Solver_ops file1=%s file2=%s op=dot"%(self.vecfile,vec2.vecfile)
		find = re_dpr.search(sys_util.RunShellCmd(cmd,get_stat=False)[0])
		if find:
			return float(find.group(1))
		else:
			raise ValueError("ERROR! Trouble parsing dot product!")
		return float(out_dot)

	def multiply(self,vec2):
		"""Function to multiply element-wise two vectors"""
		#Checking whether the input is a vector or not
		if(not isinstance(vec2,vectorOC)): raise TypeError("ERROR! Provided input vector not a vectorOC!")
		#Checking size (must have same number of elements)
		if(self.size!=vec2.size): raise ValueError("ERROR! Vector size mismatching: vec1 = %s; vec2 = %s"%(self.size,vec2.size))
		#Checking dimensionality
		if(not self.checkSame(vec2)): raise ValueError("ERROR! Vector dimensionality mismatching: vec1 = %s; vec2 = %s"%(self.naxis,vec2.naxis))
		#Performing scaling and addition
		cmd="Solver_ops file1=%s file2=%s op=multiply"%(self.vecfile,vec2.vecfile)
		sys_util.RunShellCmd(cmd,get_stat=False,get_output=False)
		return

	def isDifferent(self,vec2):
		"""Function to check if two vectors are identical using M5 hash scheme"""
		#Checking whether the input is a vector or not
		if(not isinstance(vec2,vectorOC)): raise TypeError("ERROR! Provided input vector not a vectorOC!")
		hashmd5_vec1=sys_util.hashfile(self.binfile)
		hashmd5_vec2=sys_util.hashfile(vec2.binfile)
		return (hashmd5_vec1!=hashmd5_vec2)

class vectorSEP(vector):
	"""SEP vector class based on Bob's library"""
	
	def __init__(self,input):
		"""Creating a vectorSEP using SepVector.Vector class"""
		if(isinstance(input,Hypercube.hypercube)):
			#Using an hypercube to create vector
			self.vec=SepVector.getSepVector(input)
		elif(isinstance(input,pySepVector.Vector)):
			#Using SepVector directly
			self.vec=input
		elif(isinstance(input,vectorIC)):
			#Using vectorIC
			hyper_in = Hypercube.hypercube(axes=[Hypercube.axis(n=ii) for ii in input.naxis])
			self.vec=SepVector.getSepVector(hyper_in,input.arr)
		elif(isinstance(input,vectorOC)):
			#Using vectorOC
			arr,ax_info = sep_util.read_file(input.vecfile)
			hyper_in = Hypercube.hypercube(axes=[Hypercube.axis(n=axis[0],o=axis[1],d=axis[2],label=axis[3]) for axis in ax_info[:sep_util.get_num_axes(input.vecfile)]])
			self.vec=SepVector.getSepVector(hyper_in,arr)
			del arr, ax_info
		elif(isinstance(input,np.ndarray)):
			#Using Numpy Array
			if(not np.isfortran(input)): 
				shape=tuple(reversed(input.shape)) #If C last axis is the "fastest"
			else:
				shape = input.shape
			hyper_in = Hypercube.hypercube(axes=[Hypercube.axis(n=ii) for ii in shape])
			self.vec=SepVector.getSepVector(hyper_in,input)
		elif(isinstance(input,str)):
			#Using SEP header file
			arr,ax_info = sep_util.read_file(input)
			hyper_in = Hypercube.hypercube(axes=[Hypercube.axis(n=axis[0],o=axis[1],d=axis[2],label=axis[3]) for axis in ax_info[:sep_util.get_num_axes(input)]])
			self.vec=SepVector.getSepVector(hyper_in,arr)
			del arr, ax_info
		elif(isinstance(input,tuple)):
			#Using an axis tuple
			self.vec=SepVector.getSepVector(Hypercube.hypercube(axes=[Hypercube.axis(n=ii) for ii in input]))
		else:
			#Not supported type
			raise ValueError("ERROR! Input variable not currently supported!")
		hyper = self.vec.getHyper()
		#Number of axes
		self.ndims = hyper.getNdim()
		#Number of elements per axis 
		self.naxis=tuple([hyper.getAxis(ii).n for ii in range(1,self.ndims+1)])
		#Total number of elements
		self.size = hyper.getN123()
		#Creating ax_info
		self.ax_info=[[hyper.getAxis(ii).n,hyper.getAxis(ii).o,hyper.getAxis(ii).d,hyper.getAxis(ii).label] for ii in range(1,self.ndims+1)]
		return
	
	def norm(self,N=2):
		"""Function to compute vector N-norm using Numpy"""
		if(N == 1):
			return self.vec.norm(N)
		elif (N == 2):
			return np.sqrt(self.vec.norm(N))
		else:
			raise NotImplementedError("ERROR! Norm different than L1 and L2 not currently supported!")
		return

	def zero(self):
		"""Function to zero out a vector"""
		self.vec.zero()
		return

	def scale(self,sc):
		"""Function to scale a vector"""
		self.vec.scale(sc)
		return

	def rand(self,snr=1.0):
		"""Fill vector with random number"""
		self.vec.rand()
		return

	def clone(self):
		"""Function to clone (deep copy) a vector"""
		return vectorSEP(self.vec.clone())

	def cloneSpace(self):
		"""Function to clone vector space only"""
		return vectorSEP(self.vec.cloneSpace())

	def cloneVector(self):
		"""Function to clone/allocate vector from vector space by getting the Hypercube"""
		return vectorSEP(self.vec.getHyper())

	def checkSame(self,vec2):
		"""Function to check dimensionality of vectors"""
		#Checking whether the input is a vector or not
		if(not isinstance(vec2,vectorSEP)): raise TypeError("ERROR! Provided input vector not a vectorSEP!")
		return self.vec.checkSame(vec2.vec)

	def writeVec(self,filename,mode='w'):
		"""Function to write vector to file"""
		#Check writing mode
		if(not mode in 'wa'): raise ValueError("Mode must be appending 'a' or writing 'w' ")
		#writing header/pointer file if not present and not append mode
		if(not (os.path.isfile(filename) and mode in 'a')):
			binfile = sep_util.datapath+filename.split('/')[-1]+'@'
			with open(filename,mode) as fid:
				#Writing axis info
				if(self.ax_info):
					for ii,ax_info in enumerate(self.ax_info):
						ax_id = ii + 1
						fid.write("n%s=%s o%s=%s d%s=%s label%s='%s'\n"%(ax_id,ax_info[0],ax_id,ax_info[1],ax_id,ax_info[2],ax_id,ax_info[3]))
				else:
					for ii,n_axis in enumerate(self.naxis):
						ax_id = ii + 1
						fid.write("n%s=%s o%s=0.0 d%s=1.0 \n"%(ax_id,n_axis,ax_id,ax_id))
				#Writing last axis for allowing appending (unless we are dealing with a scalar)
				if(self.naxis != (1,)):
					ax_id = self.ndims+1
					fid.write("n%s=%s o%s=0.0 d%s=1.0 \n"%(ax_id,1,ax_id,ax_id))
				fid.write("in='%s'\n"%(binfile))
			fid.close()
		else:
			binfile = sep_util.get_binary(filename)
			if(mode in 'a'):
				axes = sep_util.get_axes(filename)
				#Number of vectors already present in the file
				if(self.naxis == (1,)):
					n_vec = axes[0][0]
					append_dim = self.ndims
				else:
					n_vec = axes[self.ndims][0]
					append_dim = self.ndims+1
				with open(filename,mode) as fid:
					fid.write("n%s=%s o%s=0.0 d%s=1.0 \n"%(append_dim,n_vec+1,append_dim,append_dim))
				fid.close()
		#Writing binary file
		with open(binfile,mode+'b') as fid:
			#Creating np_array from vectorSEP
			arr = np.array(self.vec)
			#Writing big-ending floating point number
			arr.astype('>f').tofile(fid)
		fid.close()
		return

	def copy(self,vec2):
		"""Function to copy vector from input vector"""
		#Checking whether the input is a vector or not
		if(not isinstance(vec2,vectorSEP)): raise TypeError("ERROR! Provided input vector not a vectorSEP!")
		#Checking dimensionality
		if(not self.checkSame(vec2)): raise ValueError("ERROR! Dimensionality not equal: vec1 = %s; vec2 = %s"%(self.naxis,vec2.naxis))
		#Element-wise copy of the input array
		selfvec_np=np.array(self.vec,copy=False)
		vec2_np=np.array(vec2.vec,copy=False)
		selfvec_np[:]=vec2_np
		return

	def scaleAdd(self,vec2,sc1=1.0,sc2=1.0):
		"""Function to scale a vector"""
		#Checking whether the input is a vector or not
		if(not isinstance(vec2,vectorSEP)): raise TypeError("ERROR! Provided input vector not a vectorSEP!")
		#Checking dimensionality
		if(not self.checkSame(vec2)): raise ValueError("ERROR! Dimensionality not equal: vec1 = %s; vec2 = %s"%(self.naxis,vec2.naxis))
		#Performing scaling and addition
		self.vec.scaleAdd(vec2.vec,sc1,sc2)
		return

	def dot(self,vec2):
		"""Function to compute dot product between two vectors"""
		#Checking whether the input is a vector or not
		if(not isinstance(vec2,vectorSEP)): raise TypeError("ERROR! Provided input vector not a vectorSEP!")
		#Checking size (must have same number of elements)
		if(self.size!=vec2.size): raise ValuError("ERROR! Vector size mismatching: vec1 = %s; vec2 = %s"%(self.size,vec2.size))
		#Checking dimensionality
		if(not self.checkSame(vec2)): raise ValueError("ERROR! Dimensionality not equal: vec1 = %s; vec2 = %s"%(self.naxis,vec2.naxis))
		return self.vec.dot(vec2.vec)

	def multiply(self,vec2):
		"""Function to multiply element-wise two vectors"""
		#Checking whether the input is a vector or not
		if(not isinstance(vec2,vectorSEP)): raise TypeError("ERROR! Provided input vector not a vectorSEP!")
		#Checking size (must have same number of elements)
		if(self.size!=vec2.size): raise ValuError("ERROR! Vector size mismatching: vec1 = %s; vec2 = %s"%(self.size,vec2.size))
		#Checking dimensionality
		if(not self.checkSame(vec2)): raise ValueError("ERROR! Dimensionality not equal: vec1 = %s; vec2 = %s"%(self.naxis,vec2.naxis))
		#Performing element-wise multiplication
		self.vec.mult(vec2.vec)
		return

	def isDifferent(self,vec2):
		"""Function to check if two vectors are identical using built-in hash function"""
		#Checking whether the input is a vector or not
		if(not isinstance(vec2,vectorSEP)): raise TypeError("ERROR! Provided input vector not a vectorSEP!")
		return self.vec.isDifferent(vec2.vec)
		
	
	
	
	
	
	
	
