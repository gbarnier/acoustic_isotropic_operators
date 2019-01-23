import pyOperator

class dsoOp(Operator):
	"""DSO operator"""

    # Constructor
	def __init__(self,domain):
		self.setDomainRange(domain,domain)
		return

	def forward(self,add,model,data):
		self.checkDomainRange(model,data)
		if (not add): data.zero()

		return

	def adjoint(self,add,model,data):
		self.checkDomainRange(model,data)
		if (not add): model.zero()
		return



#
# #!/usr/bin/env python
# #DSO command for extended images
# import sys,os
# sys.path.append(os.environ.get('REPOSITORY')+"/python_solver/python_modules")
# import sep_util as sep
# import numpy as np
#
# if __name__ == '__main__':
#     if(len(sys.argv) == 1):
#         print("NAME \n")
#         print(" DSO - Differential Semblance Operator \n")
#         print("SYNOPSIS")
#         print(" DSO.py input.H output.H \n")
#         print("DESCRIPTION")
#         print(" Applies DSO to input.H and places the result in output.H \n")
#     else:
#         input_file=sys.argv[1]
#         output_file=sys.argv[2]
#         image,ax_info=sep.read_file(input_file)
#         #Image sorting is hx,x,z in C memory
#         nhx=image.shape[0]
#         weight=abs(np.linspace(int(-nhx/2),int(nhx/2),nhx))
#         for ihx,val in enumerate(weight):
#             image[ihx,:,:]*=val
#         sep.write_file(output_file,image,ax_info)
