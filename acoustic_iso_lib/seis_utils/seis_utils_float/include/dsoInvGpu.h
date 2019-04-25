#ifndef DSO_INV_GPU_H
#define DSO_INV_GPU_H 1

#include "operator.h"
#include "float3DReg.h"
#include <vector>
#include <omp.h>

using namespace SEP;

class dsoInvGpu : public Operator<SEP::float3DReg, SEP::float3DReg> {

	private:

        int _nz, _nx, _nExt, _hExt, _fat;
        float _zeroShift;

	public:

		/* Overloaded constructors */
		dsoInvGpu(int nz, int nx, int nExt, int _fat, float zeroShift);

		/* FWD - ADJ */
		void forward(const bool add, const std::shared_ptr<float3DReg> model, std::shared_ptr<float3DReg> data) const;
		void adjoint(const bool add, std::shared_ptr<float3DReg> model, const std::shared_ptr<float3DReg> data) const;

		/* Destructor */
		~dsoInvGpu(){};

        /* Mutators */
        void setZeroShift(float zeroShift){_zeroShift = zeroShift;}

};

#endif
