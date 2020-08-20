#ifndef SYMES_Z_GRAD_H
#define SYMES_Z_GRAD_H 1

#include "operator.h"
#include "float2DReg.h"
#include <vector>
#include <omp.h>

using namespace SEP;

class SymesZGrad : public Operator<SEP::float3DReg, SEP::float3DReg> {

	private:
        int _fat;

	public:
		/* Overloaded constructors */
		SymesZGrad(int fat);

		/* FWD - ADJ */
		void forward(const bool add, const std::shared_ptr<float3DReg> model, std::shared_ptr<float3DReg> data) const;
		void adjoint(const bool add, std::shared_ptr<float3DReg> model, const std::shared_ptr<float3DReg> data) const;

		/* Destructor */
		~SymesZGrad(){};

};

#endif
