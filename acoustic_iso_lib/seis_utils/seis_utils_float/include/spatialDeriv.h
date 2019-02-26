#ifndef SPATIAL_DERIV_H
#define SPATIAL_DERIV_H 1

#include "operator.h"
#include "float2DReg.h"
#include <vector>
#include <omp.h>

using namespace SEP;

class zGrad : public Operator<SEP::float2DReg, SEP::float2DReg> {

	private:
        int _fat;

	public:
		/* Overloaded constructors */
		zGrad(int fat);

		/* FWD - ADJ */
		void forward(const bool add, const std::shared_ptr<float2DReg> model, std::shared_ptr<float2DReg> data) const;
		void adjoint(const bool add, std::shared_ptr<float2DReg> model, const std::shared_ptr<float2DReg> data) const;

		/* Destructor */
		~zGrad(){};

};

class xGrad : public Operator<SEP::float2DReg, SEP::float2DReg> {

	private:
        int _fat;

	public:
		/* Overloaded constructors */
		xGrad(int fat);

		/* FWD - ADJ */
		void forward(const bool add, const std::shared_ptr<float2DReg> model, std::shared_ptr<float2DReg> data) const;
		void adjoint(const bool add, std::shared_ptr<float2DReg> model, const std::shared_ptr<float2DReg> data) const;

		/* Destructor */
		~xGrad(){};

};

class zxGrad : public Operator<SEP::float2DReg, SEP::float2DReg> {

	private:
        int _fat;

	public:
		/* Overloaded constructors */
		zxGrad(int fat);

		/* FWD - ADJ */
		void forward(const bool add, const std::shared_ptr<float2DReg> model, std::shared_ptr<float2DReg> data) const;
		void adjoint(const bool add, std::shared_ptr<float2DReg> model, const std::shared_ptr<float2DReg> data) const;
};

#endif
