#ifndef SECOND_TIME_DERIVATIVE_H
#define SECOND_TIME_DERIVATIVE_H 1

#include "operator.h"
#include "float2DReg.h"

using namespace SEP;

class secondTimeDerivative : public Operator<SEP::float2DReg, SEP::float2DReg>
{
	private:

		int _nt;
		float _dt2;

	public:

		/* Overloaded constructor */
		secondTimeDerivative(int nt, float dt);

		/* Destructor */
		~secondTimeDerivative(){};

  		/* Forward / Adjoint */
  		virtual void forward(const bool add, const std::shared_ptr<float2DReg> model, std::shared_ptr<float2DReg> data) const;
 		virtual void adjoint(const bool add, std::shared_ptr<float2DReg> model, const std::shared_ptr<float2DReg> data) const;

};

#endif
