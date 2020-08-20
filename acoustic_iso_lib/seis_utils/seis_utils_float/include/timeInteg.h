#ifndef TIME_INTEG_H
#define TIME_INTEG_H 1

#include "operator.h"
#include <float1DReg.h>
#include "float3DReg.h"

using namespace SEP;

class timeInteg : public Operator<SEP::float3DReg, SEP::float3DReg> {
	private:

		double _alpha;

	public:

		/* Overloaded constructor */
		timeInteg(float dt);

		/* Destructor */
		~timeInteg(){};

  		/* Forward / Adjoint */
  		virtual void forward(const bool add, const std::shared_ptr<float3DReg> model, std::shared_ptr<float3DReg> data) const;
 		virtual void adjoint(const bool add, std::shared_ptr<float3DReg> model, const std::shared_ptr<float3DReg> data) const;

};

#endif
