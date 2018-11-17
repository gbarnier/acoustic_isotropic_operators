#ifndef SECOND_TIME_DERIVATIVE_H
#define SECOND_TIME_DERIVATIVE_H 1

#include "operator.h"
#include "double2DReg.h"

using namespace SEP;

class secondTimeDerivative : public Operator<SEP::double2DReg, SEP::double2DReg>
{
	private:
	
		int _nt;
		double _dt2;
		
	public:

		/* Overloaded constructor */ 	
		secondTimeDerivative(int nt, double dt);
	
		/* Destructor */ 	
		~secondTimeDerivative(){};
	
  		/* Forward / Adjoint */
  		virtual void forward(const bool add, const std::shared_ptr<double2DReg> model, std::shared_ptr<double2DReg> data) const;
 		virtual void adjoint(const bool add, std::shared_ptr<double2DReg> model, const std::shared_ptr<double2DReg> data) const;
 		
};

#endif
