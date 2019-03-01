#ifndef INTERP_RBF_1D_H
#define INTERP_RBF_1D_H 1

#include "operator.h"
#include "float1DReg.h"
#include <vector>

using namespace SEP;

class interpRbf1d : public Operator<SEP::float1DReg, SEP::float1DReg> {

	private:

		int _nzModel, _nzData, _scaling, _fat;
		axis _zAxis;
		float _epsilon2;
		std::shared_ptr<float1DReg> _zModel, _zData, _scaleVector;

	public:

		/* Overloaded constructors */
		interpRbf1d(float epsilon, std::shared_ptr<float1DReg> zModel, axis zAxis, int scaling, int fat);

		/* Operator scaling computation */
		void computeScaleVector();

		/* Accessor */
		std::shared_ptr<float1DReg> getScaleVector(){ return _scaleVector;}

		// Accessors
		std::shared_ptr<float1DReg> getZMesh(){ return _zModel;}

		/* Forward / Adjoint */
		void forward(const bool add, const std::shared_ptr<float1DReg> model, std::shared_ptr<float1DReg> data) const;
		void adjoint(const bool add, std::shared_ptr<float1DReg> model, const std::shared_ptr<float1DReg> data) const;

};

#endif
