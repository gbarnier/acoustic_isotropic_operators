#ifndef DATA_TAPER_DOUBLE_H
#define DATA_TAPER_DOUBLE_H 1

#include "operator.h"
#include "double1DReg.h"
#include "double3DReg.h"
#include <string>

using namespace SEP;

class dataTaperDouble : public Operator<SEP::double3DReg, SEP::double3DReg>{

	private:

		double _maxOffset, _exp, _taperWidth;
		double _xMinRec, _xMaxRec, _dRec;
		double _xMinShot, _xMaxShot, _dShot;
		int _nRec, _nShot;
		std::string _muteType;
		std::shared_ptr<double3DReg> _taperMask;
		std::shared_ptr<SEP::hypercube> _dataHyper;

	public:

		/* Overloaded constructor */
		// dataTaperDouble(double maxOffset, double exp, double taperWidth, std::shared_ptr<SEP::hypercube> dataHyper, std::string muteType);
		dataTaperDouble(double maxOffset, double exp, double taperWidth, std::shared_ptr<SEP::hypercube> dataHyper, std::string muteType);

		/* Destructor */
		~dataTaperDouble(){};

  		/* Forward / Adjoint */
  		virtual void forward(const bool add, const std::shared_ptr<double3DReg> model, std::shared_ptr<double3DReg> data) const;
 		virtual void adjoint(const bool add, std::shared_ptr<double3DReg> model, const std::shared_ptr<double3DReg> data) const;

		/* Mutator */
		void setTaperMask(std::shared_ptr<double3DReg> taperMask) {_taperMask = taperMask;}

		/* Accessor */
		std::shared_ptr<double3DReg> getTaperMask() {return _taperMask;}

};

#endif
