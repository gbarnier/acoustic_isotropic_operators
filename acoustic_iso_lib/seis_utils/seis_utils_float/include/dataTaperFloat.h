#ifndef DATA_TAPER_FLOAT_H
#define DATA_TAPER_FLOAT_H 1

#include "operator.h"
#include "float3DReg.h"
#include <string>

using namespace SEP;

class dataTaperFloat : public Operator<SEP::float3DReg, SEP::float3DReg>{

	private:

		float _maxOffset, _exp, _taperWidth, _velMute;
		float _xMinRec, _xMaxRec, _dRec;
		float _xMinShot, _xMaxShot, _dShot;
		float _ots, _dts, _tMax;
		float _t0;
		int _nts, _nRec, _nShot;
		std::string _mouveout;
		std::shared_ptr<float3DReg> _taperMask;
		std::shared_ptr<SEP::hypercube> _dataHyper;

	public:

		/* Overloaded constructor */
		dataTaperFloat(float maxOffset, float exp, float taperWidth, std::shared_ptr<SEP::hypercube> dataHyper); // Offset tapering
		dataTaperFloat(float t0, float velMute, float exp, float taperWidth, std::shared_ptr<SEP::hypercube> dataHyper, std::string moveout); // Time tapering

		/* Destructor */
		~dataTaperFloat(){};

  		/* Forward / Adjoint */
  		virtual void forward(const bool add, const std::shared_ptr<float3DReg> model, std::shared_ptr<float3DReg> data) const;
 		virtual void adjoint(const bool add, std::shared_ptr<float3DReg> model, const std::shared_ptr<float3DReg> data) const;

		/* Mutator */
		void setTaperMask(std::shared_ptr<float3DReg> taperMask) {_taperMask = taperMask;}

		/* Accessor */
		std::shared_ptr<float3DReg> getTaperMask() {return _taperMask;}

};

#endif
