#ifndef SEISMIC_OERATOR_2D_H
#define SEISMIC_OERATOR_2D_H 1

#include "interpTimeLinTbb.h"
#include "operator.h"
#include "double2DReg.h"
#include "double3DReg.h"
#include "ioModes.h"
#include "operator.h"
#include "fdParam.h"
#include "deviceGpu.h"
#include "secondTimeDerivative.h"
#include <omp.h>

using namespace SEP;

template <class V1, class V2>
class seismicOperator2D : public Operator <V1, V2> {

	protected:

		std::shared_ptr<fdParam> _fdParam;
		std::shared_ptr<deviceGpu> _sources, _receivers;
		int *_sourcesPositionReg, *_receiversPositionReg;
		int _nSourcesReg, _nReceiversReg;
		int _nts;
		int _saveWavefield;
		int _iGpu, _nGpu;
		std::shared_ptr<interpTimeLinTbb> _timeInterp;
		std::shared_ptr<secondTimeDerivative> _secTimeDer;
		std::shared_ptr<V2> _sourcesSignals, _sourcesSignalsRegDts, _sourcesSignalsRegDtsDt2, _sourcesSignalsRegDtwDt2, _sourcesSignalsRegDtw;

	public:

		// QC
		virtual bool checkParfileConsistency(std::shared_ptr<V1> model, std::shared_ptr<V2> data) const = 0; // Pure virtual: needs to implemented in derived class

		// Sources
		void setSources(std::shared_ptr<deviceGpu> sources); // This one is for the nonlinear modeling operator
		void setSources(std::shared_ptr<deviceGpu> sources, std::shared_ptr<V2> sourcesSignals); // For the other operators (Born + Tomo + Wemva)

		// Receivers
		void setReceivers(std::shared_ptr<deviceGpu> receivers);

		// Acquisition
		void setAcquisition(std::shared_ptr<deviceGpu> sources, std::shared_ptr<deviceGpu> receivers, const std::shared_ptr<V1> model, const std::shared_ptr<V2> data); // Nonlinear
		void setAcquisition(std::shared_ptr<deviceGpu> sources, std::shared_ptr<V2> sourcesSignals, std::shared_ptr<deviceGpu> receivers, const std::shared_ptr<V1> model, const std::shared_ptr<V2> data); // Born + Tomo + Wemva

		// Scaling
		void scaleSeismicSource(const std::shared_ptr<deviceGpu> seismicSource, std::shared_ptr<V2> signal, const std::shared_ptr<fdParam> parObj) const;

		// Other mutators
		void setGpuNumber(int iGpu){_iGpu = iGpu;}
		std::shared_ptr<double3DReg> setWavefield(int wavefieldFlag); // If flag=1, allocates a wavefield (with the correct size) and returns it. If flag=0, return a dummy wavefield of size 1x1x1
		virtual void setAllWavefields(int wavefieldFlag) = 0; // Allocates all wavefields associated with a seismic operator --> this function has to be implemented by child classes

		// Accessors
		std::shared_ptr<fdParam> getFdParam(){ return _fdParam; }

};

#include "seismicOperator2D.cpp"

#endif
