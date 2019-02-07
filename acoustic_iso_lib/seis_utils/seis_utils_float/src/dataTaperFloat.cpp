#include <float3DReg.h>
#include "dataTaperFloat.h"

using namespace SEP;

dataTaperFloat::dataTaperFloat(float maxOffset, float exp, float taperWidth, std::shared_ptr<SEP::hypercube> dataHyper){

	_maxOffset=std::abs(maxOffset); // Beyond this offset, you start muting and tapering [km]
	_exp=exp; // Exponent on the cosine/sine exponential decay
	_taperWidth=std::abs(taperWidth); // Tapering width on each side of the shot position [km]
	_dataHyper=dataHyper;

	// Shots geometry
	_nShot=_dataHyper->getAxis(3).n; // Number of shots
	_xMinShot = _dataHyper->getAxis(3).o; // Minimum horizontal position of shots [km]
	_dShot = _dataHyper->getAxis(3).d; // Shots spacing [km]
	_xMaxShot = _xMinShot+(_nShot-1)*_dShot; // Maximum horizontal position of shots [km]

	// Receiver geometry
	_nRec=_dataHyper->getAxis(2).n; // Number of receivers per shot (assumed to be constant)
	_xMinRec=_dataHyper->getAxis(2).o; // Minimum horizontal position of receivers [km]
	_dRec = _dataHyper->getAxis(2).d; // Receiver spacing [km]
	_xMaxRec = _xMinRec+(_nRec-1)*_dRec; // Maximum horizontal position of receivers [km]

	// Allocate and computer taper mask
	_taperMask=std::make_shared<float3DReg>(_dataHyper);
	_taperMask->set(1.0); // Set mask value to 1

	// Generate taper mask for offset muting and tapering
	for (int iShot=0; iShot<_nShot; iShot++){

		// Compute shot position
		float s=_xMinShot+iShot*_dShot;

		//     |------- 0 ------||------- Taper ------||----------- 1 -----------||------- Taper -------||----- 0 --------|
		// _xMinRec---------xMinRecMute2--------xMinRecMute1--------s--------xMaxRecMute1--------xMaxRecMute2--------_xMaxRec

		// Compute cutoff position [km] of the receivers from each side of the source
		float xMinRecMute1=std::max(s-_maxOffset, _xMinRec);
		float xMaxRecMute1=std::min(s+_maxOffset, _xMaxRec);

		// Compute cutoff position [km] of the receivers from each side of the source
		// Beyond that cutoff, zero out the traces
		float xMinRecMute2=std::max(s-_maxOffset-_taperWidth, _xMinRec);
		float xMaxRecMute2=std::min(s+_maxOffset+_taperWidth, _xMaxRec);

		// Compute inner bounds [samples]
		int ixMinRecMute1=(xMinRecMute1-_xMinRec)/_dRec;
		int ixMaxRecMute1=(xMaxRecMute1-_xMinRec)/_dRec;

		// Compute outer bounds [samples]
		int ixMinRecMute2=(xMinRecMute2-_xMinRec)/_dRec;
		int ixMaxRecMute2=(xMaxRecMute2-_xMinRec)/_dRec;

		#pragma omp parallel for
		for (int iRec=0; iRec<_nRec; iRec++){

			// Outside
			if( (iRec<ixMinRecMute2) || (iRec>ixMaxRecMute2) ){
				for (int its=0; its<_dataHyper->getAxis(1).n; its++){
					(*_taperMask->_mat)[iShot][iRec][its] = 0.0;
				}
			}
			// Left tapering zone
			if((iRec>=ixMinRecMute2) && (iRec<ixMinRecMute1)){
				float argument=1.0*(iRec-ixMinRecMute2)/(ixMinRecMute1-ixMinRecMute2);
				float weight=pow(sin(3.14159/2.0*argument), _exp);
				for (int its=0; its<_dataHyper->getAxis(1).n; its++){
					(*_taperMask->_mat)[iShot][iRec][its] = weight;
				}
			}
			// Right tapering zone
			if((iRec>ixMaxRecMute1) && (iRec<=ixMaxRecMute2)){
				float argument=1.0*(iRec-ixMaxRecMute1)/(ixMaxRecMute2-ixMaxRecMute1);
				float weight=pow(cos(3.14159/2.0*argument), _exp);
				for (int its=0; its<_dataHyper->getAxis(1).n; its++){
					(*_taperMask->_mat)[iShot][iRec][its] = weight;
				}
			}
		}
	}
}

dataTaperFloat::dataTaperFloat(float t0, float velMute, float exp, float taperWidth, std::shared_ptr<SEP::hypercube> dataHyper, std::string moveout){

	_t0=t0; // Time shift for zero offset [s]
	_velMute=velMute; // Velocity for mutin equation t=t0+offset/vel or t =sqrt(t0^2+(offset/vel)^2) [km/s]
	_exp=exp; // Exponent on the cosine/sine exponential decay
	_taperWidth=std::abs(taperWidth); // Width of taper [s]
	_dataHyper=dataHyper;
	_mouveout=moveout; // Linear or hyperbolic

	// Shots geometry
	_nShot=_dataHyper->getAxis(3).n; // Number of shots
	_xMinShot = _dataHyper->getAxis(3).o; // Minimum horizontal position of shots [km]
	_dShot = _dataHyper->getAxis(3).d; // Shots spacing [km]
	_xMaxShot = _xMinShot+(_nShot-1)*_dShot; // Maximum horizontal position of shots [km]

	// Receiver geometry
	_nRec=_dataHyper->getAxis(2).n; // Number of receivers per shot (assumed to be constant)
	_xMinRec=_dataHyper->getAxis(2).o; // Minimum horizontal position of receivers [km]
	_dRec = _dataHyper->getAxis(2).d; // Receiver spacing [km]
	_xMaxRec = _xMinRec+(_nRec-1)*_dRec; // Maximum horizontal position of receivers [km]

	// Time sampling parameters
	_nts=_dataHyper->getAxis(1).n; // Number of time samples
	_ots=_dataHyper->getAxis(1).o; // Initial time
	_dts = _dataHyper->getAxis(1).d; // Time sampling
	_tMax = _ots+(_nts-1)*_dts; // Max time for recording

	// Check that t0 is bigger than time axis origin
	if (_t0<_ots){
		std::cout << "**** ERROR: Please provide t0 > ots ****";
		assert(1==0);
	}

	// Allocate and computer taper mask
	_taperMask=std::make_shared<float3DReg>(_dataHyper);
	_taperMask->set(1.0); // Set default mask value to 1

	// Compute mask
	for (int iShot=0; iShot<_nShot; iShot++){

		float s=_xMinShot+iShot*_dShot;	// Compute shot position
		// std::cout << "shot position = " << s << std::endl;
		for (int iRec=0; iRec<_nRec; iRec++){ // Loop over receivers

			float r=_xMinRec+iRec*_dRec; // Compute receiver position
			float offset=std::abs(s-r); // Compute offset
			float tCutoff1;

			// Compute time cutoffs
			if (moveout=="linear"){
				tCutoff1=_t0+offset/_velMute; // Compute linear time cutoff 1
			}
			if (moveout=="hyperbolic"){
				tCutoff1=std::sqrt(_t0*_t0+offset*offset/(_velMute*_velMute)); // Compute hyperbolic time cutoff 1
			}
			float tCutoff2=tCutoff1+_taperWidth; // Compute cutoff 2

			// Convert time cutoffs to index [sample]
			int itCutoff1=(tCutoff1-_ots)/_dts;
			int itCutoff2=(tCutoff2-_ots)/_dts;

			// Loop over time - First zone where we set the data to zero
			for (int its=_ots; its<std::min(_nts-1,itCutoff1); its++){
				(*_taperMask->_mat)[iShot][iRec][its] = 0.0;
			}
			// Loop over time - Second zone where we taper the data
			for (int its=std::min(_nts-1,itCutoff1); its<std::min(_nts-1,itCutoff2); its++){
				float argument=1.0*(its-itCutoff1)/(itCutoff2-itCutoff1);
				float weight=pow(sin(3.14159/2.0*argument), _exp);
				(*_taperMask->_mat)[iShot][iRec][its] = weight;
			}
		}
	}
}

void dataTaperFloat::forward(const bool add, const std::shared_ptr<float3DReg> model, std::shared_ptr<float3DReg> data) const{

	if (!add) data->scale(0.0);

	// Apply tapering mask to seismic data
	#pragma omp parallel for collapse(3)
	for (int iShot=0; iShot<_nShot; iShot++){
		for (int iRec=0; iRec<_nRec; iRec++){
			for (int its=0; its<_dataHyper->getAxis(1).n; its++){
				(*data->_mat)[iShot][iRec][its] += (*model->_mat)[iShot][iRec][its]*(*_taperMask->_mat)[iShot][iRec][its];
			}
		}
	}
}

void dataTaperFloat::adjoint(const bool add, std::shared_ptr<float3DReg> model, const std::shared_ptr<float3DReg> data) const{

	if (!add) model->scale(0.0);

	// Apply tapering mask to seismic data
	#pragma omp parallel for collapse(3)
	for (int iShot=0; iShot<_nShot; iShot++){
		for (int iRec=0; iRec<_nRec; iRec++){
			for (int its=0; its<_dataHyper->getAxis(1).n; its++){
				(*model->_mat)[iShot][iRec][its] += (*data->_mat)[iShot][iRec][its]*(*_taperMask->_mat)[iShot][iRec][its];
			}
		}
	}
}
