#include <float1DReg.h>
#include <float3DReg.h>
#include "dataTaper.h"

using namespace SEP;

// Constructor for both time and offset
dataTaper::dataTaper(float t0, float velMute, float expTime, float taperWidthTime, std::string moveout, int reverseTime, float maxOffset, float expOffset, float taperWidthOffset, int reverseOffset, std::shared_ptr<SEP::hypercube> dataHyper, float taperEndTraceWidth){

	// Shots geometry
	_dataHyper = dataHyper;
	_nShot = _dataHyper->getAxis(3).n; // Number of shots
	_xMinShot = _dataHyper->getAxis(3).o; // Minimum horizontal position of shots [km]
	_dShot = _dataHyper->getAxis(3).d; // Shots spacing [km]
	_xMaxShot = _xMinShot+(_nShot-1)*_dShot; // Maximum horizontal position of shots [km]

	// Receiver geometry
	_nRec=_dataHyper->getAxis(2).n; // Number of receivers per shot (assumed to be constant)
	_xMinRec=_dataHyper->getAxis(2).o; // Minimum horizontal position of receivers [km]
	_dRec = _dataHyper->getAxis(2).d; // Receiver spacing [km]
	_xMaxRec = _xMinRec+(_nRec-1)*_dRec; // Maximum horizontal position of receivers [km]

	// Time mask parameters
	_t0=t0;
	_velMute=velMute;
	_expTime=expTime;
	_taperWidthTime=std::abs(taperWidthTime);
	_reverseTime=reverseTime;
	_moveout=moveout;

	// Time sampling parameters
	_nts=_dataHyper->getAxis(1).n; // Number of time samples
	_ots=_dataHyper->getAxis(1).o; // Initial time
	_dts = _dataHyper->getAxis(1).d; // Time sampling
	_tMax = _ots+(_nts-1)*_dts; // Max time for recording

	// Offset mask parameters
	_maxOffset=std::abs(maxOffset);
	_expOffset=expOffset;
	_taperWidthOffset=std::abs(taperWidthOffset);
	_reverseOffset=reverseOffset;

	// Compute time taper mask
	computeTaperMaskTime();

	// Compute offset taper mask
	computeTaperMaskOffset();

	// Compute total mask
	_taperMask=_taperMaskTime;
	_taperMask->mult(_taperMaskOffset);

	// Compute weighting for end of trace
	_taperEndTraceWidth = taperEndTraceWidth; // Taper width [s]
	computeTaperEndTrace();

}

// Constructor for no tapering
dataTaper::dataTaper(std::shared_ptr<SEP::hypercube> dataHyper, float taperEndTraceWidth){

	// Shots geometry
	_dataHyper = dataHyper;
	_nShot = _dataHyper->getAxis(3).n; // Number of shots
	_nRec=_dataHyper->getAxis(2).n; // Number of receivers (constant over shots)

	// Allocate and compute taper mask
	_taperMask=std::make_shared<float3DReg>(_dataHyper);
	_taperMask->set(1.0); // Set mask value to 1
	// Time sampling parameters
	_nts=_dataHyper->getAxis(1).n; // Number of time samples
	_ots=_dataHyper->getAxis(1).o; // Initial time
	_dts = _dataHyper->getAxis(1).d; // Time sampling
	_tMax = _ots+(_nts-1)*_dts; // Max time for recording
	std::cout << "Inside 1" << std::endl;
	// Compute weighting for end of trace
	_taperEndTraceWidth = taperEndTraceWidth; // Taper width [s]
	std::cout << "_taperEndTraceWidth" << _taperEndTraceWidth << std::endl;
	computeTaperEndTrace();

}

// Constructor for time only
dataTaper::dataTaper(float t0, float velMute, float expTime, float taperWidthTime, std::shared_ptr<SEP::hypercube> dataHyper, std::string moveout, int reverseTime, float taperEndTraceWidth){

	// Shots geometry
	_dataHyper = dataHyper;
	_nShot = _dataHyper->getAxis(3).n; // Number of shots
	_xMinShot = _dataHyper->getAxis(3).o; // Minimum horizontal position of shots [km]
	_dShot = _dataHyper->getAxis(3).d; // Shots spacing [km]
	_xMaxShot = _xMinShot+(_nShot-1)*_dShot; // Maximum horizontal position of shots [km]

	// Receiver geometry
	_nRec=_dataHyper->getAxis(2).n; // Number of receivers per shot (assumed to be constant)
	_xMinRec=_dataHyper->getAxis(2).o; // Minimum horizontal position of receivers [km]
	_dRec = _dataHyper->getAxis(2).d; // Receiver spacing [km]
	_xMaxRec = _xMinRec+(_nRec-1)*_dRec; // Maximum horizontal position of receivers [km]

	// Time mask parameters
	_t0=t0;
	_velMute=velMute;
	_expTime=expTime;
	_taperWidthTime=std::abs(taperWidthTime);
	_reverseTime=reverseTime;
	_moveout=moveout;

	// Time sampling parameters
	_nts=_dataHyper->getAxis(1).n; // Number of time samples
	_ots=_dataHyper->getAxis(1).o; // Initial time
	_dts = _dataHyper->getAxis(1).d; // Time sampling
	_tMax = _ots+(_nts-1)*_dts; // Max time for recording

	// Compute time taper mask
	computeTaperMaskTime();

	// Total mask
	_taperMask=_taperMaskTime;

	// Compute weighting for end of trace
	_taperEndTraceWidth = taperEndTraceWidth; // Taper width [s]
	computeTaperEndTrace();

}

// Constructor for offset only
dataTaper::dataTaper(float maxOffset, float expOffset, float taperWidthOffset, std::shared_ptr<SEP::hypercube> dataHyper, int reverseOffset, float taperEndTraceWidth){

	// Shots geometry
	_dataHyper = dataHyper;
	_nShot = _dataHyper->getAxis(3).n; // Number of shots
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

	// Offset mask parameters
	_maxOffset=std::abs(maxOffset);
	_expOffset=expOffset;
	_taperWidthOffset=std::abs(taperWidthOffset);
	_reverseOffset=reverseOffset;

	// Compute offset taper mask
	computeTaperMaskOffset();

	// Total mask
	_taperMask=_taperMaskOffset;

	// Compute weighting for end of trace
	_taperEndTraceWidth = taperEndTraceWidth; // Taper width [s]
	computeTaperEndTrace();

	// Delete other masks
	// delete _taperMaskOffset;
}

// Compute mask
void dataTaper::computeTaperMaskOffset(){

	// Allocate and computer taper mask
	_taperMaskOffset=std::make_shared<float3DReg>(_dataHyper);
	_taperMaskOffset->set(1.0); // Set mask value to 1

	// Generate taper mask for offset muting and tapering
	for (int iShot=0; iShot<_nShot; iShot++){

		if (_reverseOffset==0) {

			// Compute shot position
			float s=_xMinShot+iShot*_dShot;
			float xMinRecMute1, xMaxRecMute1;
			float xMinRecMute2, xMaxRecMute2;
			int ixMinRecMute2, ixMinRecMute1, ixMaxRecMute1, ixMaxRecMute2;
			int ixMinRecMute2True, ixMinRecMute1True, ixMaxRecMute1True, ixMaxRecMute2True;

			//     |------- 0 ------||------- Taper ------||----------- 1 -----------||------- Taper -------||----- 0 --------|
			// _xMinRec---------xMinRecMute2--------xMinRecMute1--------s--------xMaxRecMute1--------xMaxRecMute2--------_xMaxRec

			// Compute cutoff position [km] of the receivers from each side of the source
			xMinRecMute1=s-_maxOffset; // Inner bounds
			xMaxRecMute1=s+_maxOffset;
			xMinRecMute2=xMinRecMute1-_taperWidthOffset; // Outer bounds
			xMaxRecMute2=xMaxRecMute1+_taperWidthOffset;

			// Compute theoretical index of min1
			if (xMinRecMute1-_xMinRec >= 0){
				ixMinRecMute1True=(xMinRecMute1-_xMinRec)/_dRec+0.5;
			} else {
				ixMinRecMute1True=(xMinRecMute1-_xMinRec)/_dRec-0.5;
			}
			ixMinRecMute1=std::max(ixMinRecMute1True, 0);

			// Compute theoretical index of min2
			if (xMinRecMute2-_xMinRec >= 0){
				ixMinRecMute2True=(xMinRecMute2-_xMinRec)/_dRec+0.5;
			} else {
				ixMinRecMute2True=(xMinRecMute2-_xMinRec)/_dRec-0.5;
			}
			ixMinRecMute2=std::max(ixMinRecMute2True, 0);

			// Compute index of max inner bound
			ixMaxRecMute1True=(xMaxRecMute1-_xMinRec)/_dRec+0.5;
			ixMaxRecMute1=std::min(ixMaxRecMute1True, _nRec-1);

			// Compute index of max outer bound
			ixMaxRecMute2True=(xMaxRecMute2-_xMinRec)/_dRec+0.5;
			ixMaxRecMute2=std::min(ixMaxRecMute2True, _nRec-1);

			if (ixMinRecMute2True == ixMinRecMute1True){
				std::cout << "**** ERROR [Offset muting]: Cutoff min indices are identical. Use a larger taperWidth value ****" << std::endl;
				throw std::runtime_error("");
			}
			if (ixMaxRecMute2True == ixMaxRecMute1True){
				std::cout << "**** ERROR [Offset muting]: Cutoff min indices are identical. Use a larger taperWidth value ****" << std::endl;
				throw std::runtime_error("");
			}

			// #pragma omp parallel for
			for (int iRec=0; iRec<_nRec; iRec++){

				// Outside
				if( (iRec<ixMinRecMute2) || (iRec>ixMaxRecMute2) ){
					for (int its=0; its<_dataHyper->getAxis(1).n; its++){
						(*_taperMaskOffset->_mat)[iShot][iRec][its] = 0.0;
					}
				}
				// Left tapering zone
				if( (iRec>=ixMinRecMute2) && (iRec<=ixMinRecMute1) ){
					float argument=1.0*(iRec-ixMinRecMute2True)/(ixMinRecMute1True-ixMinRecMute2True);
					float weight=pow(sin(3.14159/2.0*argument), _expOffset);
					for (int its=0; its<_dataHyper->getAxis(1).n; its++){
						(*_taperMaskOffset->_mat)[iShot][iRec][its] = weight;
					}
				}
				// Right tapering zone
				if( (iRec>=ixMaxRecMute1) && (iRec<=ixMaxRecMute2) ){
					float argument=1.0*(iRec-ixMaxRecMute1True)/(ixMaxRecMute2True-ixMaxRecMute1True);
					float weight=pow(cos(3.14159/2.0*argument), _expOffset);
					for (int its=0; its<_dataHyper->getAxis(1).n; its++){
						(*_taperMaskOffset->_mat)[iShot][iRec][its] = weight;
					}
				}
			}
		}
		else {

			// Compute shot position
			float s=_xMinShot+iShot*_dShot;
			float xMinRecMute1, xMaxRecMute1;
			float xMinRecMute2, xMaxRecMute2;
			int ixMinRecMute2, ixMinRecMute1, ixMaxRecMute1, ixMaxRecMute2;
			int ixMinRecMute2True, ixMinRecMute1True, ixMaxRecMute1True, ixMaxRecMute2True;

			// Compute cutoff position [km] of the receivers from each side of the source
			xMinRecMute1=s-_maxOffset;
			xMaxRecMute1=s+_maxOffset;
			xMinRecMute2=xMinRecMute1+_taperWidthOffset;
			xMaxRecMute2=xMaxRecMute1-_taperWidthOffset;

			// Make sure taperWidth < maxOffset
			if (_taperWidthOffset > _maxOffset){
				std::cout << "**** ERROR [Offset muting]: Make sure taperWidthOffset < maxOffset ****" << std::endl;
				throw std::runtime_error("");
			}

			// Compute theoretical index of min1
			if (xMinRecMute1-_xMinRec >= 0){
				ixMinRecMute1True=(xMinRecMute1-_xMinRec)/_dRec+0.5;
			} else {
				ixMinRecMute1True=(xMinRecMute1-_xMinRec)/_dRec-0.5;
			}
			ixMinRecMute1=std::max(ixMinRecMute1True, 0);

			// Compute theoretical index of min2
			if (xMinRecMute2-_xMinRec >= 0){
				ixMinRecMute2True=(xMinRecMute2-_xMinRec)/_dRec+0.5;
			} else {
				ixMinRecMute2True=(xMinRecMute2-_xMinRec)/_dRec-0.5;
			}
			ixMinRecMute2=std::max(ixMinRecMute2True, 0);

			if (ixMinRecMute2True == ixMinRecMute1True){
				std::cout << "**** ERROR [Offset muting]: Cutoff min indices are identical. Use a larger taperWidth value ****" << std::endl;
				throw std::runtime_error("");
			}
			if (ixMaxRecMute2True == ixMaxRecMute1True){
				std::cout << "**** ERROR [Offset muting]: Cutoff min indices are identical. Use a larger taperWidth value ****" << std::endl;
				throw std::runtime_error("");
			}

			// Compute index of max inner bound
			ixMaxRecMute1True=(xMaxRecMute1-_xMinRec)/_dRec+0.5;
			ixMaxRecMute1=std::min(ixMaxRecMute1True, _nRec-1);

			// Compute index of max outer bound
			ixMaxRecMute2True=(xMaxRecMute2-_xMinRec)/_dRec+0.5;
			ixMaxRecMute2=std::min(ixMaxRecMute2True, _nRec-1);

			#pragma omp parallel for
			for (int iRec=0; iRec<_nRec; iRec++){

				// Inside
				if ( (iRec>=ixMinRecMute2) && (iRec<=ixMaxRecMute2) ){
					for (int its=0; its<_dataHyper->getAxis(1).n; its++){
						(*_taperMaskOffset->_mat)[iShot][iRec][its] = 0.0;
					}
				}
				// Left tapering zone
				if ( (iRec>=ixMinRecMute1) && (iRec<ixMinRecMute2) ){
					float argument=1.0*(iRec-ixMinRecMute1True)/(ixMinRecMute2True-ixMinRecMute1True);
					float weight=pow(cos(3.14159/2.0*argument), _expOffset);
					for (int its=0; its<_dataHyper->getAxis(1).n; its++){
						(*_taperMaskOffset->_mat)[iShot][iRec][its] = weight;
					}
				}
				// Right tapering zone
				if ( (iRec>ixMaxRecMute2) && (iRec<=ixMaxRecMute1) ){
					float argument=1.0*(iRec-ixMaxRecMute2True)/(ixMaxRecMute1True-ixMaxRecMute2True);
					float weight=pow(sin(3.14159/2.0*argument), _expOffset);
					for (int its=0; its<_dataHyper->getAxis(1).n; its++){
						(*_taperMaskOffset->_mat)[iShot][iRec][its] = weight;
					}
				}
			}
		}
	}
}
void dataTaper::computeTaperMaskTime(){

	// Allocate and computer taper mask
	_taperMaskTime=std::make_shared<float3DReg>(_dataHyper);
	_taperMaskTime->set(1.0); // Set mask value to 1

	// Mute late arrivals
	if (_reverseTime==0) {
		for (int iShot=0; iShot<_nShot; iShot++){

			float s=_xMinShot+iShot*_dShot;	// Compute shot position

			#pragma omp parallel for
			for (int iRec=0; iRec<_nRec; iRec++){ // Loop over receivers

				float r=_xMinRec+iRec*_dRec; // Compute receiver position
				float offset=std::abs(s-r); // Compute offset
				float tCutoff1, tCutoff2;
				int itCutoff1True, itCutoff2True, itCutoff1, itCutoff2;

				// Time cutoff #1
				if (_moveout=="linear"){
					tCutoff1=_t0+offset/_velMute; // Compute linear time cutoff 1
				}
				if (_moveout=="hyperbolic" && _t0>=0){
					tCutoff1=std::sqrt(_t0*_t0+offset*offset/(_velMute*_velMute)); // Compute hyperbolic time cutoff 1
				}
				if (_moveout=="hyperbolic" && _t0<0){
					tCutoff1=std::sqrt(_t0*_t0+offset*offset/(_velMute*_velMute)) - 2.0*std::abs(_t0);
				}
				if (tCutoff1 < _ots){
					 itCutoff1True = (tCutoff1-_ots)/_dts-0.5; // Theoretical cutoff index (can be negative)
				} else {
	 				itCutoff1True = (tCutoff1-_ots)/_dts+0.5;
				}
				itCutoff1 = std::min(itCutoff1True, _nts-1);
				itCutoff1 = std::max(itCutoff1, 0);

				// Time cutoff #2
				tCutoff2=tCutoff1+_taperWidthTime;
				if (tCutoff2 < _ots){
					itCutoff2True = (tCutoff2-_ots)/_dts-0.5;
				} else {
					itCutoff2True = (tCutoff2-_ots)/_dts+0.5;
				}
				itCutoff2 = std::min(itCutoff2True, _nts-1);
				itCutoff2 = std::max(itCutoff2, 0);

				// Check the cutoff indices are different
				if (itCutoff2True == itCutoff1True){
					std::cout << "**** ERROR [Time muting]: Cutoff indices are identical. Use a larger taperWidth value ****" << std::endl;
					throw std::runtime_error("");
				}

				// Loop over time - Second zone where we taper the data
				for (int its=itCutoff1; its<itCutoff2; its++){
					float argument=1.0*(its-itCutoff1True)/(itCutoff2True-itCutoff1True);
					float weight=pow(cos(3.14159/2.0*argument), _expTime);
					(*_taperMaskTime->_mat)[iShot][iRec][its] = weight;
				}
				// Mute times after itCutoff2
				for (int its=itCutoff2; its<_nts; its++){
					(*_taperMaskTime->_mat)[iShot][iRec][its] = 0.0;
				}
			}
		}
	}
	// Mute early arrivals
	else {

		for (int iShot=0; iShot<_nShot; iShot++){

			float s=_xMinShot+iShot*_dShot;	// Compute shot position

			#pragma omp parallel for
			for (int iRec=0; iRec<_nRec; iRec++){ // Loop over receivers

				float r=_xMinRec+iRec*_dRec; // Compute receiver position
				float offset=std::abs(s-r); // Compute offset
				float tCutoff1, tCutoff2;
				int itCutoff1True, itCutoff2True, itCutoff1, itCutoff2;

				// Time cutoff #1
				if (_moveout=="linear"){
					tCutoff1=_t0+offset/_velMute; // Compute linear time cutoff 1
				}
				if (_moveout=="hyperbolic" && _t0>=0){
					tCutoff1=std::sqrt(_t0*_t0+offset*offset/(_velMute*_velMute)); // Compute hyperbolic time cutoff 1
				}
				if (_moveout=="hyperbolic" && _t0<0){
					tCutoff1=std::sqrt(_t0*_t0+offset*offset/(_velMute*_velMute)) - 2.0*std::abs(_t0);
				}
				if (tCutoff1 < _ots){
					 itCutoff1True = (tCutoff1-_ots)/_dts-0.5; // Theoretical cutoff index (can be negative)
				} else {
	 				itCutoff1True = (tCutoff1-_ots)/_dts+0.5;
				}
				itCutoff1 = std::min(itCutoff1True, _nts-1);
				itCutoff1 = std::max(itCutoff1, 0);

				// Time cutoff #2
				tCutoff2=tCutoff1-_taperWidthTime;

				// Convert time cutoffs to index [sample]
				if (tCutoff2 < _ots){
					itCutoff2True = (tCutoff2-_ots)/_dts-0.5;
				} else {
					itCutoff2True = (tCutoff2-_ots)/_dts+0.5;
				}
				itCutoff2 = std::min(itCutoff2True, _nts-1);
				itCutoff2 = std::max(itCutoff2, 0);


				// Loop over time - Mute the earlier times
				for (int its=0; its<itCutoff2; its++){
					(*_taperMaskTime->_mat)[iShot][iRec][its] = 0.0;
				}
				// Loop over time - Second zone where we taper the data
				for (int its=itCutoff2; its<itCutoff1; its++){
					float argument=1.0*(its-itCutoff2True)/(itCutoff1True-itCutoff2True);
					float weight=pow(sin(3.14159/2.0*argument), _expTime);
					(*_taperMaskTime->_mat)[iShot][iRec][its] = weight;
				}
			}
		}
	}
}
void dataTaper::computeTaperEndTrace(){

	// Allocate and computer taper mask
	_taperEndTrace=std::make_shared<float1DReg>(_nts);
	_taperEndTrace->set(1.0); // Set mask value to 1

	// Compute trace taper mask
	std::cout << "_taperEndTraceWidth" << _taperEndTraceWidth << std::endl;
	if (_taperEndTraceWidth>0.0){

		// Time after which we start tapering the trace [s]
		float tTaperEndTrace = _tMax - _taperEndTraceWidth;
		std::cout << "tMax = " << _tMax << std::endl;
		std::cout << "taperEndTraceWidth = " << _taperEndTraceWidth << std::endl;
		std::cout << "tTaperEndTrace = " << tTaperEndTrace << std::endl;

		// Make sure you're not out of bounds
		if (tTaperEndTrace < _ots){
			std::cout << "**** ERROR [End trace muting]: Make sure taperEndTraceWidth < total recording time ****" << std::endl;
			throw std::runtime_error("");
		}
		// Compute index from which you start tapering
		int itTaperEndTrace = (tTaperEndTrace-_ots)/_dts; // Index from which we start tapering
		// Compute trace taper
		for (int its=itTaperEndTrace; its<_nts; its++){
			float argument = 1.0*(its-itTaperEndTrace)/(_nts-1-itTaperEndTrace);
			(*_taperEndTrace->_mat)[its] = pow(cos(3.14159/2.0*argument), 2);
		}
		// Apply trace taper to taperMask
		#pragma omp parallel for collapse(3)
		for (int iShot=0; iShot<_nShot; iShot++){
			for (int iReceiver=0; iReceiver<_nRec; iReceiver++){
				for (int its=itTaperEndTrace; its<_nts; its++){
					(*_taperMask->_mat)[iShot][iReceiver][its] *= (*_taperEndTrace->_mat)[its];
				}
			}
		}
	}
}

void dataTaper::forward(const bool add, const std::shared_ptr<float3DReg> model, std::shared_ptr<float3DReg> data) const{

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

void dataTaper::adjoint(const bool add, std::shared_ptr<float3DReg> model, const std::shared_ptr<float3DReg> data) const{

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
