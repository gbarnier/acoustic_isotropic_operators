#include <double3DReg.h>
#include "dataTaper.h"

using namespace SEP;

dataTaper::dataTaper(double maxOffset, double exp, double taperWidth, std::shared_ptr<SEP::hypercube> dataHyper, std::string muteType){

	_maxOffset=std::abs(maxOffset); // Beyond this offset, you start muting and tapering [km]
	_exp=exp; // Exponent on the cosine/sine exponential decay
	_taperWidth=std::abs(taperWidth); // Tapering width on each side of the shot position [km]
	_dataHyper=dataHyper;
	_muteType=muteType;

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
	_taperMask=std::make_shared<double3DReg>(_dataHyper);
	_taperMask->set(1.0); // Set mask value to 1

	// Generate taper mask for offset muting and tapering
	if (_muteType == "offset"){

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
}

void dataTaper::forward(const bool add, const std::shared_ptr<double3DReg> model, std::shared_ptr<double3DReg> data) const{

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

void dataTaper::adjoint(const bool add, std::shared_ptr<double3DReg> model, const std::shared_ptr<double3DReg> data) const{

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
