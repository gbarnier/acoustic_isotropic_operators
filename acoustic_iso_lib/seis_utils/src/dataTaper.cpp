#include <double3DReg.h>
#include "dataTaper.h"

using namespace SEP;

dataTaper::dataTaper(double maxOffset, double exp, int taperWidth, std::shared_ptr<SEP::hypercube> dataHyper, std::string mute){

	_maxOffset=maxOffset; // Beyond this offset, you start muting [km]
	_exp=exp;
	_taperWidth=taperWidth; // Number of points for tapering [samples]
	_dataHyper=dataHyper;
	_mute=mute;

	// Shots geometry
	_nShot=_dataHyper->getAxis(3).n; // Number of shot
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

	for (int iShot=0; iShot<_nShot; iShot++){

		// Compute shot position
		float s=_xMinShot+iShot*_dShot;

		// Compute cutoff position [km] of the receivers from each side of the source
		float xMinRecMute=std::max(s-_maxOffset, _xMinRec);
		float xMaxRecMute=std::min(s+_maxOffset, _xMaxRec);

		// Convert receiver inner bounds to samples
		int ixMinRecMuteIn=(xMinRecMute-_xMinRec)/_dRec;
		int ixMaxRecMuteIn=(xMaxRecMute-_xMinRec)/_dRec;

		// Compute outer bounds [samples]
		int ixMinRecMuteOut=ixMinRecMuteIn-_taperWidth;
		int ixMaxRecMuteOut=ixMaxRecMuteIn+_taperWidth;

		for (int iRec=0; iRec<_nRec; iRec++){
			// Outside
			if( (iRec<ixMinRecMuteOut) || (iRec>ixMaxRecMuteOut) ){
				for (int its=0; its<_dataHyper->getAxis(1).n; its++){
					(*_taperMask->_mat)[iShot][iRec][its] = 0.0;
				}
			}
			// Left tapering zone
			if((iRec>=ixMinRecMuteOut) && (iRec<ixMinRecMuteIn)){
				float argument=1.0*(iRec-ixMinRecMuteOut)/(ixMinRecMuteIn-ixMinRecMuteOut);
				float weight=pow(sin(3.14159/2.0*argument), _exp);
				for (int its=0; its<_dataHyper->getAxis(1).n; its++){
					(*_taperMask->_mat)[iShot][iRec][its] = weight;
				}
			}
			// Right tapering zone
			if((iRec>ixMaxRecMuteIn) && (iRec<=ixMaxRecMuteOut)){
				float argument=1.0*(iRec-ixMaxRecMuteIn)/(ixMaxRecMuteOut-ixMaxRecMuteIn);
				float weight=pow(cos(3.14159/2.0*argument), _exp);
				for (int its=0; its<_dataHyper->getAxis(1).n; its++){
					(*_taperMask->_mat)[iShot][iRec][its] = weight;
				}
			}
		}
	}
}

void dataTaper::forward(const bool add, const std::shared_ptr<double3DReg> model, std::shared_ptr<double3DReg> data) const{

	if (!add) data->scale(0.0);

	// Apply tapering mask to seismic data
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
	for (int iShot=0; iShot<_nShot; iShot++){
		for (int iRec=0; iRec<_nRec; iRec++){
			for (int its=0; its<_dataHyper->getAxis(1).n; its++){
				(*model->_mat)[iShot][iRec][its] += (*data->_mat)[iShot][iRec][its]*(*_taperMask->_mat)[iShot][iRec][its];
			}
		}
	}

}
