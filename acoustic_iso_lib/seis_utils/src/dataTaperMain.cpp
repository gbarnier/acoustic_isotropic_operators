#include <string>
#include <iostream>
#include "float2DReg.h"
#include "float3DReg.h"
#include "ioModes.h"
#include <cmath>

using namespace SEP;

int main(int argc, char **argv) {

	// IO bullshit
	ioModes modes(argc, argv);
	std::shared_ptr <SEP::genericIO> io = modes.getDefaultIO();
	std::shared_ptr <paramObj> par = io->getParamObj();

	// Read model (seismic data)
	std::shared_ptr <genericRegFile> modelFile = io->getRegFile("model", usageIn);
 	std::shared_ptr<SEP::hypercube> modelHyper = modelFile->getHyper();

	// Case where you only have one shot
	if (modelHyper->getNdim()==2){
		axis sourceAxis(1, par->getFloat("oShot", -1.0), 1.0);
		modelHyper->addAxis(sourceAxis);
	}

	// Get model dimensions
	int nts = modelHyper->getAxis(1).n;
	int nRec = modelHyper->getAxis(2).n;
	int nShot = modelHyper->getAxis(3).n;

	// Read model
 	std::shared_ptr<SEP::float3DReg> model(new SEP::float3DReg(modelHyper));
	modelFile->readFloatStream(model);

	// Allocate data (muted seismic data) + tapering mask
	std::shared_ptr <genericRegFile> dataFile = io->getRegFile("data", usageOut);
	dataFile->setHyper(modelHyper);
	dataFile->writeDescription();
	std::shared_ptr <genericRegFile> taperFile = io->getRegFile("taper", usageOut);
	taperFile->setHyper(modelHyper);
	taperFile->writeDescription();
 	std::shared_ptr<SEP::float3DReg> data(new SEP::float3DReg(modelHyper));
	std::shared_ptr<SEP::float3DReg> taper;

	// Muting type
	std::string mute=par->getString("mute");

	// Offset muting
	if (mute == "offset"){

		std::cout << " " << std::endl;
		std::cout << "-------------------------------------------------------------------" << std::endl;
		std::cout << "--------------------------- Offset muting -------------------------" << std::endl;
		std::cout << "-------------------------------------------------------------------" << std::endl;
		std::cout << " " << std::endl;

		// Tapering parameters
		float maxOffset=par->getFloat("maxOffset", 0.0); // Max offset [km] to keep intact in the shot gather. After this, start tapering
		maxOffset=std::abs(maxOffset);
		int saveTaper=par->getFloat("saveTaper", 0); // Flag to save the taper and output it on disk
		if (saveTaper==1){
			taper=data->clone();
			taper->set(1.0);
		}

		int taperWidth=par->getInt("taperWidth"); // Taper width [samples]
		float exp=par->getFloat("exp", -1.0); // Exponent for the cosine decay [-]

		// Shots geometry
		float xMinShot = modelHyper->getAxis(3).o; // Minimum horizontal position of shots [km]
		float dShot = modelHyper->getAxis(3).d; // Shots spacing [km]
		float xMaxShot = xMinShot+(nShot-1)*dShot; // Maximum horizontal position of shots [km]

		// Receivers geometry
		float xMinRec = modelHyper->getAxis(2).o; // Minimum horizontal position of receivers [km]
		float dRec = modelHyper->getAxis(2).d; // Receiver spacing [km]
		float xMaxRec = xMinRec+(nRec-1)*dRec; // Maximum horizontal position of receivers [km]

		// Apply tapering
		for (int iShot=0; iShot<nShot; iShot++){

			// Compute shot position
			float s=xMinShot+iShot*dShot;

			// Compute cutoff position [km] of the receivers from each side of the source
			float xMinRecMute=std::max(s-maxOffset, xMinRec);
			float xMaxRecMute=std::min(s+maxOffset, xMaxRec);

			// Convert receiver inner bounds to samples
			int ixMinRecMuteIn=(xMinRecMute-xMinRec)/dRec;
			int ixMaxRecMuteIn=(xMaxRecMute-xMinRec)/dRec;

			// Compute outer bounds [samples]
			int ixMinRecMuteOut=ixMinRecMuteIn-taperWidth;
			int ixMaxRecMuteOut=ixMaxRecMuteIn+taperWidth;

			for (int iRec=0; iRec<nRec; iRec++){

				// Outside
				if( (iRec<ixMinRecMuteOut) || (iRec>ixMaxRecMuteOut) ){
					for (int its=0; its<nts; its++){
						if (saveTaper == 1){(*taper->_mat)[iShot][iRec][its] = 0.0;}
						(*data->_mat)[iShot][iRec][its] = 0.0;
					}
				}

				// Middle zone
				if((iRec>=ixMinRecMuteIn) && (iRec<=ixMaxRecMuteIn)){
					for (int its=0; its<nts; its++){
						if (saveTaper == 1){(*taper->_mat)[iShot][iRec][its] = 1.0;}
						(*data->_mat)[iShot][iRec][its] = (*model->_mat)[iShot][iRec][its];
					}
				}

				// Left tapering zone
				if((iRec>=ixMinRecMuteOut) && (iRec<ixMinRecMuteIn)){
					float argument=1.0*(iRec-ixMinRecMuteOut)/(ixMinRecMuteIn-ixMinRecMuteOut);
					float weight=pow(sin(3.14159/2.0*argument), exp);
					for (int its=0; its<nts; its++){
						if (saveTaper == 1){(*taper->_mat)[iShot][iRec][its] = weight;}
						(*data->_mat)[iShot][iRec][its] = weight*(*model->_mat)[iShot][iRec][its];
					}
				}
				// Right tapering zone
				if((iRec>ixMaxRecMuteIn) && (iRec<=ixMaxRecMuteOut)){
					float argument=1.0*(iRec-ixMaxRecMuteIn)/(ixMaxRecMuteOut-ixMaxRecMuteIn);
					float weight=pow(cos(3.14159/2.0*argument), exp);
					for (int its=0; its<nts; its++){
						if(saveTaper == 1){(*taper->_mat)[iShot][iRec][its] = weight;}
						(*data->_mat)[iShot][iRec][its] = weight*(*model->_mat)[iShot][iRec][its];
					}
				}
			}
		}

		// Write data and taper
		dataFile->writeFloatStream(data);
		if (saveTaper == 1){taperFile->writeFloatStream(taper);}
	}

	return 0;
}
