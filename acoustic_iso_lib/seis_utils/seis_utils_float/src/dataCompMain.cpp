#include <omp.h>
#include <iostream>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include "float1DReg.h"
#include "float2DReg.h"
#include "float3DReg.h"
#include "float4DReg.h"
#include "ioModes.h"
#include <vector>
#include <string>

using namespace SEP;

int main(int argc, char **argv) {


    ioModes modes(argc, argv);
	std::shared_ptr <SEP::genericIO> io = modes.getDefaultIO();
	std::shared_ptr <paramObj> par = io->getParamObj();
	std::string side = par->getString("side", "right");
	int iShot = par->getInt("iShot");
    int nbPanel = par->getInt("nbPanel",3);

    // Observed data
    std::shared_ptr<SEP::genericRegFile> dObsFile = io->getRegFile("obsIn",usageIn);
	std::shared_ptr<SEP::hypercube> dObsHyper = dObsFile->getHyper();
	std::shared_ptr<SEP::float2DReg> dObs(new SEP::float2DReg(dObsHyper));
	dObsFile->readFloatStream(dObs);

    // Data residuals or predicted data
	int res = par->getInt("res",1);
    std::shared_ptr<SEP::genericRegFile> modelFile = io->getRegFile("model",usageIn);
	std::shared_ptr<SEP::hypercube> modelHyper = modelFile->getHyper();
	std::shared_ptr<SEP::float3DReg> model(new SEP::float3DReg(modelHyper));
	modelFile->readFloatStream(model);

    // Iteration
    int nIter=model->getHyper()->getAxis(3).n;
	axis iterAxis = axis(nIter, 0.0, 1.0);

    // Shots
    int oShotGrid=par->getInt("xSource");
	int dShotGrid=par->getInt("spacingShots");

    // Receivers
    float oRec=model->getHyper()->getAxis(2).o;
    float dRec=model->getHyper()->getAxis(2).d;
    int nRec=model->getHyper()->getAxis(2).n;
	axis recAxis = axis(nRec, oRec, dRec);

    // Time
    float ots=model->getHyper()->getAxis(1).o;
    float dts=model->getHyper()->getAxis(1).d;
    int nts=model->getHyper()->getAxis(1).n;
	axis timeAxis = axis(nts, ots, dts);

	// Find indices on the "shot grid" and "receiver grid"
	int iShotRecGrid=oShotGrid+iShot*dShotGrid;
	float xShotNew=oRec+iShotRecGrid*dRec;



    // Find the number of traces for each side
    int nRecNew;
	if(side=="right"){
		nRecNew=nRec-iShotRecGrid-1;
	} else{
		nRecNew=iShotRecGrid;
    }

    // Total number of receivers for the super shot gather
    if (nbPanel == 3){

        // Compute total number of receiver in super shot gather
        int nRecTotal = 3*nRecNew+2;

        // Allocate super shot gather
        axis recNewAxis=axis(nRecTotal,-nRecNew*dRec,dRec);
    	std::shared_ptr<hypercube> dataHyper(new hypercube(timeAxis, recNewAxis, iterAxis));
    	std::shared_ptr<SEP::float3DReg> data(new SEP::float3DReg(dataHyper));
        data->scale(0.0);

        // Fill data
        if (side == "right"){

            #pragma omp parallel for
            for (int iIter=0; iIter<nIter; iIter++){
                for (int iRec=0; iRec<nRecNew; iRec++){
                    for (int its=0; its<nts; its++){
                        (*data->_mat)[iIter][iRec][its] = (*dObs->_mat)[iShotRecGrid+nRecNew-iRec][its];
                        (*data->_mat)[iIter][iRec+nRecNew+1][its] = (*model->_mat)[iIter][iShotRecGrid+iRec+1][its]+(*dObs->_mat)[iShotRecGrid+iRec+1][its];
                        (*data->_mat)[iIter][nRecNew][its] = (*dObs->_mat)[iShotRecGrid][its];
                        (*data->_mat)[iIter][iRec+2*nRecNew+1][its] = (*dObs->_mat)[iShotRecGrid+nRecNew-iRec][its];
                    }
                }
            }

        } else {

            #pragma omp parallel for
            for (int iIter=0; iIter<nIter; iIter++){
                for (int iRec=0; iRec<nRecNew; iRec++){
                    for (int its=0; its<nts; its++){
                        (*data->_mat)[iIter][iRec][its] = (*dObs->_mat)[iRec][its];
                        (*data->_mat)[iIter][iRec+nRecNew+1][its] = (*model->_mat)[iIter][iShotRecGrid-iRec-1][its]+(*dObs->_mat)[iShotRecGrid-iRec-1][its];
                        (*data->_mat)[iIter][nRecNew][its] = (*dObs->_mat)[iShotRecGrid][its];
                        (*data->_mat)[iIter][iRec+2*nRecNew+1][its] = (*dObs->_mat)[iRec][its];
                    }
                }
            }
        }

        /* Files shits */
        std::shared_ptr <genericRegFile> dataFile;
        dataFile = io->getRegFile(std::string("data"), usageOut);
        dataFile->setHyper(dataHyper);
        dataFile->writeDescription();
    	dataFile->writeFloatStream(data);

    } else {

        // Compute total number of receiver in super shot gather
        int nRecTotal=2*nRecNew+1;

        // Allocate super shot gather
        axis recNewAxis=axis(nRecTotal,-nRecNew*dRec,dRec);
    	std::shared_ptr<hypercube> dataHyper(new hypercube(timeAxis, recNewAxis, iterAxis));
    	std::shared_ptr<SEP::float3DReg> data(new SEP::float3DReg(dataHyper));
        data->scale(0.0);

        // Fill data
        if (side == "right"){
            #pragma omp parallel for
            for (int iIter=0; iIter<nIter; iIter++){
                for (int iRec=0; iRec<nRecNew; iRec++){
                    for (int its=0; its<nts; its++){
                        (*data->_mat)[iIter][iRec][its] = (*dObs->_mat)[iShotRecGrid+nRecNew-iRec][its];
                        (*data->_mat)[iIter][iRec+nRecNew+1][its] = (*model->_mat)[iIter][iShotRecGrid+iRec+1][its]+(*dObs->_mat)[iShotRecGrid+iRec+1][its];
                        (*data->_mat)[iIter][nRecNew][its] = (*dObs->_mat)[iShotRecGrid][its];
                    }
                }
            }
        } else {
            #pragma omp parallel for
            for (int iIter=0; iIter<nIter; iIter++){
                for (int iRec=0; iRec<nRecNew; iRec++){
                    for (int its=0; its<nts; its++){
                        (*data->_mat)[iIter][iRec][its] = (*dObs->_mat)[iRec][its];
                        (*data->_mat)[iIter][iRec+nRecNew+1][its] = (*model->_mat)[iIter][iShotRecGrid-iRec-1][its]+(*dObs->_mat)[iShotRecGrid-iRec-1][its];
                        (*data->_mat)[iIter][nRecNew][its] = (*dObs->_mat)[iShotRecGrid][its];
                    }
                }
            }
        }

        /* Files shits */
        std::shared_ptr <genericRegFile> dataFile;
        dataFile = io->getRegFile(std::string("data"), usageOut);
        dataFile->setHyper(dataHyper);
        dataFile->writeDescription();
    	dataFile->writeFloatStream(data);

    }

    return 0;
}
