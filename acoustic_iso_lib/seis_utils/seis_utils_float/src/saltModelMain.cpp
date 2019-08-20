#include <omp.h>
#include <iostream>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include "float1DReg.h"
#include "float2DReg.h"
#include "ioModes.h"
#include <vector>
#include <string>

using namespace SEP;

int main(int argc, char **argv) {

    // Bullshit stuff
    ioModes modes(argc, argv);
	std::shared_ptr <SEP::genericIO> io = modes.getDefaultIO();
	std::shared_ptr <paramObj> par = io->getParamObj();
    int step = par->getInt("step",1);

    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////// Find water bottom index /////////////////////////
    ////////////////////////////////////////////////////////////////////////////

    // Read velocity model
    std::shared_ptr<SEP::genericRegFile> modelFile = io->getRegFile("model",usageIn);
	std::shared_ptr<SEP::hypercube> modelHyper = modelFile->getHyper();
	std::shared_ptr<SEP::float2DReg> model(new SEP::float2DReg(modelHyper));
	modelFile->readFloatStream(model);
	int nz = model->getHyper()->getAxis(1).n;
	float oz = model->getHyper()->getAxis(1).o;
	float dz = model->getHyper()->getAxis(1).d;
	int nx = model->getHyper()->getAxis(2).n;
	float ox = model->getHyper()->getAxis(2).o;
	float dx = model->getHyper()->getAxis(2).d;

    // // Create water bottom INDEX and DEPTH arrays
    std::shared_ptr<hypercube> wbIndexHyper(new hypercube(modelHyper->getAxis(2)));
    std::shared_ptr<SEP::float1DReg> wbIndex(new SEP::float1DReg(wbIndexHyper));
    std::shared_ptr<hypercube> wbDepthHyper(new hypercube(modelHyper->getAxis(2)));
    std::shared_ptr<SEP::float1DReg> wbDepth(new SEP::float1DReg(wbDepthHyper));

    // Get water velocity (velocity not padded)
    float waterVelocity = (*model->_mat)[0][0];
	std::cout << "Water velocity is " << waterVelocity << std::endl;

    // Compute water bottom indexp and depth
    #pragma omp parallel for
    for (int ix=0; ix<nx; ix++){
        int iz = 0;
        while ( (*model->_mat)[ix][iz] == waterVelocity ){
            (*wbIndex->_mat)[ix] = iz;
            (*wbDepth->_mat)[ix] = (iz-1)*dz+oz;
            iz++;
        }
    }

    // Write water bottom depth and index
    std::shared_ptr<SEP::genericRegFile> wbIndexFile, wbDepthFile;
    wbIndexFile = io->getRegFile(std::string("wbIndex"), usageOut);
    wbIndexFile->setHyper(wbIndexHyper);
    wbIndexFile->writeDescription();
	wbIndexFile->writeFloatStream(wbIndex);

    wbDepthFile = io->getRegFile(std::string("wbDepth"), usageOut);
    wbDepthFile->setHyper(wbDepthHyper);
    wbDepthFile->writeDescription();
    wbDepthFile->writeFloatStream(wbDepth);

    std::cout << "Done computing water-bottom index" << std::endl;

    if (step == 1) {

        std::cout << "STEP 1" << std::endl;

        ////////////////////////////////////////////////////////////////////////////
        ////////////////////////// Compute salt mask ///////////////////////////////
        ////////////////////////////////////////////////////////////////////////////

    	// Manual input: coordinates of box containing the salt body
    	float zMaxSalt=11.200; // [km]
    	float zMaxCarapace=3.500; // [km]
    	float xMinSalt=2.800; // [km]
    	float xMaxSalt=23.500; // [km]
    	float velMinSalt=4440; // [m/s]
    	float velMaxSalt=4500; // [m/s]
    	float velMinCarapace=4000; // [m/s]

        // // Allocate salt mask
    	std::shared_ptr<genericRegFile> saltMaskFile = io->getRegFile(std::string("saltMask"), usageOut);
     	std::shared_ptr<SEP::float2DReg> saltMask(new SEP::float2DReg(modelHyper));
        saltMask->scale(0.0); // Initialize to zero

        // Fill in salt mask
        #pragma omp parallel for
        for (int ix=0; ix<nx; ix++){
            float x = ox+(ix-1)*dx;
            if (x > xMinSalt && x < xMaxSalt){
                float zWaterBottom = (*wbDepth->_mat)[ix];
                for (int iz=0; iz<nz; iz++){
                    float z = oz + (iz-1) * dz;
                    if (z > zWaterBottom && z < zMaxSalt){

                        // Case 1: you are in the depth range where there are carapaces
                        if (z <= zMaxCarapace && (*model->_mat)[ix][iz] >= velMinCarapace){
                            (*saltMask->_mat)[ix][iz] = 1.0;
                        }
                        // Case 2: you are in the depth range where there are carapaces
                        if (z >= zMaxCarapace && (*model->_mat)[ix][iz] >= velMinSalt && (*model->_mat)[ix][iz] <= velMaxSalt){
                            (*saltMask->_mat)[ix][iz] = 1.0;
                        }
                    }
                }
            }
        }

        // Write water bottom depth and index
        saltMaskFile->setHyper(modelHyper);
        saltMaskFile->writeDescription();
    	saltMaskFile->writeFloatStream(saltMask);

        std::cout << "Done computing salt mask" << std::endl;

        ////////////////////////////////////////////////////////////////////////////
        /////////////////// Fill in water layer with sediment velocity /////////////
        ////////////////////////////////////////////////////////////////////////////
        std::shared_ptr<genericRegFile> modelFillFile = io->getRegFile(std::string("modelFill"), usageOut);
        std::shared_ptr<SEP::float2DReg> modelFill(new SEP::float2DReg(modelHyper));
        #pragma omp parallel for
        for (int ix=0; ix<nx; ix++){
            int index = (*wbIndex->_mat)[ix]+1;
            for (int iz=0; iz<index; iz++){
                (*modelFill->_mat)[ix][iz] = (*model->_mat)[ix][index];
            }
            for (int iz=index; iz<nz; iz++){
                (*modelFill->_mat)[ix][iz] = (*model->_mat)[ix][iz];
            }
        }

        // Write water bottom depth and index
        modelFillFile->setHyper(modelHyper);
        modelFillFile->writeDescription();
    	modelFillFile->writeFloatStream(modelFill);

        std::cout << "Done filling in the water velocity with sediment velocity" << std::endl;

        ////////////////////////////////////////////////////////////////////////////
        ////////// Replace salt velocity with interpolated sediment velocity ///////
        ////////////////////////////////////////////////////////////////////////////

        // Read search parameters
    	int elementCountMin = par->getInt("elementCountMin",50);
        int zRadiusInit = par->getInt("zRadius",5);
        int xRadiusInit = par->getInt("xRadius",20);
        int dzRadius = par->getInt("dzRadius",10);
        int dxRadius = par->getInt("dxRadius",10);
        int interpType = par->getInt("type",1);

        std::cout << "zRadius= " << zRadiusInit << std::endl;
        std::cout << "xRadius= " << xRadiusInit << std::endl;
        std::cout << "elementCountMin= " << elementCountMin << std::endl;

        // Allocate arrays
        std::shared_ptr<genericRegFile> elementCountFile = io->getRegFile(std::string("elementCount"), usageOut);
        std::shared_ptr<SEP::float2DReg> elementCount(new SEP::float2DReg(modelHyper));
        std::shared_ptr<genericRegFile> dataFile = io->getRegFile(std::string("data"), usageOut);
        std::shared_ptr<SEP::float2DReg> data(new SEP::float2DReg(modelHyper));

        // Fill data with the values of modelFill
        #pragma omp parallel for collapse(2)
        for (int ix=0; ix<nx; ix++){
            for (int iz=0; iz<nz; iz++){
                (*data->_mat)[ix][iz] = (*modelFill->_mat)[ix][iz];
                (*elementCount->_mat)[ix][iz] = -1.0;
            }
        }

        if (interpType == 1){
            // Loop over model points
            #pragma omp parallel for collapse(2)
            for (int ix=0; ix<nx; ix++){
                for (int iz=0; iz<nz; iz++){

                    // Only operate on salt points
                    if ( (*saltMask->_mat)[ix][iz] == 1 ){

                        // Initialize cumulative sum
                        float velSum=0.0;
                        (*elementCount->_mat)[ix][iz] = 0.0; // Set the element count for this point to zero
                        int zRadius=zRadiusInit;
                        int xRadius=xRadiusInit;

                        for (int ix2=0; ix2<nx; ix2++){

                            if ((*saltMask->_mat)[ix2][iz] != 1){
                                velSum=velSum+(*modelFill->_mat)[ix2][iz];
                                (*elementCount->_mat)[ix][iz]++;
                            }
                        }
                        // Compute average velocity
                        (*data->_mat)[ix][iz] = velSum / (*elementCount->_mat)[ix][iz];

                    }
                }
            }
        }
        if (interpType == 2){

            // Loop over model points
            #pragma omp parallel for collapse(2)
            for (int ix=0; ix<nx; ix++){
                for (int iz=0; iz<nz; iz++){

                    // Only operate on salt points
                    if ( (*saltMask->_mat)[ix][iz] == 1 ){

                        // Initialize cumulative sum
                        float velSum=0.0;
                        (*elementCount->_mat)[ix][iz] = 0.0; // Set the element count for this point to zero
                        int zRadius=zRadiusInit;
                        int xRadius=xRadiusInit;

                        // Loop until you found enough elements to interpolate from
                        while ( (*elementCount->_mat)[ix][iz] < elementCountMin ){

                            // Search inside the box
                            for (int ix2=ix-xRadius; ix2<ix+xRadius; ix2++){

                                // Check index is not out of bounds
                                if (ix2>=0 && ix2<nx){
                                    // Loop in z
                                    for (int iz2=iz-zRadius; iz2<iz+zRadius; iz2++){
                                        // Check index is not out of bounds
                                        if (iz2>=0 && iz2<nz){

                                            // Check if the proposed point is not salt
                                            if ((*saltMask->_mat)[ix2][iz2] != 1){
                                                // Add contribution of this point
                                                velSum=velSum+(*modelFill->_mat)[ix2][iz2];
                                                (*elementCount->_mat)[ix][iz]++;
                                            }
                                        }
                                    }
                                }
                            }

                            // If not enough points have been found, increase box radius
                            zRadius=zRadius+dzRadius;
                            xRadius=xRadius+dxRadius;
                        }

                        // Compute average velocity
                        (*data->_mat)[ix][iz] = velSum / (*elementCount->_mat)[ix][iz];
                    }
                }
            }
        }

        // Write water bottom depth and index
        modelFillFile->setHyper(modelHyper);
        modelFillFile->writeDescription();
    	modelFillFile->writeFloatStream(modelFill);

        // Write element count
        elementCountFile->setHyper(modelHyper);
        elementCountFile->writeDescription();
    	elementCountFile->writeFloatStream(elementCount);

        // Write data
        dataFile->setHyper(modelHyper);
        dataFile->writeDescription();
    	dataFile->writeFloatStream(data);

    }

    if (step == 2) {

        std::cout << "STEP 2" << std::endl;

        // Read smoothed model
        std::shared_ptr<SEP::genericRegFile> modelSmoothFile = io->getRegFile("modelSmooth",usageIn);
    	std::shared_ptr<SEP::float2DReg> modelSmooth(new SEP::float2DReg(modelHyper));
        modelSmoothFile->readFloatStream(modelSmooth);
        std::shared_ptr<genericRegFile> modelInitialFile = io->getRegFile(std::string("modelInitial"), usageOut);
        std::shared_ptr<SEP::float2DReg> modelInitial(new SEP::float2DReg(modelHyper));

        // Replace top layer of smoothed model with water velocity
        #pragma omp parallel for
        for (int ix=0; ix<nx; ix++){
            for (int iz=0; iz<nz; iz++){
                if (iz < (*wbIndex->_mat)[ix]+1){
                    (*modelInitial->_mat)[ix][iz] = (*model->_mat)[ix][iz];
                } else{
                    (*modelInitial->_mat)[ix][iz] = (*modelSmooth->_mat)[ix][iz];
                }
            }
        }

        // Write smooth model
        modelInitialFile->setHyper(modelHyper);
        modelInitialFile->writeDescription();
    	modelInitialFile->writeFloatStream(modelInitial);

    }

    return 0;
}
