/********************************************//**
   Author: Stuart Farris
   Date: Dec 2017
   Description:  This will model all a series of shots forward or backward in
      time
 ***********************************************/
 #pragma once
#include <Operator.h>
#include <float1DReg.h>
#include <float2DReg.h>
#include <float3DReg.h>

/** Nonlinear acoustic wavefield modeling forward and backward in time.
 */
class AcousticNonlinearModeling : public Operator {
public:

  AcousticNonlinearModeling(std::shared_ptr<float1DReg>sourceFunctionCourse,
                            std::shared_ptr<float3DReg>recFieldRecord,
                            std::shared_ptr<float2DReg>vel,
                            int                        velPadx,
                            int                        velPadz,
                            int                        isx,
                            int                        isz,
                            int                        desample);

  /** model all shots forward in time.
   */
  void forward(const bool                         add,
               const std::shared_ptr<giee::Vector>model,
               std::shared_ptr<giee::Vector>      data);

  /** model all shots gathers backward in time
   */
  void adjoint(const bool                         add,
               std::shared_ptr<giee::Vector>      model,
               const std::shared_ptr<giee::Vector>data);


  void writeWavefield(); // write out wavefield at given time

private:

  // 3d arrays[3][2][1]
  // 2d boostarrays [2][1]
  // In order of appearance

  // Ls
  // #########################################################################
  std::shared_ptr<float1DReg>_sourceFunctionCourse; // the source passed to
                                                    // constructor injected at
                                                    // each location. [t]
  std::shared_ptr<float1DReg>_sourceFunctionFine;   // the source function to be
                                                    // injected after
                                                    // interpolating to grid
                                                    // fine enough for prop. [t]
  std::shared_ptr<float2DReg>_vel;                  // used for numerical
                                                    // stability calc [x][z]
  float _minVel, _maxVel;                           // used for numerical
                                                    // stability calc
  float _dt, _dw, _ot, _CFL;                        // _dt->prop delta time,
                                                    // _dw->source delta time,
                                                    // _ot-> time origin
  int _nt;
  int _nx, _nxp, _nz, _nzp;                         // nx and nz with and
                                                    // without padding
  float _ox, _oxp, _oz, _oxz;
  float _dx, _dz;
  int _desample;                                    // used for numerical
                                                    // stability calc
  std::shared_ptr<InterpSource>_Ls;

  // Ks
  // #########################################################################
  int _velPadx;                          // how much to pad vel in x direction
  int _velPadz;
  int _                                  // how much to pad vel in z direction
  std::shared_ptr<float2DReg>_velPadded; // vel after padding [x][z]
  std::shared_ptr<float3DReg>_pressure;  // pressure cube same size as vel with
                                         // source function added [x][z][t]
  std::shared_ptr<PadSource>_Ks;

  // G
  // ##########################################################################
  std::shared_ptr<ScaleSourceAcousticMonopole>_S;
  std::shared_ptr<PropagateStepperAcoustic>_C456;
  std::shared_ptr<PropagateAcoustic>_G;

  // Kr*
  // ########################################################################
  std::shared_ptr<float2DReg>_recFieldFine; // receiver field at every x and t
                                            // for one depth. At propagation
                                            // sampling. [x][t]
  std::shared_ptr<PadRec>_Kr;

  // Lr*
  // ########################################################################
  std::shared_ptr<float2DReg>_recFieldRecord; // receiver field at every x and t
                                              // for one depth. At recording
                                              // sampling. [x][t]
  std::sahred_ptr<InterpRec>_Lr;

  // writeWavefield()
  // ###########################################################

  bool _verbose;
};
