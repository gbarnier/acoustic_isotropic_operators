#include<AcousticNonlinearModeling.h>
using namespace giee;

AcousticNonlinearModeling::AcousticNonlinearModeling(
                          std::shared_ptr<float1DReg> sourceFunctionCourse,
                          std::shared_ptr<float2DReg> recFieldRecord,
                          std::shared_ptr<float2DReg> vel, int velPadx, int velPadz,
                          int desample, bool verbose){
  //check that that dimensions make sense
  //receiver locations should be less than or equal to model points in x direction
  if(recFieldRecord->getHyper()->getAxis(2).n > vel->getHyper()->getAxis(2).n){
    std::cerr << "More x recording locations than x model locations\n";
    std::terminate();
  }
  //receiver recording time should be that of input source function
  if(recFieldRecord->getHyper()->getAxis(1).n != sourceFunctionCourse->getHyper()->getAxis(1).n){
    std::cerr << "Receiver recording different than input source function\n";
    std::terminate();
  }

  //set private scalar variables
  _verbose = verbose;
  _vel = vel;
  _velPadx = velPadx;
  _velPadz = velPadz;
  _nx = vel->getHyper()->getAxis(1).n;
  _nz = vel->getHyper()->getAxis(2).n;
  _nxp = _nx + 2*_velPadx;
  _nzp = _nx + 2*_velPadz;
  _maxVel = _vel->absMax();
  _minVel = _vel->min();
  _desample = desample;
  _dw = sourceFunctionCourse->getHyper()->getAxis(1).d;
  _dt = _dw/_desample;
  _nw = sourceFunctionCourse->getHyper()->getAxis(1).n;
  _nt = _nw*_desample;
  _ot = sourceFunctionCourse->getHyper()->getAxis(1).o;
  _ox = vel->getHyper()->getAxis(1).o;
  _oz = vel->getHyper()->getAxis(2).o;
  _dx = vel->getHyper()->getAxis(1).d;
  _dz = vel->getHyper()->getAxis(2).d;
  _oxp = _ox-_velPadx*_dx;
  _ozp = _ox-_velPadx*_dx;

  //check numerical stability
    //-calculate _minVel _maxVel
  _CFL = dt*_maxVel/std::min({vel->getHyper()->getAxis(1).d,vel->getHyper()->getAxis(2).d })
  if(_verbose && _CFL < 0.2) std::cerr << "WARNING CFL < 0.2\n";

  //allocate necessary 1D, 2D, 3D floatRegs
  _sourceFunctionCourse = sourceFunctionCourse;
  _sourceFunctionFine = new float1DReg(SEP::axis tAxisFine(_nt,_ot,_dt));
  _velPadded = new float2DReg(SEP::axis xAxisFine(_nxp,_oxp,_dx), SEP::axis zAxisFine(_nzp,_ozp,_dz))
  _pressure = new float3DReg(xAxisFine, zAxisFine, tAxisFine);
  _pressureProp = new float3DReg(_pressure->getHyper());
  _recFieldFine = new float2DReg(xAxisCourse,tAxisFine);
  _recFieldFullRecord = new float2DReg(xAxisCourse,SEP::axis tAxisCourse(_nw,_ot,_dw));

  //initialize operators
    //-Ls
  _Ls = new InterpSource(_sourceFunctionCourse, _sourceFunctionFine);

    //-Ks
  _Ks = new PadSource(_sourceFunctionFine, _pressure, isz, isx);

    //-G //not done
  _S = new ScaleSourceAcousticMonopole(_pressure,_pressure,_velPadded);
  _C56 = new PropagateStepperAcoustic()
  _G = new PropagateAcoustic(_pressure,_pressureProp,ScaleSourceOp,StepperOp);

    //-Kr
  _Kr = new PadRec(_recFieldFine,_pressureProp,nt-1);

    //-Lr //not done with InterpRec
  _Lr = new InterpRec(_recFieldRecord, _recFieldFine, dataCoordinates,upSample);

  //if verbose write out params
  if(_verbose){
    std::cerr << "Temporal Nyquist frequency = " << 1/(2*_dt) << std::endl;
    std::cerr << "Number of samples for 50 Hz wave = ", (_minVel/(50.*std::max({vel->getHyper()->getAxis(1).d,vel->getHyper()->getAxis(2).d))) << std::endl;
    std::cerr << "CFL number = " << _CFL << std::endl;
    std::cerr << "Min vel [m/s] = " << _minVel << std::endl;
    std::cerr << "Min vel [m/s] = " << _maxVel << std::endl;
  }

}
void AcousticNonlinearModeling::forward(){
  /* Ls - Interp Source Forward ##############################################*/
    // 1D->1D

  /* Ks - Pad Source Forward #################################################*/
    // 1D->3D

  /* G - Propagate Forward (includes source scaling)##########################*/
    //3D->3D

  /* Kr* - Pad Receiver Adjoint (Truncate) ###################################*/
    //3D->2D

  /* Lr* - Interp Rec Adjoint ################################################*/
    //2D->2D
}
void AcousticNonlinearModeling::adjoint(){
  /* Lr - Interp Rec Forward #################################################*/

  /* Kr - Pad Receiver Forward ###############################################*/

  /* G* - Propagate Ajoint (includes source scaling)##########################*/

  /* Ks* - Pad Source Adjoint (Truncate) #####################################*/

  /* Ls* - Interp Source Adjoint #############################################*/
}
