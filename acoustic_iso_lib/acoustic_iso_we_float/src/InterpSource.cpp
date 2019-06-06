#include <InterpSource.h>
using namespace SEP;

/*
   InterpSource::InterpSource(const std::shared_ptr<SEP::float1DReg> model,
        const std::shared_ptr<SEP::float1DReg> data,
    float oversamp){

   assert(data->getHyper()->getAxis(1).n ==
      _dataCoordinates->getHyper()->getAxis(1).n);
   setDomainRange(model,data);

   _o1=model->getHyper()->getAxis(1).o;
   _d1=model->getHyper()->getAxis(1).d;
   _dataCoordinates=dataCoordinates;
   _scale = 1.0 / (sqrt(oversamp));
   }*/

InterpSource::InterpSource(const std::shared_ptr<SEP::float1DReg>model,
                           const std::shared_ptr<SEP::float1DReg>data,
                           const std::shared_ptr<SEP::float1DReg>dataCoordinates,
                           float                                  oversamp) {
  assert(data->getHyper()->getAxis(1).n == dataCoordinates->getHyper()->getAxis(
           1).n);
  setDomainRange(model, data);

  _o1              = model->getHyper()->getAxis(1).o;
  _d1              = model->getHyper()->getAxis(1).d;
  _dataCoordinates = dataCoordinates;
  _scale           = 1.0 / (sqrt(oversamp));
}

void InterpSource::forward(const bool                         add,
                           const std::shared_ptr<SEP::float1DReg>model,
                           std::shared_ptr<SEP::float1DReg>      data) {
  assert(checkDomainRange(model, data, true));

  if (!add) data->scale(0.);
  const std::shared_ptr<float1D> m =
    ((std::dynamic_pointer_cast<float1DReg>(model))->_mat);
  std::shared_ptr<float1D> d =
    ((std::dynamic_pointer_cast<float1DReg>(data))->_mat);
  const std::shared_ptr<float1D> dloc =
    ((std::dynamic_pointer_cast<float1DReg>(_dataCoordinates))->_mat);
  int nm =
    (std::dynamic_pointer_cast<float1DReg>(model))->getHyper()->getAxis(1).n;
  int nd =
    (std::dynamic_pointer_cast<float1DReg>(data))->getHyper()->getAxis(1).n;

  for (int i1 = 0; i1 < nd; i1++) {
    float mloc   = ((*dloc)[i1] - _o1) / _d1;
    int   mindex = mloc;

    if ((mindex >= 0) && (mindex < nm - 1)) {
      float fx = mloc - mindex;
      float gx = 1 - fx;
      (*d)[i1] += gx * (*m)[mindex] * _scale + fx * (*m)[mindex + 1] * _scale;
    }
  }
}

void InterpSource::adjoint(const bool                         add,
                           std::shared_ptr<SEP::float1DReg>      model,
                           const std::shared_ptr<SEP::float1DReg>data) {
  assert(checkDomainRange(model, data, true));

  if (!add) model->scale(0.);
  std::shared_ptr<float1D> m =
    ((std::dynamic_pointer_cast<float1DReg>(model))->_mat);
  const std::shared_ptr<float1D> d =
    ((std::dynamic_pointer_cast<float1DReg>(data))->_mat);
  const std::shared_ptr<float1D> dloc =
    ((std::dynamic_pointer_cast<float1DReg>(_dataCoordinates))->_mat);
  int nm =
    (std::dynamic_pointer_cast<float1DReg>(model))->getHyper()->getAxis(1).n;
  int nd =
    (std::dynamic_pointer_cast<float1DReg>(data))->getHyper()->getAxis(1).n;

  for (int i1 = 0; i1 < nd; i1++) {
    float mloc   = ((*dloc)[i1] - _o1) / _d1;
    int   mindex = mloc;

    if ((mindex >= 0) && (mindex < nm - 1)) {
      float fx = mloc - mindex;
      float gx = 1 - fx;
      (*m)[mindex]     += gx * (*d)[i1] * _scale;
      (*m)[mindex + 1] += fx * (*d)[i1] * _scale;
    }
  }
}
