#include <InterpRec.h>
using namespace SEP;

InterpRec::InterpRec(const std::shared_ptr<SEP::float2DReg>model,
                     const std::shared_ptr<SEP::float2DReg>data,
                     const std::shared_ptr<SEP::float1DReg>dataCoordinates,
                     float                                  oversamp) {
  assert(data->getHyper()->getAxis(1).n == dataCoordinates->getHyper()->getAxis(
           1).n);
  assert(model->getHyper()->getAxis(2).n == data->getHyper()->getAxis(2).n);
  setDomainRange(model, data);

  _o1              = model->getHyper()->getAxis(1).o;
  _d1              = model->getHyper()->getAxis(1).d;
  _dataCoordinates = dataCoordinates;
  _scale           = 1.0 / (sqrt(oversamp));
}

void InterpRec::forward(const bool                         add,
                        const std::shared_ptr<SEP::float2DReg>model,
                        std::shared_ptr<SEP::float2DReg>      data) {
  assert(checkDomainRange(model, data));

  if (!add) data->scale(0.);
  const std::shared_ptr<float2D> m =
    ((std::dynamic_pointer_cast<float2DReg>(model))->_mat);
  std::shared_ptr<float2D> d =
    ((std::dynamic_pointer_cast<float2DReg>(data))->_mat);
  const std::shared_ptr<float1D> dloc =
    ((std::dynamic_pointer_cast<float1DReg>(_dataCoordinates))->_mat);
  int nm =
    (std::dynamic_pointer_cast<float2DReg>(model))->getHyper()->getAxis(1).n;
  int nd =
    (std::dynamic_pointer_cast<float2DReg>(data))->getHyper()->getAxis(1).n;
  int n2 =
    (std::dynamic_pointer_cast<float2DReg>(data))->getHyper()->getAxis(2).n;

  for (int i2 = 0; i2 < n2; i2++) {
    for (int i1 = 0; i1 < nd; i1++) {
      float mloc   = ((*dloc)[i1] - _o1) / _d1;
      int   mindex = mloc;

      if ((mindex >= 0) && (mindex < nm - 1)) {
        float fx = mloc - mindex;
        float gx = 1 - fx;
        (*d)[i2][i1] += gx * (*m)[i2][mindex] * _scale + fx *
                        (*m)[i2][mindex + 1] * _scale;
      }
    }
  }
}

void InterpRec::adjoint(const bool                         add,
                        std::shared_ptr<SEP::float2DReg>      model,
                        const std::shared_ptr<SEP::float2DReg>data) {
  assert(checkDomainRange(model, data));

  if (!add) model->scale(0.);
  std::shared_ptr<float2D> m =
    ((std::dynamic_pointer_cast<float2DReg>(model))->_mat);
  const std::shared_ptr<float2D> d =
    ((std::dynamic_pointer_cast<float2DReg>(data))->_mat);
  const std::shared_ptr<float1D> dloc =
    ((std::dynamic_pointer_cast<float1DReg>(_dataCoordinates))->_mat);
  int nm =
    (std::dynamic_pointer_cast<float2DReg>(model))->getHyper()->getAxis(1).n;
  int nd =
    (std::dynamic_pointer_cast<float2DReg>(data))->getHyper()->getAxis(1).n;
  int n2 =
    (std::dynamic_pointer_cast<float2DReg>(data))->getHyper()->getAxis(2).n;

  for (int i2 = 0; i2 < n2; i2++) {
    for (int i1 = 0; i1 < nd; i1++) {
      float mloc   = ((*dloc)[i1] - _o1) / _d1;
      int   mindex = mloc;

      if ((mindex >= 0) && (mindex < nm - 1)) {
        float fx = mloc - mindex;
        float gx = 1 - fx;
        (*m)[i2][mindex]     += gx * (*d)[i2][i1] * _scale;
        (*m)[i2][mindex + 1] += fx * (*d)[i2][i1] * _scale;
      }
    }
  }
}
