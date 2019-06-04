#include <G.h>
using namespace giee;
using namespace waveform;

G::G(
  const std::shared_ptr<giee::float2DReg>model,
  const std::shared_ptr<giee::float2DReg>data,
  const int                              velPadx,
  const int                              velPadz)
{
  assert(data->getHyper()->sameSize(model->getHyper()));
  setDomainRange(model, data);

  _velPadx = velPadx;
  _velPadz = velPadz;
}

void G::forward(const bool                         add,
                const std::shared_ptr<giee::Vector>model,
                std::shared_ptr<giee::Vector>      data)
{
  assert(checkDomainRange(model, data, true));

  if (!add) data->scale(0.);
  const std::shared_ptr<float2D> m =
    ((std::dynamic_pointer_cast<float2DReg>(model))->_mat);
  std::shared_ptr<float2D> d =
    ((std::dynamic_pointer_cast<float2DReg>(data))->_mat);
  int n1 =
    (std::dynamic_pointer_cast<float2DReg>(model))->getHyper()->getAxis(1).n;
  int n2 =
    (std::dynamic_pointer_cast<float2DReg>(model))->getHyper()->getAxis(2).n;
  double d1 =
    (std::dynamic_pointer_cast<float2DReg>(model))->getHyper()->getAxis(1).d;
  double d2 =
    (std::dynamic_pointer_cast<float2DReg>(model))->getHyper()->getAxis(2).d;


  for (int i1 = 0; i1 < n1; i1++) {         // z
    (*d)[0][i1] += -1 * double((*m)[0][i1]) / d2;

    for (int i2 = 1; i2 < _velPadx; i2++) { // x
      (*d)[i2][i1] += double((*m)[i2 - 1][i1] - (*m)[i2][i1]) / d2;
    }
  }


  for (int i1 = 0; i1 < n1; i1++) {                   // z
    (*d)[n2 - 1][i1] += -1 * double((*m)[n2 - 1][i1]) / d2;

    for (int i2 = n2 - _velPadx; i2 < n2 - 1; i2++) { // x
      (*d)[i2][i1] += double((*m)[i2 + 1][i1] - (*m)[i2][i1]) / d2;
    }
  }


  for (int i2 = 0; i2 < n2; i2++) {         // nx
    (*d)[i2][0] += -1 * double((*m)[i2][0]) / d1;

    for (int i1 = 1; i1 < _velPadz; i1++) { // nz
      (*d)[i2][i1] += double((*m)[i2][i1 - 1] - (*m)[i2][i1]) / d1;
    }
  }


  for (int i2 = 0; i2 < n2; i2++) {                   // nx
    (*d)[i2][n1 - 1] += -1 * double((*m)[i2][n1 - 1]) / d1;

    for (int i1 = n1 - _velPadz; i1 < n1 - 1; i1++) { // nz
      (*d)[i2][i1] += double((*m)[i2][i1 + 1] - (*m)[i2][i1]) / d1;
    }
  }
}

void G::adjoint(const bool                         add,
                const std::shared_ptr<giee::Vector>model,
                std::shared_ptr<giee::Vector>      data)
{
  assert(checkDomainRange(model, data, true));

  if (!add) model->scale(0.);
  std::shared_ptr<float2D> m =
    ((std::dynamic_pointer_cast<float2DReg>(model))->_mat);
  const std::shared_ptr<float2D> d =
    ((std::dynamic_pointer_cast<float2DReg>(data))->_mat);

  // const std::shared_ptr<float2D> vel =
  //   ((std::dynamic_pointer_cast<float2DReg>(_velPadded))->_mat);
  int n1 =
    (std::dynamic_pointer_cast<float2DReg>(model))->getHyper()->getAxis(1).n;
  int n2 =
    (std::dynamic_pointer_cast<float2DReg>(model))->getHyper()->getAxis(2).n;
  double d1 =
    (std::dynamic_pointer_cast<float2DReg>(model))->getHyper()->getAxis(1).d;
  double d2 =
    (std::dynamic_pointer_cast<float2DReg>(model))->getHyper()->getAxis(2).d;


  for (int i1 = 0; i1 < n1; i1++) {             // z
    (*m)[_velPadx - 1][i1] +=  -1 * double((*d)[_velPadx - 1][i1]) / d2;

    for (int i2 = 0; i2 < _velPadx - 1; i2++) { // x
      (*m)[i2][i1] +=  double((*d)[i2 + 1][i1] - (*d)[i2][i1]) / d2;
    }
  }


  for (int i1 = 0; i1 < n1; i1++) {                   // z
    (*m)[n2 - _velPadx][i1] += -1 * double((*d)[n2 - _velPadx][i1]) / d2;

    for (int i2 = n2 - _velPadx + 1; i2 < n2; i2++) { // x
      (*m)[i2][i1] += double((*d)[i2 - 1][i1] - (*d)[i2][i1]) / d2;
    }
  }


  for (int i2 = 0; i2 < n2; i2++) {             // nx
    (*m)[i2][_velPadz - 1] += -1 * double((*d)[i2][_velPadz - 1]) / d1;

    for (int i1 = 0; i1 < _velPadz - 1; i1++) { // nz
      (*m)[i2][i1] += double((*d)[i2][i1 + 1] - (*d)[i2][i1]) / d1;
    }
  }


  for (int i2 = 0; i2 < n2; i2++) {                   // nx
    (*m)[i2][n1 - _velPadz] += -1 * double((*d)[i2][n1 - _velPadz]) / d1;

    for (int i1 = n1 - _velPadz + 1; i1 < n1; i1++) { // nz
      (*m)[i2][i1] += double((*d)[i2][i1 - 1] - (*d)[i2][i1]) / d1;
    }
  }
}
