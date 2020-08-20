#include <TruncateSpatialReg.h>
#include <algorithm>
using namespace SEP;

TruncateSpatialReg::TruncateSpatialReg(const std::shared_ptr<SEP::float3DReg>model,
                                 const std::shared_ptr<SEP::float3DReg>data)
{
  // data[3][2][1] - first dimension is time, second dimension is z, third
  // dimenstion is x. The x and z locaitons to pull/push from/to model are
  // gathered from data hypercube. This allows the data to be made of traces
  // with regular sampling in the model space.

  // build xCoordinates and zCoordinates
  n1d = data->getHyper()->getAxis(1).n;
  n2d = data->getHyper()->getAxis(2).n;
  n3d = data->getHyper()->getAxis(3).n;
  o1d = data->getHyper()->getAxis(1).o;
  o2d = data->getHyper()->getAxis(2).o;
  o3d = data->getHyper()->getAxis(3).o;
  d1d = data->getHyper()->getAxis(1).d;
  d2d = data->getHyper()->getAxis(2).d;
  d3d = data->getHyper()->getAxis(3).d;

  n1m = model->getHyper()->getAxis(1).n;
  n2m = model->getHyper()->getAxis(2).n;
  n3m = model->getHyper()->getAxis(3).n;
  o1m = model->getHyper()->getAxis(1).o;
  o2m = model->getHyper()->getAxis(2).o;
  o3m = model->getHyper()->getAxis(3).o;
  d1m = model->getHyper()->getAxis(1).d;
  d2m = model->getHyper()->getAxis(2).d;
  d3m = model->getHyper()->getAxis(3).d;

  // model and data should have same time axis (3)size
  assert(o3d == o3m);
  assert(n3d == n3m);
  assert(d3d == d3m);

  //#pragma omp parallel for

  for (int i2d = 0; i2d < n2d; i2d++) {
    float fx = o2d + (float)i2d * d2d;
    float ixm = ((float)fx-(float)o2m)/(float)d2m;
    float fxm = ixm*(float)d2m+(float)o2m;
    //std::cerr << "fx:" << fx << " o2m:" << o2m << " d2m:" << d2m << " fx-o2m:" << fx-o2m << " (fx-o2m)/d2m:" << (fx-o2m)/d2m << std::endl;
    //std::cerr << "ix:" << i2d << " fx:" << fx << " ixm:" << ixm << " fxm:" << fxm << std::endl << std::endl;
    assert(ixm >= 0); //assert data in in domain of model
    assert(fx-fxm < 0.000001); //assert they fall on same grid
  }

  for (int i1d = 0; i1d < n1d; i1d++) {
    float fz = o1d + i1d * d1d;
    float izm = (fz-o1m)/d1m;
    float fzm = izm*d1m+o1m;
    //std::cerr << "iz:" << i1d << " fz:" << fz << " izm:" << izm << " fzm:" << fzm << std::endl;
    assert(izm >= 0); //assert data in in domain of model
    assert(fz-fzm < 0.000001); //assert they fall on same grid
  }



  // set domain and range
  setDomainRange(model, data);

}

void TruncateSpatialReg::forward(const bool                         add,
                          const std::shared_ptr<SEP::float3DReg>model,
                          std::shared_ptr<SEP::float3DReg>      data) const{

  assert(checkDomainRange(model, data));

  if (!add) data->scale(0.);

  std::shared_ptr<float3D> d = data->_mat;
  const std::shared_ptr<float3D> m = model->_mat;


  #pragma omp parallel for collapse(3)
  for (int i3d = 0; i3d < n3d; i3d++) {
    for (int i2d = 0; i2d < n2d; i2d++) {
      for (int i1d = 0; i1d < n1d; i1d++) {
        float fx = o2d + (float)i2d * d2d;
        int ixm = (fx-o2m)/d2m;
        float fz = o1d + (float)i1d * d1d;
        int izm = (fz-o1m)/d1m;
        // std::cerr << "(*m)[" << i3d << "][" << ixm << "][" << izm << "]\n";
        // std::cerr << "(*d)[" << i3d << "][" << i2d << "][" << i1d << "]\n\n";
        (*d)[i3d][i2d][i1d] += (*m)[i3d][ixm][izm];
      }
    }
  }

}

void TruncateSpatialReg::adjoint(const bool                         add,
                          std::shared_ptr<SEP::float3DReg>      model,
                          const std::shared_ptr<SEP::float3DReg>data) const{
  assert(checkDomainRange(model, data));

  if (!add) model->scale(0.);

  std::shared_ptr<float3D> m = model->_mat;

  const std::shared_ptr<float3D> d = data->_mat;

  #pragma omp parallel for collapse(3)
  for (int i3d = 0; i3d < n3d; i3d++) {
    for (int i2d = 0; i2d < n2d; i2d++) {
      for (int i1d = 0; i1d < n1d; i1d++) {
        float fx = o2d + (float)i2d * d2d;
        int ixm = (fx-o2m)/d2m;
        float fz = o1d + (float)i1d * d1d;
        int izm = (fz-o1m)/d1m;
        // std::cerr << "(*m)[" << i3d << "][" << ixm << "][" << izm << "]\n";
        // std::cerr << "(*d)[" << i3d << "][" << i2d << "][" << i1d << "]\n\n";
        (*m)[i3d][ixm][izm] += (*d)[i3d][i2d][i1d];
      }
    }
  }

}
// //
// // regular grid
// void TruncateSpatialReg::pullToData(const std::shared_ptr<SEP::float3DReg>model,
//                                   std::shared_ptr<SEP::float3DReg>      data) {
//   const std::shared_ptr<float3D> m = model->_mat;
//
//   std::shared_ptr<float3D> d = data->_mat;
//
//   // const std::shared_ptr<float1D> xCoord =
//   //   ((std::dynamic_pointer_cast<float1DReg>(_xCoordinates))->_mat);
//   // const std::shared_ptr<float1D> zCoord =
//   //   ((std::dynamic_pointer_cast<float1DReg>(_zCoordinates))->_mat);
//   const std::shared_ptr<float2D> recCoordArray = _recCoordinates->_mat;
//   int n1 = model->getHyper()->getAxis(1).n;
//   float o3d = data->getHyper()->getAxis(3).o;
//   float d3d = data->getHyper()->getAxis(3).d;
//   float o2d = data->getHyper()->getAxis(2).o;
//   float d2d = data->getHyper()->getAxis(2).d;
//   float o3m = model->getHyper()->getAxis(3).o;
//   float d3m = model->getHyper()->getAxis(3).d;
//   float o2m = model->getHyper()->getAxis(2).o;
//   float d2m = model->getHyper()->getAxis(2).d;
//
//   // for each data[2][1] find the x and z coordinate and push to
//   // model
//   for (int id = 0; id < _recCoordinates->getHyper()->getAxis(1).n; id++) {
//     // using xCoord find model index on axis 3
//     // float fx  = (*xCoord)[id];
//     float fx  = (*recCoordArray)[1][id];
//     int   im3 = round((fx - o3m) / d3m);
//     int   id3 = round((fx - o3d) / d3d);
//
//     // float fz  = (*zCoord)[id];
//     float fz  = (*recCoordArray)[0][id];
//     int   im2 = round((fz - o2m) / d2m);
//     int   id2 = round((fz - o2d) / d2d);
//     #pragma omp parallel for
//
//     for (int i1 = 0; i1 < n1; i1++) {
//       (*d)[id3][id2][i1] += (*m)[im3][im2][i1];
//     }
//   }
// }
// //
// // regular grid
// void TruncateSpatialReg::pushToModel(std::shared_ptr<giee::Vector>          model,
//                                   const std::shared_ptr<giee::float3DReg>data) {
//   const std::shared_ptr<float3D> m = model->_mat;
//
//   std::shared_ptr<float3D> d = data->_mat;
//
//   // const std::shared_ptr<float1D> xCoord =
//   //   ((std::dynamic_pointer_cast<float1DReg>(_xCoordinates))->_mat);
//   // const std::shared_ptr<float1D> zCoord =
//   //   ((std::dynamic_pointer_cast<float1DReg>(_zCoordinates))->_mat);
//   const std::shared_ptr<float2D> recCoordArray = _recCoordinates->_mat;
//   int n1 = model->getHyper()->getAxis(1).n;
//   float o3d = data->getHyper()->getAxis(3).o;
//   float d3d = data->getHyper()->getAxis(3).d;
//   float o2d = data->getHyper()->getAxis(2).o;
//   float d2d = data->getHyper()->getAxis(2).d;
//   float o3m = model->getHyper()->getAxis(3).o;
//   float d3m = model->getHyper()->getAxis(3).d;
//   float o2m = model->getHyper()->getAxis(2).o;
//   float d2m = model->getHyper()->getAxis(2).d;
//
//   // for each data[2][1] find the x and z coordinate and push to
//   // model
//   for (int id = 0; id < _recCoordinates->getHyper()->getAxis(1).n; id++) {
//     // using xCoord find model index on axis 3
//     // float fx  = (*xCoord)[id];
//     float fx  = (*recCoordArray)[1][id];
//     int   im3 = round((fx - o3m) / d3m);
//     int   id3 = round((fx - o3d) / d3d);
//
//     // float fz  = (*zCoord)[id];
//     float fz  = (*recCoordArray)[0][id];
//     int   im2 = round((fz - o2m) / d2m);
//     int   id2 = round((fz - o2d) / d2d);
//
// #pragma omp parallel for
//
//     for (int i1 = 0; i1 < n1; i1++) {
//       (*m)[im3][im2][i1] += (*d)[id3][id2][i1];
//     }
//   }
// }
