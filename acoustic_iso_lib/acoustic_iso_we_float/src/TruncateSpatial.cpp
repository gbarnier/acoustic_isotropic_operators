#include <TruncateSpatial.h>
#include <algorithm>

// TruncateSpatial::TruncateSpatial(const
// std::shared_ptr<SEP::float3DReg>model,
//                                  const std::shared_ptr<SEP::float2DReg>data,
//                                  const
// std::shared_ptr<SEP::float1DReg>xCoordinates,
//                                  const
// std::shared_ptr<SEP::float1DReg>zCoordinates)
TruncateSpatial::TruncateSpatial(const std::shared_ptr<SEP::float3DReg>model,
                                 const std::shared_ptr<SEP::float2DReg>data,
                                 const std::shared_ptr<SEP::float2DReg>recCoordinates)
{
  // data[2][1] - first dimension is time, second dimension is trace. Each trace
  // should have an (x,z) coordinate pair from xCoordinates and zCoordinates,
  // respectively. This allows the data to be made of traces with irregular
  // sampling in the model space.
  assert(recCoordinates->getHyper()->getAxis(
           1).n == data->getHyper()->getAxis(
           2).n);
  assert(recCoordinates->getHyper()->getAxis(
           2).n == 2);

  // model and data should have same fast axis size
  assert(model->getHyper()->getAxis(1).n == data->getHyper()->getAxis(1).n);

  // every data (x,z) coordinate should corespond to a location in the model
  // space
  const std::shared_ptr<float2D> recCoordArray =
    ((std::dynamic_pointer_cast<float2DReg>(recCoordinates))->_mat);
  float o3m = model->getHyper()->getAxis(3).o;
  float d3m = model->getHyper()->getAxis(3).d;
  int   n3m = model->getHyper()->getAxis(3).n;
  float o2m = model->getHyper()->getAxis(2).o;
  float d2m = model->getHyper()->getAxis(2).d;
  int   n2m = model->getHyper()->getAxis(2).n;

  #pragma omp parallel for

  for (int id = 0; id < recCoordinates->getHyper()->getAxis(1).n; id++) {
    // float fx = (*xCoord)[id];
    // float fz = (*zCoord)[id];
    float fx = (*recCoordArray)[1][id];
    float fz = (*recCoordArray)[0][id];

    bool exists = 0;

    for (int i3m = 0; i3m < n3m; i3m++) {
      for (int i2m = 0; i2m < n2m; i2m++) {
        if (((o3m + d3m * i3m) - fx < .0001) &&
            ((o2m + d2m * i2m) - fz < .0001)) {
          exists = 1;
          break;
        }

        if (exists) break;
      }
    }
    assert(exists == 1);
  }

  // set domain and range
  setDomainRange(model, data);

  // _xCoordinates = xCoordinates;
  // _zCoordinates = zCoordinates;
  _recCoordinates = recCoordinates;
}
//
// TruncateSpatial::TruncateSpatial(const std::shared_ptr<SEP::float3DReg>model,
//                                  const std::shared_ptr<SEP::float3DReg>data)
// {
//   // data[3][2][1] - first dimension is time, second dimension is z, third
//   // dimenstion is x. The x and z locaitons to pull/push from/to model are
//   // gathered from data hypercube. This allows the data to be made of traces
//   // with regular sampling in the model space.
//
//   // model and data should have same fast axis size
//   assert(model->getHyper()->getAxis(1).n == data->getHyper()->getAxis(1).n);
//
//   // build xCoordinates and zCoordinates
//   int   n3d = data->getHyper()->getAxis(3).n;
//   int   n2d = data->getHyper()->getAxis(2).n;
//   float o3d = data->getHyper()->getAxis(3).o;
//   float o2d = data->getHyper()->getAxis(2).o;
//   float d3d = data->getHyper()->getAxis(3).d;
//   float d2d = data->getHyper()->getAxis(2).d;
//
//   // float1D xCoordArray(boost::extents[n3d *n2d]);
//   // float1D zCoordArray(boost::extents[n3d *n2d]);
//   float2D recCoordArray(boost::extents[2][n3d *n2d]);
//
//   #pragma omp parallel for
//
//   for (int i3 = 0; i3 < n3d; i3++) {
//     float fx = o3d + i3 * d3d;
//
//     #pragma omp parallel for
//
//     for (int i2 = 0; i2 < n2d; i2++) {
//       float fz = o2d + i2 * d2d;
//
//       // xCoordArray[i2 * n3d + i3] = fx;
//       // zCoordArray[i2 * n3d + i3] = fz;
//       recCoordArray[1][i2 * n3d + i3] = fx;
//       recCoordArray[0][i2 * n3d + i3] = fz;
//     }
//   }
//
//   // std::shared_ptr<SEP::float1DReg> xCoordinates(new float1DReg(n3d
//   // *n2d,xCoordArray));
//   // std::shared_ptr<SEP::float1DReg> zCoordinates(new float1DReg(n3d
//   // *n2d,zCoordArray));
//   // _xCoordinates.reset(new float1DReg(n3d * n2d, xCoordArray));
//   // _zCoordinates.reset(new float1DReg(n3d * n2d, zCoordArray));
//   _recCoordinates.reset(new float2DReg(n3d * n2d, 2, recCoordArray));
//
//   // every data (x,z) coordinate should corespond to a location in the model
//   // space
//   // const std::shared_ptr<float1D> xCoord =
//   //   ((std::dynamic_pointer_cast<float1DReg>(_xCoordinates))->_mat);
//   // const std::shared_ptr<float1D> zCoord =
//   //   ((std::dynamic_pointer_cast<float1DReg>(_zCoordinates))->_mat);
//   float o3m = model->getHyper()->getAxis(3).o;
//   float d3m = model->getHyper()->getAxis(3).d;
//   int   n3m = model->getHyper()->getAxis(3).n;
//   float o2m = model->getHyper()->getAxis(2).o;
//   float d2m = model->getHyper()->getAxis(2).d;
//   int   n2m = model->getHyper()->getAxis(2).n;
//
//   for (int id = 0; id < _recCoordinates->getHyper()->getAxis(1).n; id++) {
//     // float     fx      = (*xCoord)[id];
//     // float     fz      = (*zCoord)[id];
//     float fx = recCoordArray[1][id];
//     float fz = recCoordArray[0][id];
//
//     bool exists = 0;
//
//     for (int i3m = 0; i3m < n3m; i3m++) {
//       for (int i2m = 0; i2m < n2m; i2m++) {
//         if (((o3m + d3m * i3m) - fx < .0001) &&
//             ((o2m + d2m * i2m) - fz < .0001)) {
//           exists = 1;
//           break;
//         }
//
//         if (exists) break;
//       }
//     }
//     assert(exists == 1);
//   }
//
//   // set domain and range
//   setDomainRange(model, data);
//
//   // _xCoordinates = xCoordinates;
//   // _zCoordinates = zCoordinates;
// }

void TruncateSpatial::forward(const bool                         add,
                              const std::shared_ptr<SEP::float3DReg>model,
                              std::shared_ptr<SEP::float2DReg>      data) const{
  assert(checkDomainRange(model, data));

  if (!add) data->scale(0.);

  // std::shared_ptr<float2DReg> dataIrreg = std::dynamic_pointer_cast<float2DReg>(
  //   data);

  // if (!dataIrreg) {
  //   std::shared_ptr<float3DReg> dataReg = std::dynamic_pointer_cast<float3DReg>(
  //     data);
  //   pullToData(model, dataReg);
  // }
  // else {
  //   pullToData(model, dataIrreg);
  // }


  const std::shared_ptr<float3D> m =
    ((std::dynamic_pointer_cast<float3DReg>(model))->_mat);

  std::shared_ptr<float2D> d = data->_mat;

  // const std::shared_ptr<float1D> xCoord =
  //   ((std::dynamic_pointer_cast<float1DReg>(_xCoordinates))->_mat);
  // const std::shared_ptr<float1D> zCoord =
  //   ((std::dynamic_pointer_cast<float1DReg>(_zCoordinates))->_mat);
  const std::shared_ptr<float2D> recCoordArray =
    ((std::dynamic_pointer_cast<float2DReg>(_recCoordinates))->_mat);
  int n1 =
    (std::dynamic_pointer_cast<float3DReg>(model))->getHyper()->getAxis(1).n;
  float o3m =
    (std::dynamic_pointer_cast<float3DReg>(model))->getHyper()->getAxis(
      3).o;
  float d3m =
    (std::dynamic_pointer_cast<float3DReg>(model))->getHyper()->getAxis(
      3).d;
  float o2m =
    (std::dynamic_pointer_cast<float3DReg>(model))->getHyper()->getAxis(
      2).o;
  float d2m =
    (std::dynamic_pointer_cast<float3DReg>(model))->getHyper()->getAxis(
      2).d;

  // for each data trace find the x and z coordinate and extract from
  // model
  for (int id = 0; id < data->getHyper()->getAxis(2).n; id++) {
    // using xCoord find model index on axis 3
    // float fx  = (*xCoord)[id];
    float fx  = (*recCoordArray)[1][id];
    int   im3 = round((fx - o3m) / d3m);

    // int im3 = (*recCoordArray)[1][id];

    // using zCoord find model index on axis 2
    // float fz  = (*zCoord)[id];
    float fz  = (*recCoordArray)[0][id];
    int   im2 = round((fz - o2m) / d2m);

    // int im2 = (*recCoordArray)[0][id];

    #pragma omp parallel for

    for (int i1 = 0; i1 < n1; i1++) {
      // std::cerr << "id=" << id << " i1=" << i1 << " im3=" << im3 << " im2="
      // <<
      //   im2 << std::endl;
      (*d)[id][i1] += (*m)[im3][im2][i1];
    }
  }

}

void TruncateSpatial::adjoint(const bool                         add,
                              std::shared_ptr<SEP::float3DReg>      model,
                              const std::shared_ptr<SEP::float2DReg>data) const{
  assert(checkDomainRange(model, data));

  if (!add) model->scale(0.);

  // std::shared_ptr<float2DReg> dataIrreg = std::dynamic_pointer_cast<float2DReg>(
  //   data);

  // if (!dataIrreg) {
  //   std::shared_ptr<float3DReg> dataReg = std::dynamic_pointer_cast<float3DReg>(
  //     data);
  //   pushToModel(model, dataReg);
  // }
  // else {
  //   pushToModel(model, dataIrreg);
  // }


  std::shared_ptr<float3D> m =
    ((std::dynamic_pointer_cast<float3DReg>(model))->_mat);
  const std::shared_ptr<float2D> d = data->_mat;

  // const std::shared_ptr<float1D> xCoord =
  //   ((std::dynamic_pointer_cast<float1DReg>(_xCoordinates))->_mat);
  // const std::shared_ptr<float1D> zCoord =
  //   ((std::dynamic_pointer_cast<float1DReg>(_zCoordinates))->_mat);
  const std::shared_ptr<float2D> recCoordArray =
    ((std::dynamic_pointer_cast<float2DReg>(_recCoordinates))->_mat);
  int n1 =
    (std::dynamic_pointer_cast<float3DReg>(model))->getHyper()->getAxis(
      1).n;
  float o3m =
    (std::dynamic_pointer_cast<float3DReg>(model))->getHyper()->getAxis(
      3).o;
  float d3m =
    (std::dynamic_pointer_cast<float3DReg>(model))->getHyper()->getAxis(
      3).d;
  float o2m =
    (std::dynamic_pointer_cast<float3DReg>(model))->getHyper()->getAxis(
      2).o;
  float d2m =
    (std::dynamic_pointer_cast<float3DReg>(model))->getHyper()->getAxis(
      2).d;

  // for each data[2][1] find the x and z coordinate and push to
  // model
  for (int id = 0; id < data->getHyper()->getAxis(2).n; id++) {
    // using xCoord find model index on axis 3
    // float fx  = (*xCoord)[id];
    float fx  = (*recCoordArray)[1][id];
    int   im3 = round((fx - o3m) / d3m);

    // float fz  = (*zCoord)[id];
    float fz  = (*recCoordArray)[0][id];
    int   im2 = round((fz - o2m) / d2m);
#pragma omp parallel for

    for (int i1 = 0; i1 < n1; i1++) {
      (*m)[im3][im2][i1] += (*d)[id][i1];
    }
  }

}

// irregular grid
void TruncateSpatial::pullToData(const std::shared_ptr<SEP::float3DReg>model,
                                 std::shared_ptr<SEP::float2DReg>  data) {
  const std::shared_ptr<float3D> m =
    ((std::dynamic_pointer_cast<float3DReg>(model))->_mat);

  std::shared_ptr<float2D> d = data->_mat;

  // const std::shared_ptr<float1D> xCoord =
  //   ((std::dynamic_pointer_cast<float1DReg>(_xCoordinates))->_mat);
  // const std::shared_ptr<float1D> zCoord =
  //   ((std::dynamic_pointer_cast<float1DReg>(_zCoordinates))->_mat);
  const std::shared_ptr<float2D> recCoordArray =
    ((std::dynamic_pointer_cast<float2DReg>(_recCoordinates))->_mat);
  int n1 =
    (std::dynamic_pointer_cast<float3DReg>(model))->getHyper()->getAxis(1).n;
  float o3m =
    (std::dynamic_pointer_cast<float3DReg>(model))->getHyper()->getAxis(
      3).o;
  float d3m =
    (std::dynamic_pointer_cast<float3DReg>(model))->getHyper()->getAxis(
      3).d;
  float o2m =
    (std::dynamic_pointer_cast<float3DReg>(model))->getHyper()->getAxis(
      2).o;
  float d2m =
    (std::dynamic_pointer_cast<float3DReg>(model))->getHyper()->getAxis(
      2).d;

  // for each data trace find the x and z coordinate and extract from
  // model
  for (int id = 0; id < data->getHyper()->getAxis(2).n; id++) {
    // using xCoord find model index on axis 3
    // float fx  = (*xCoord)[id];
    float fx  = (*recCoordArray)[1][id];
    int   im3 = round((fx - o3m) / d3m);

    // int im3 = (*recCoordArray)[1][id];

    // using zCoord find model index on axis 2
    // float fz  = (*zCoord)[id];
    float fz  = (*recCoordArray)[0][id];
    int   im2 = round((fz - o2m) / d2m);

    // int im2 = (*recCoordArray)[0][id];

    #pragma omp parallel for

    for (int i1 = 0; i1 < n1; i1++) {
      // std::cerr << "id=" << id << " i1=" << i1 << " im3=" << im3 << " im2="
      // <<
      //   im2 << std::endl;
      (*d)[id][i1] += (*m)[im3][im2][i1];
    }
  }
}
//
// // regular grid
// void TruncateSpatial::pullToData(const std::shared_ptr<SEP::Vector>model,
//                                  std::shared_ptr<SEP::float3DReg>  data) {
//   const std::shared_ptr<float3D> m =
//     ((std::dynamic_pointer_cast<float3DReg>(model))->_mat);
//
//   std::shared_ptr<float3D> d = data->_mat;
//
//   // const std::shared_ptr<float1D> xCoord =
//   //   ((std::dynamic_pointer_cast<float1DReg>(_xCoordinates))->_mat);
//   // const std::shared_ptr<float1D> zCoord =
//   //   ((std::dynamic_pointer_cast<float1DReg>(_zCoordinates))->_mat);
//   const std::shared_ptr<float2D> recCoordArray =
//     ((std::dynamic_pointer_cast<float2DReg>(_recCoordinates))->_mat);
//   int n1 =
//     (std::dynamic_pointer_cast<float3DReg>(model))->getHyper()->getAxis(
//       1).n;
//   float o3d =
//     (std::dynamic_pointer_cast<float3DReg>(data))->getHyper()->getAxis(
//       3).o;
//   float d3d =
//     (std::dynamic_pointer_cast<float3DReg>(data))->getHyper()->getAxis(
//       3).d;
//   float o2d =
//     (std::dynamic_pointer_cast<float3DReg>(data))->getHyper()->getAxis(
//       2).o;
//   float d2d =
//     (std::dynamic_pointer_cast<float3DReg>(data))->getHyper()->getAxis(
//       2).d;
//   float o3m =
//     (std::dynamic_pointer_cast<float3DReg>(model))->getHyper()->getAxis(
//       3).o;
//   float d3m =
//     (std::dynamic_pointer_cast<float3DReg>(model))->getHyper()->getAxis(
//       3).d;
//   float o2m =
//     (std::dynamic_pointer_cast<float3DReg>(model))->getHyper()->getAxis(
//       2).o;
//   float d2m =
//     (std::dynamic_pointer_cast<float3DReg>(model))->getHyper()->getAxis(
//       2).d;
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

// irregular grid
void TruncateSpatial::pushToModel(std::shared_ptr<SEP::float3DReg>          model,
                                  const std::shared_ptr<SEP::float2DReg>data) {
  std::shared_ptr<float3D> m =
    ((std::dynamic_pointer_cast<float3DReg>(model))->_mat);
  const std::shared_ptr<float2D> d = data->_mat;

  // const std::shared_ptr<float1D> xCoord =
  //   ((std::dynamic_pointer_cast<float1DReg>(_xCoordinates))->_mat);
  // const std::shared_ptr<float1D> zCoord =
  //   ((std::dynamic_pointer_cast<float1DReg>(_zCoordinates))->_mat);
  const std::shared_ptr<float2D> recCoordArray =
    ((std::dynamic_pointer_cast<float2DReg>(_recCoordinates))->_mat);
  int n1 =
    (std::dynamic_pointer_cast<float3DReg>(model))->getHyper()->getAxis(
      1).n;
  float o3m =
    (std::dynamic_pointer_cast<float3DReg>(model))->getHyper()->getAxis(
      3).o;
  float d3m =
    (std::dynamic_pointer_cast<float3DReg>(model))->getHyper()->getAxis(
      3).d;
  float o2m =
    (std::dynamic_pointer_cast<float3DReg>(model))->getHyper()->getAxis(
      2).o;
  float d2m =
    (std::dynamic_pointer_cast<float3DReg>(model))->getHyper()->getAxis(
      2).d;

  // for each data[2][1] find the x and z coordinate and push to
  // model
  for (int id = 0; id < data->getHyper()->getAxis(2).n; id++) {
    // using xCoord find model index on axis 3
    // float fx  = (*xCoord)[id];
    float fx  = (*recCoordArray)[1][id];
    int   im3 = round((fx - o3m) / d3m);

    // float fz  = (*zCoord)[id];
    float fz  = (*recCoordArray)[0][id];
    int   im2 = round((fz - o2m) / d2m);
#pragma omp parallel for

    for (int i1 = 0; i1 < n1; i1++) {
      (*m)[im3][im2][i1] += (*d)[id][i1];
    }
  }
}
//
// // regular grid
// void TruncateSpatial::pushToModel(std::shared_ptr<SEP::Vector>          model,
//                                   const std::shared_ptr<SEP::float3DReg>data) {
//   const std::shared_ptr<float3D> m =
//     ((std::dynamic_pointer_cast<float3DReg>(model))->_mat);
//
//   std::shared_ptr<float3D> d = data->_mat;
//
//   // const std::shared_ptr<float1D> xCoord =
//   //   ((std::dynamic_pointer_cast<float1DReg>(_xCoordinates))->_mat);
//   // const std::shared_ptr<float1D> zCoord =
//   //   ((std::dynamic_pointer_cast<float1DReg>(_zCoordinates))->_mat);
//   const std::shared_ptr<float2D> recCoordArray =
//     ((std::dynamic_pointer_cast<float2DReg>(_recCoordinates))->_mat);
//   int n1 =
//     (std::dynamic_pointer_cast<float3DReg>(model))->getHyper()->getAxis(
//       1).n;
//   float o3d =
//     (std::dynamic_pointer_cast<float3DReg>(data))->getHyper()->getAxis(
//       3).o;
//   float d3d =
//     (std::dynamic_pointer_cast<float3DReg>(data))->getHyper()->getAxis(
//       3).d;
//   float o2d =
//     (std::dynamic_pointer_cast<float3DReg>(data))->getHyper()->getAxis(
//       2).o;
//   float d2d =
//     (std::dynamic_pointer_cast<float3DReg>(data))->getHyper()->getAxis(
//       2).d;
//   float o3m =
//     (std::dynamic_pointer_cast<float3DReg>(model))->getHyper()->getAxis(
//       3).o;
//   float d3m =
//     (std::dynamic_pointer_cast<float3DReg>(model))->getHyper()->getAxis(
//       3).d;
//   float o2m =
//     (std::dynamic_pointer_cast<float3DReg>(model))->getHyper()->getAxis(
//       2).o;
//   float d2m =
//     (std::dynamic_pointer_cast<float3DReg>(model))->getHyper()->getAxis(
//       2).d;
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
