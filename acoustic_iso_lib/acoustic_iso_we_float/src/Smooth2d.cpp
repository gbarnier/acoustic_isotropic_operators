#include <Smooth2d.h>
#include <math.h>
using namespace SEP;

Smooth2d::Smooth2d(
    const std::shared_ptr<SEP::float2DReg>model,
    const std::shared_ptr<SEP::float2DReg>data,
    int                                    nfilt1,
    int                                    nfilt2)
{
  // model and data have the same dimensions
  int n1, n2;

  n1 = model->getHyper()->getAxis(1).n;
  n2 = model->getHyper()->getAxis(2).n;
  assert(n1 == data->getHyper()->getAxis(1).n);
  assert(n2 == data->getHyper()->getAxis(2).n);



  // set domain and range
  setDomainRange(model, data);

  _nfilt1 = nfilt1;
  _nfilt2 = nfilt2;

  buffer.reset(new SEP::float2DReg(data->getHyper()->getAxis(1).n+2*_nfilt1,
                                    data->getHyper()->getAxis(2).n+2*_nfilt2));
}

// forward
void Smooth2d::forward(const bool                         add,
                     const std::shared_ptr<SEP::float2DReg>model,
                     std::shared_ptr<SEP::float2DReg>      data) const {
  assert(checkDomainRange(model, data));

  if (!add) data->scale(0.);

  const std::shared_ptr<float2D> m =
    ((std::dynamic_pointer_cast<float2DReg>(model))->_mat);
  std::shared_ptr<float2D> d =
    ((std::dynamic_pointer_cast<float2DReg>(data))->_mat);
  int n1 =
    (std::dynamic_pointer_cast<float2DReg>(data))->getHyper()->getAxis(1).n;
  int n2 =
    (std::dynamic_pointer_cast<float2DReg>(data))->getHyper()->getAxis(2).n;

  buffer->scale(0);
  std::shared_ptr<float2D> b =
    ((std::dynamic_pointer_cast<float2DReg>(buffer))->_mat);
   #pragma omp parallel for collapse(2)
    for (int ix = 0; ix < n2; ix++) {
      for (int iz = 0; iz < n1; iz++) {
        (*b)[ix+_nfilt2][iz+_nfilt1] = (*m)[ix][iz];
      }
    }
std::cerr << "here0\n";
   //#pragma omp parallel for collapse(2)
  for (int i2 = 0; i2 < n2; i2++) {
    for (int i1 = 0 ; i1 < n1; i1++) {
        for(int if2=-_nfilt2; if2<_nfilt2; if2++)  {
          for(int if1=-_nfilt1; if1<_nfilt1; if1++){
        	  (*d)[i2][i1] += (*b)[i2+_nfilt2+if2][i1+_nfilt1+if1];
          }
        }
        (*d)[i2][i1] = (*d)[i2][i1]/(2*_nfilt1*2*_nfilt2); 
    }
  }
std::cerr << "here1\n";
}

// adjoint
void Smooth2d::adjoint(const bool                         add,
                     std::shared_ptr<SEP::float2DReg>      model,
                     const std::shared_ptr<SEP::float2DReg>data) const {
  assert(checkDomainRange(model, data));

  if (!add) model->scale(0.);

  std::shared_ptr<float2D> m =
    ((std::dynamic_pointer_cast<float2DReg>(model))->_mat);
  const std::shared_ptr<float2D> d =
    ((std::dynamic_pointer_cast<float2DReg>(data))->_mat);
  int n1 =
    (std::dynamic_pointer_cast<float2DReg>(data))->getHyper()->getAxis(1).n;
  int n2 =
    (std::dynamic_pointer_cast<float2DReg>(data))->getHyper()->getAxis(2).n;

  buffer->scale(0);
  std::shared_ptr<float2D> b =
    ((std::dynamic_pointer_cast<float2DReg>(buffer))->_mat);
   #pragma omp parallel for collapse(2)
    for (int ix = 0; ix < n2; ix++) {
      for (int iz = 0; iz < n1; iz++) {
        (*b)[ix+_nfilt2][iz+_nfilt1] = (*d)[ix][iz];
      }
    }

   //#pragma omp parallel for collapse(2)
  for (int i2 = 0; i2 < n2; i2++) {
    for (int i1 = 0 ; i1 < n1; i1++) {
        for(int if2=-_nfilt2; if2<_nfilt2; if2++)  {
          for(int if1=-_nfilt1; if1<_nfilt1; if1++){
        	  (*m)[i2][i1] += (*b)[i2+_nfilt2+if2][i1+_nfilt1+if1];
          }
        }
        (*m)[i2][i1] = (*m)[i2][i1]/(2*_nfilt1*2*_nfilt2); 
    }
  }
}

