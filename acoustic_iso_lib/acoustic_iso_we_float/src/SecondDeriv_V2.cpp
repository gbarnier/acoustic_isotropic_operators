#include <SecondDeriv_V2.h>

SecondDeriv_V2::SecondDeriv_V2(const std::shared_ptr<SEP::float3DReg>model,
                         const std::shared_ptr<SEP::float3DReg>data
                         ) {
std::cerr << "VERSION 8" << std::endl;
  // ensure model and data dimensions match
  assert(model->getHyper()->getAxis(1).n == data->getHyper()->getAxis(1).n);
  assert(model->getHyper()->getAxis(2).n == data->getHyper()->getAxis(2).n);
  assert(model->getHyper()->getAxis(3).n == data->getHyper()->getAxis(3).n);
  assert(model->getHyper()->getAxis(1).d == data->getHyper()->getAxis(1).d);
  assert(model->getHyper()->getAxis(2).d == data->getHyper()->getAxis(2).d);
  assert(model->getHyper()->getAxis(3).d == data->getHyper()->getAxis(3).d);

  n1 =model->getHyper()->getAxis(1).n; //z
   n2 =model->getHyper()->getAxis(2).n; //x
   n3 =model->getHyper()->getAxis(3).n; //t

  // set domain and range
  setDomainRange(model, data);

  _dt = model->getHyper()->getAxis(3).d;

  // calculate lapl coefficients

  C0t_10 = -2.927222222 / (_dt * _dt);
  C1t_10 =  1.666666667 / (_dt * _dt);
  C2t_10 = -0.23809524 / (_dt * _dt);
  C3t_10 =  0.03968254 / (_dt * _dt);
  C4t_10 = -0.00496032 / (_dt * _dt);
  C5t_10 =  0.00031746 / (_dt * _dt);

  C0t_8 = -2.84722222222 / (_dt * _dt);
  C1t_8 =  1.600000000 / (_dt * _dt);
  C2t_8 = -0.200000000/ (_dt * _dt);
  C3t_8 =  0.02539682539 / (_dt * _dt);
  C4t_8 = -0.00178571428 / (_dt * _dt);

  C0t_6 = -2.72222222222 / (_dt * _dt);
  C1t_6 =  1.500000000 / (_dt * _dt);
  C2t_6 = -0.150000000/ (_dt * _dt);
  C3t_6 =  0.01111111111 / (_dt * _dt);

  C0t_4 = -2.50000000000 / (_dt * _dt);
  C1t_4 =  1.33333333333 / (_dt * _dt);
  C2t_4 = -0.08333333333 / (_dt * _dt);

  C0t_2 = -2.00000000000 / (_dt * _dt);
  C1t_2 =  1.000000000 / (_dt * _dt);

  setDomainRange(model, data);

}

void SecondDeriv_V2::forward(const bool                         add,
                          const std::shared_ptr<SEP::float3DReg>model,
                          std::shared_ptr<SEP::float3DReg>      data) const {
  assert(checkDomainRange(model, data));
  if (!add) data->scale(0.);


    const std::shared_ptr<float3D> m =
      ((std::dynamic_pointer_cast<float3DReg>(model))->_mat);
     std::shared_ptr<float3D> d =
      ((std::dynamic_pointer_cast<float3DReg>(data))->_mat);

  //boundary condition
  #pragma omp parallel for collapse(2)
  for (int ix = 0; ix < n2; ix++) { //x
    for (int iz = 0; iz < n1; iz++) { //z
      (*d)[0][ix][iz] += //second time deriv
		//C0t_2*(  (*m)[1][ix ][iz ])  -
       		(C0t_2*  (*m)[1][ix ][iz ] + \
                C1t_2*(  (*m)[1-1][ix ][iz ]+(*m)[1+1][ix ][iz])) ;
      (*d)[1][ix][iz] += //second time deriv
       		//C0t_4*(  (*m)[2][ix ][iz ])   -
       		(C0t_4*  (*m)[2][ix ][iz ] + \
                C1t_4*(  (*m)[2-1][ix ][iz ]+(*m)[2 + 1][ix ][iz]) + \
                C2t_4*(  (*m)[2-2][ix ][iz ]+(*m)[2 + 2][ix ][iz])) ;
      (*d)[2][ix][iz] += //second time deriv
		//C0t_6*(  (*m)[3][ix ][iz ])   -
       		(C0t_6*  (*m)[3][ix ][iz ] + \
                C1t_6*(  (*m)[3-1][ix ][iz ]+(*m)[3 + 1][ix ][iz]) + \
                C2t_6*(  (*m)[3-2][ix ][iz ]+(*m)[3 + 2][ix ][iz]) + \
                C3t_6*(  (*m)[3-3][ix ][iz ]+(*m)[3 + 3][ix ][iz])) ;
      (*d)[3][ix][iz] += //second time deriv
		//C0t_8*(  (*m)[4][ix ][iz ])   -
       		(C0t_8*  (*m)[4][ix ][iz ] + \
                C1t_8*(  (*m)[4-1][ix ][iz ]+(*m)[4 + 1][ix ][iz]) + \
                C2t_8*(  (*m)[4-2][ix ][iz ]+(*m)[4 + 2][ix ][iz]) + \
                C3t_8*(  (*m)[4-3][ix ][iz ]+(*m)[4 + 3][ix ][iz])  + \
                C4t_8*(  (*m)[4-4][ix ][iz ]+(*m)[4 + 4][ix ][iz])) ;
      (*d)[n3-6][ix][iz] += //second time deriv
		//C0t_8*(  (*m)[n3-5][ix ][iz ])   - 
       		(C0t_8*  (*m)[n3-5][ix ][iz ]+ \
                C1t_8*(  (*m)[n3-5-1][ix ][iz ]+(*m)[n3-5 + 1][ix ][iz]) + \
                C2t_8*(  (*m)[n3-5-2][ix ][iz ]+(*m)[n3-5 + 2][ix ][iz]) + \
                C3t_8*(  (*m)[n3-5-3][ix ][iz ]+(*m)[n3-5 + 3][ix ][iz])  + \
                C4t_8*(  (*m)[n3-5-4][ix ][iz ]+(*m)[n3-5 + 4][ix ][iz])) ;
      (*d)[n3-5][ix][iz] += //second time deriv
		//C0t_6*(  (*m)[n3-4][ix][iz])   -
       		(C0t_6*  (*m)[n3-4][ix ][iz ] + \
                C1t_6*(  (*m)[n3-4-1][ix ][iz ]+(*m)[n3-4 + 1][ix ][iz]) + \
                C2t_6*(  (*m)[n3-4-2][ix ][iz ]+(*m)[n3-4 + 2][ix ][iz]) + \
                C3t_6*(  (*m)[n3-4-3][ix ][iz ]+(*m)[n3-4 + 3][ix ][iz])) ;
      (*d)[n3-4][ix][iz] += //second time deriv
		//C0t_4*(  (*m)[n3-3][ix ][iz ])   - 
       		(C0t_4*  (*m)[n3-3][ix ][iz ] + \
                C1t_4*(  (*m)[n3-3-1][ix ][iz ]+(*m)[n3-3 + 1][ix ][iz]) + \
                C2t_4*(  (*m)[n3-3-2][ix ][iz ]+(*m)[n3-3 + 2][ix ][iz])) ;
      (*d)[n3-3][ix][iz] += //second time deriv
		//C0t_2*(  (*m)[n3-2][ix ][iz ])   -
       		(C0t_2*  (*m)[n3-2][ix ][iz ] + \
                C1t_2*(  (*m)[n3-2-1][ix ][iz ]+(*m)[n3-2 + 1][ix ][iz])) ;
      (*d)[n3-2][ix][iz] += (*m)[0 ][ix ][iz]-(*m)[1 ][ix ][iz];
      (*d)[n3-1][ix][iz] += (*m)[1 ][ix ][iz];
    }
  }
  #pragma omp parallel for collapse(3)
  for (int it = 5; it < n3-5; it++) { //time
    for (int ix = 0; ix < n2; ix++) { //x
      for (int iz = 0; iz < n1; iz++) { //z
        (*d)[it-1][ix][iz] +=//second time deriv
       		(C0t_10*  (*m)[it][ix ][iz ] + \
                C1t_10*(  (*m)[it-1][ix ][iz ]+(*m)[it + 1][ix ][iz]) + \
                C2t_10*(  (*m)[it-2][ix ][iz ]+(*m)[it + 2][ix ][iz]) + \
                C3t_10*(  (*m)[it-3][ix ][iz ]+(*m)[it + 3][ix ][iz]) + \
                C4t_10*(  (*m)[it-4][ix ][iz ]+(*m)[it + 4][ix ][iz]) + \
                C5t_10*(  (*m)[it-5][ix ][iz ]+(*m)[it + 5][ix ][iz]));
      }
    }
  }

}

void SecondDeriv_V2::adjoint(const bool                         add,
                          std::shared_ptr<SEP::float3DReg>      model,
                          const std::shared_ptr<SEP::float3DReg>data) const{
  assert(checkDomainRange(model, data));

  if (!add) model->scale(0.);

    std::shared_ptr<float3D> m =
      ((std::dynamic_pointer_cast<float3DReg>(model))->_mat);
    const std::shared_ptr<float3D> d =
      ((std::dynamic_pointer_cast<float3DReg>(data))->_mat);


  #pragma omp parallel for collapse(2)
  for (int ix = 0; ix < n2; ix++) { //x
    for (int iz = 0; iz < n1; iz++) { //z
      (*m)[0][ix][iz] +=   //second time deriv
                         (C1t_2*( (*d)[0][ix ][iz]) + \
                         C2t_4*( (*d)[1][ix ][iz]) + \
                         C3t_6*( (*d)[2][ix ][iz]) + \
                         C4t_8*( (*d)[3][ix ][iz]) + \
                         C5t_10*( (*d)[4][ix ][iz]))  + \
                         //initial condition 
       			 (*d)[n3-2][ix ][iz ];
      (*m)[1][ix][iz] +=   //second time deriv
			//C0t_2*( (*d)[0][ix ][iz])   -
                         (C0t_2*( (*d)[0][ix ][iz]) + \
                         C1t_4*( (*d)[1][ix ][iz]) + \
                         C2t_6*( (*d)[2][ix ][iz]) + \
                         C3t_8*( (*d)[3][ix ][iz]) + \
                         C4t_10*( (*d)[4][ix ][iz]) + \
                         C5t_10*( (*d)[5][ix ][iz]))  +
                         //initial condition 
       			 (*d)[n3-1][ix ][iz ] - (*d)[n3-2][ix ][iz ];
      (*m)[2][ix][iz] +=   //second time deriv
			//C0t_4*( (*d)[1][ix ][iz])   -
                         (C0t_4*( (*d)[1][ix ][iz]) + \
                         C1t_2*(*d)[1-1][ix ][iz] + C1t_6*( (*d)[1+1][ix ][iz]) + \
                         C2t_8*( (*d)[1+2][ix ][iz]) + \
                         C3t_10*( (*d)[1+3][ix ][iz]) + \
                         C4t_10*( (*d)[1+4][ix ][iz]) + \
                         C5t_10*( (*d)[1+5][ix ][iz])) ;
      (*m)[3][ix][iz] +=   //second time deriv
			//C0t_6*( (*d)[2][ix ][iz])   -
                         (C0t_6*( (*d)[2][ix ][iz]) + \
                         C1t_4*(*d)[1][ix ][iz] + C1t_8*( (*d)[3][ix ][iz]) + \
                         C2t_10*( (*d)[4][ix ][iz]) + \
                         C3t_10*( (*d)[5][ix ][iz]) + \
                         C4t_10*( (*d)[6][ix ][iz]) + \
                         C5t_10*( (*d)[7][ix ][iz])) ;
      (*m)[4][ix][iz] +=   //second time deriv
                         //C0t_8*( (*d)[3][ix ][iz])   -
                         (C0t_8*( (*d)[3][ix ][iz]) + \
                         C1t_6*(*d)[2][ix ][iz] + C1t_10*( (*d)[4][ix ][iz]) + \
                         C2t_4*(*d)[1][ix ][iz] + C2t_10*( (*d)[5][ix ][iz]) + \
                         C3t_10*( (*d)[6][ix ][iz]) + \
                         C4t_10*( (*d)[7][ix ][iz]) + \
                         C5t_10*( (*d)[8][ix ][iz])) ;
      (*m)[5][ix][iz] +=   //second time deriv
			//C0t_10*( (*d)[4][ix ][iz])   -
                         (C0t_10*( (*d)[4][ix ][iz]) + \
                         C1t_8*(*d)[3][ix ][iz] + C1t_10*( (*d)[5][ix ][iz]) + \
                         C2t_6*(*d)[2][ix ][iz] + C2t_10*( (*d)[6][ix ][iz]) + \
                         C3t_10*( (*d)[7][ix ][iz]) + \
                         C4t_10*( (*d)[8][ix ][iz]) + \
                         C5t_10*( (*d)[9][ix ][iz])) ;
      (*m)[6][ix][iz] +=   //second time deriv
			//C0t_10*( (*d)[5][ix ][iz])   - 
                        (C0t_10*( (*d)[5][ix ][iz]) + \
                         C1t_10*(*d)[4][ix ][iz] + C1t_10*( (*d)[6][ix ][iz]) + \
                         C2t_8*(*d)[3][ix ][iz] + C2t_10*( (*d)[7][ix ][iz]) + \
                         C3t_6*(*d)[2][ix ][iz] + C3t_10*( (*d)[8][ix ][iz]) + \
                         C4t_10*( (*d)[9][ix ][iz]) + \
                         C5t_10*( (*d)[10][ix ][iz])) ;
      (*m)[7][ix][iz] +=   //second time deriv
			//C0t_10*( (*d)[6][ix ][iz])   -
                         (C0t_10*( (*d)[6][ix ][iz])+ \
                         C1t_10*(*d)[5][ix ][iz] + C1t_10*( (*d)[7][ix ][iz]) + \
                         C2t_10*(*d)[4][ix ][iz] + C2t_10*( (*d)[8][ix ][iz]) + \
                         C3t_8*(*d)[3][ix ][iz] + C3t_10*( (*d)[9][ix ][iz]) + \
                         C4t_10*( (*d)[10][ix ][iz]) + \
                         C5t_10*( (*d)[11][ix ][iz])) ; 
      (*m)[8][ix][iz] +=   //second time deriv
			//C0t_10*( (*d)[7][ix ][iz])  - 
                        (C0t_10*( (*d)[7][ix ][iz]) + \
                         C1t_10*(*d)[6][ix ][iz] + C1t_10*( (*d)[8][ix ][iz]) + \
                         C2t_10*(*d)[5][ix ][iz] + C2t_10*( (*d)[9][ix ][iz]) + \
                         C3t_10*(*d)[4][ix ][iz] + C3t_10*( (*d)[10][ix ][iz]) + \
                         C4t_8*(*d)[3][ix ][iz] + C4t_10*( (*d)[11][ix ][iz]) + \
                         C5t_10*( (*d)[12][ix ][iz])) ;
      (*m)[9][ix][iz] +=   //second time deriv
			//C0t_10*( (*d)[8][ix ][iz])  -
                        (C0t_10*( (*d)[8][ix ][iz]) + \
                         C1t_10*(*d)[7][ix ][iz] + C1t_10*( (*d)[9][ix ][iz]) + \
                         C2t_10*(*d)[6][ix ][iz] + C2t_10*( (*d)[10][ix ][iz]) + \
                         C3t_10*(*d)[5][ix ][iz] + C3t_10*( (*d)[11][ix ][iz]) + \
                         C4t_10*(*d)[4][ix ][iz] + C4t_10*( (*d)[12][ix ][iz]) + \
                         C5t_10*( (*d)[13][ix ][iz])) ;
      (*m)[n3-10][ix][iz] +=   //second time deriv
			//C0t_10*( (*d)[n3-11][ix ][iz])   -
                         (C0t_10*( (*d)[n3-11][ix ][iz]) + \
                         C1t_10*(*d)[n3-11-1][ix ][iz] + C1t_10*( (*d)[n3-11+1][ix ][iz]) + \
                         C2t_10*(*d)[n3-11-2][ix ][iz] + C2t_10*( (*d)[n3-11+2][ix ][iz]) + \
                         C3t_10*(*d)[n3-11-3][ix ][iz] + C3t_10*( (*d)[n3-11+3][ix ][iz]) + \
                         C4t_10*(*d)[n3-11-4][ix ][iz] + C4t_10*( (*d)[n3-11+4][ix ][iz]) + \
                         C5t_10*((*d)[n3-11-5][ix ][iz])) ;
      (*m)[n3-9][ix][iz] +=   //second time deriv
			//C0t_10*( (*d)[n3-10][ix ][iz])   -
                         (C0t_10*( (*d)[n3-10][ix ][iz]) + \
                         C1t_10*(*d)[n3-10-1][ix ][iz] + C1t_10*( (*d)[n3-10+1][ix ][iz]) + \
                         C2t_10*(*d)[n3-10-2][ix ][iz] + C2t_10*( (*d)[n3-10+2][ix ][iz]) + \
                         C3t_10*(*d)[n3-10-3][ix ][iz] + C3t_10*( (*d)[n3-10+3][ix ][iz]) + \
                         C4t_10*(*d)[n3-10-4][ix ][iz] + C4t_8*( (*d)[n3-10+4][ix ][iz]) + \
                         C5t_10*((*d)[n3-10-5][ix ][iz])) ;
      (*m)[n3-8][ix][iz] +=   //second time deriv
			//C0t_10*( (*d)[n3-9][ix ][iz])   -
                         (C0t_10*( (*d)[n3-9][ix ][iz]) + \
                         C1t_10*(*d)[n3-9-1][ix ][iz] + C1t_10*( (*d)[n3-9+1][ix ][iz]) + \
                         C2t_10*(*d)[n3-9-2][ix ][iz] + C2t_10*( (*d)[n3-9+2][ix ][iz]) + \
                         C3t_10*(*d)[n3-9-3][ix ][iz] + C3t_8*( (*d)[n3-9+3][ix ][iz]) + \
                         C4t_10*(*d)[n3-9-4][ix ][iz] + \
                         C5t_10*((*d)[n3-9-5][ix ][iz])) ;
      (*m)[n3-7][ix][iz] +=   //second time deriv
			//C0t_10*( (*d)[n3-8][ix ][iz])   -
                         (C0t_10*( (*d)[n3-8][ix ][iz]) + \
                         C1t_10*(*d)[n3-8-1][ix ][iz] + C1t_10*( (*d)[n3-8+1][ix ][iz]) + \
                         C2t_10*(*d)[n3-8-2][ix ][iz] + C2t_8*( (*d)[n3-8+2][ix ][iz]) + \
                         C3t_10*(*d)[n3-8-3][ix ][iz] + C3t_6*( (*d)[n3-8+3][ix ][iz]) + \
                         C4t_10*(*d)[n3-8-4][ix ][iz] + \
                         C5t_10*((*d)[n3-8-5][ix ][iz])) ;
      (*m)[n3-6][ix][iz] +=   //second time deriv
			//C0t_10*( (*d)[n3-7][ix ][iz])  - 
                         (C0t_10*( (*d)[n3-7][ix ][iz]) + \
                         C1t_10*(*d)[n3-7-1][ix ][iz] + C1t_8*( (*d)[n3-7+1][ix ][iz]) + \
                         C2t_10*(*d)[n3-7-2][ix ][iz] + C2t_6*( (*d)[n3-7+2][ix ][iz]) + \
                         C3t_10*(*d)[n3-7-3][ix ][iz] + \
                         C4t_10*(*d)[n3-7-4][ix ][iz] + \
                         C5t_10*((*d)[n3-7-5][ix ][iz])) ;
      (*m)[n3-5][ix][iz] +=   //second time deriv
			//C0t_8*( (*d)[n3-6][ix ][iz])   -
                         (C0t_8*( (*d)[n3-6][ix ][iz]) + \
                         C1t_10*(*d)[n3-6-1][ix ][iz] + C1t_6*( (*d)[n3-6+1][ix ][iz]) + \
                         C2t_10*(*d)[n3-6-2][ix ][iz] + C2t_4*( (*d)[n3-6+2][ix ][iz]) + \
                         C3t_10*(*d)[n3-6-3][ix ][iz] + \
                         C4t_10*(*d)[n3-6-4][ix ][iz] + \
                         C5t_10*((*d)[n3-6-5][ix ][iz])) ;
      (*m)[n3-4][ix][iz] +=   //second time deriv
			//C0t_6*( (*d)[n3-5][ix ][iz])   -
                         (C0t_6*( (*d)[n3-5][ix ][iz]) + \
                         C1t_8*(*d)[n3-5-1][ix ][iz] + C1t_4*( (*d)[n3-5+1][ix ][iz]) + \
                         C2t_10*(*d)[n3-5-2][ix ][iz] + \
                         C3t_10*(*d)[n3-5-3][ix ][iz] + \
                         C4t_10*(*d)[n3-5-4][ix ][iz] + \
                         C5t_10*((*d)[n3-5-5][ix ][iz])) ;
      (*m)[n3-3][ix][iz] +=   //second time deriv
			//C0t_4*( (*d)[n3-4][ix ][iz])   -
                         (C0t_4*( (*d)[n3-4][ix ][iz]) + \
                         C1t_6*(*d)[n3-4-1][ix ][iz] + C1t_2*( (*d)[n3-4+1][ix ][iz]) + \
                         C2t_8*(*d)[n3-4-2][ix ][iz] + \
                         C3t_10*(*d)[n3-4-3][ix ][iz] + \
                         C4t_10*(*d)[n3-4-4][ix ][iz] + \
                         C5t_10*((*d)[n3-4-5][ix ][iz])) ;
      (*m)[n3-2][ix][iz] +=   //second time deriv
			//C0t_2*( (*d)[n3-3][ix ][iz])   -
                         (C0t_2*( (*d)[n3-3][ix ][iz]) + \
                         C1t_4*(*d)[n3-3-1][ix ][iz] + \
                         C2t_6*(*d)[n3-3-2][ix ][iz] + \
                         C3t_8*(*d)[n3-3-3][ix ][iz] + \
                         C4t_10*(*d)[n3-3-4][ix ][iz] + \
                         C5t_10*((*d)[n3-3-5][ix ][iz]))  ;
      (*m)[n3-1][ix][iz] +=   //second time deriv
                         (C1t_2*(*d)[n3-2-1][ix ][iz] + \
                         C2t_4*(*d)[n3-2-2][ix ][iz] + \
                         C3t_6*(*d)[n3-2-3][ix ][iz] + \
                         C4t_8*(*d)[n3-2-4][ix ][iz] + \
                         C5t_10*((*d)[n3-2-5][ix ][iz])) ;
    }
  }

  #pragma omp parallel for collapse(3)
  for (int it = 9; it < n3-11; it++) {
    for (int ix = 0; ix < n2; ix++) {
      for (int iz = 0; iz < n1; iz++) {
        (*m)[it+1][ix][iz] += //second time deriv
                         //C0t_10*( (*d)[it][ix ][iz])*(*s)[ix][iz] - 
       			 (C0t_10*  (*d)[it][ix ][iz ]+ \
                         C1t_10*((*d)[it-1][ix ][iz ] +  (*d)[it + 1][ix ][iz]) + \
                         C2t_10*((*d)[it-2][ix ][iz ] +  (*d)[it + 2][ix ][iz]) + \
                         C3t_10*((*d)[it-3][ix ][iz ] +  (*d)[it + 3][ix ][iz]) + \
                         C4t_10*((*d)[it-4][ix ][iz ] +  (*d)[it + 4][ix ][iz]) + \
                         C5t_10*((*d)[it-5][ix ][iz ] +  (*d)[it + 5][ix ][iz]));
      }
    }
  }
}
