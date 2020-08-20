/********************************************//**
   Author: Stuart Farris
   Date: 06NOV2017
   Description:  2D Laplcian to the tenth order. If 3D cube is passed in, 2D
      lapl is taken for each slice of fast axis.
 ***********************************************/

#pragma once
#include <operator.h>
#include <float2DReg.h>
#include <float3DReg.h>
#include <complex4DReg.h>

using namespace SEP;

class Laplacian2d : public Operator<SEP::float3DReg, SEP::float3DReg> {
  public:


    /**
       2d lapl of each slice of fast axis
     */
    Laplacian2d(const std::shared_ptr<SEP::float3DReg>model,
                const std::shared_ptr<SEP::float3DReg>data);

    /**
       lapl(model) -> data
     */
    void forward(const bool                         add,
                         const std::shared_ptr<SEP::float3DReg>model,
                         std::shared_ptr<SEP::float3DReg>      data) const ;

    /**
       lapl(data) -> model
     */
    void adjoint(const bool                         add,
                         std::shared_ptr<SEP::float3DReg>      model,
                         const std::shared_ptr<SEP::float3DReg>data) const ;

  private:

  int n1,n2,n3;
    float _da, _db;                                                   // spatial
                                                                      // sampling
                                                                      // of two
                                                                      // axis
    bool _3d;                                                         // 3d flag
    float C0z, C1z, C2z, C3z, C4z, C5z, C0x, C1x, C2x, C3x, C4x, C5x; // lapl
                                                                      // coeff
    int _laplOrder= 5;
    std::shared_ptr<SEP::float3DReg>buffer;
  };


  class Laplacian2d_multi_exp : public Operator<SEP::float4DReg, SEP::float4DReg> {
    public:


      /**
         2d lapl of each slice of fast axis
       */
      Laplacian2d_multi_exp(const std::shared_ptr<SEP::float4DReg>model,
                  const std::shared_ptr<SEP::float4DReg>data);

      /**
         lapl(model) -> data
       */
      void forward(const bool                         add,
                           const std::shared_ptr<SEP::float4DReg>model,
                           std::shared_ptr<SEP::float4DReg>      data) const ;

      /**
         lapl(data) -> model
       */
      void adjoint(const bool                         add,
                           std::shared_ptr<SEP::float4DReg>      model,
                           const std::shared_ptr<SEP::float4DReg>data) const ;

    private:

    int n1,n2,n3,n4;
      float _da, _db;                                                   // spatial
                                                                        // sampling
                                                                        // of two
                                                                        // axis
      bool _3d;                                                         // 3d flag
      float C0z, C1z, C2z, C3z, C4z, C5z, C0x, C1x, C2x, C3x, C4x, C5x; // lapl
                                                                        // coeff
      int _laplOrder= 5;
      std::shared_ptr<SEP::float4DReg>buffer;
    };

  class Laplacian2d_multi_exp_complex : public Operator<SEP::complex4DReg, SEP::complex4DReg> {
    public:


      /**
         2d lapl of each slice of fast axis
       */
      Laplacian2d_multi_exp_complex(const std::shared_ptr<SEP::complex4DReg>model,
                  const std::shared_ptr<SEP::complex4DReg>data);

      /**
         lapl(model) -> data
       */
      void forward(const bool                         add,
                           const std::shared_ptr<SEP::complex4DReg>model,
                           std::shared_ptr<SEP::complex4DReg>      data) const ;

      /**
         lapl(data) -> model
       */
      void adjoint(const bool                         add,
                           std::shared_ptr<SEP::complex4DReg>      model,
                           const std::shared_ptr<SEP::complex4DReg>data) const ;

      bool dotTest(const bool verbose = false, const float maxError = .00001) const{
        std::cerr << "cpp dot test not implemented.\n";
      }
    private:

    int n1,n2,n3,n4;
      float _da, _db;                                                   // spatial
                                                                        // sampling
                                                                        // of two
                                                                        // axis
      bool _3d;                                                         // 3d flag
      float C0z, C1z, C2z, C3z, C4z, C5z, C0x, C1x, C2x, C3x, C4x, C5x; // lapl
                                                                        // coeff
      int _laplOrder= 5;
      std::shared_ptr<SEP::complex4DReg>buffer;
    };
