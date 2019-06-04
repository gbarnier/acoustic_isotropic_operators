#include "innerLapl.h"

export void innerLapl(uniform int n, uniform int n12,uniform int n123,
 uniform float curW[], uniform float coef[])
 {

foreach(i=0...n){
  curW[i]=
            /*LAP*/
              coef[0]*curW[i]+
              coef[1]*(
                 curW[i-n12]+curW[i+n12]+
                 curW[i-n123]+ curW[i+n123])+
              coef[2]*(
                 curW[i-2*n12]+curW[i+2*n12]+
                 curW[i-2*n123]+ curW[i+2*n123])+
               coef[3]*(
                 curW[i-3*n12]+curW[i+3*n12]+
                 curW[i-3*n123]+ curW[i+3*n123])+
              coef[4]*(
                 curW[i-4*n12]+curW[i+4*n12]+
                 curW[i-4*n123]+ curW[i+4*n123])+
              coef[5]*(
                 curW[i-5*n12]+curW[i+5*n12]+
                 curW[i-5*n123]+ curW[i+5*n123]);
            }

}
