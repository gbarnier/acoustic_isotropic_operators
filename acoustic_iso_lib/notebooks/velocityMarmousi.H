Input from a tty

Segy2sep:   ettore88@cees-mazama   Sun Nov  5 10:15:22 2017
		sets next: in="/scratch/ettore/Segy2sep@7CcLst"
		junkME=0
		hff="Dat/MODEL_P-WAVE_VELOCITY_1.25m.H@@"
		same_record_number=1
	 	n2=1 o2=1 d2=1 label2="trace number" 

n3=1 n4=1 n5=1 n6=1 n7=1 n8=1 n9=1 

	n1=2801  o1=0.000000  d1=0.001250   label1="none"   unit1="none"
		gff="-1"
		esize=4
		data_format="xdr_float"
		junkME=0
		same_record_number=1
	 	n2=13601 o2=1 d2=1 label2="trace number" 

n3=1 n4=1 n5=1 n6=1 n7=1 n8=1 n9=1 

	n1=2801  o1=0.000000  d1=0.001250   label1="none"   unit1="none"
		gff="-1"
		esize=4


Window:   ettore88@cees-mazama   Sun Nov  5 10:15:24 2017
		sets next: in="/scratch/ettore/Window@BZ53XL"
		In:	float	Fortran (2801,13601,1,1)	C [1][1][13601][2801]
		Out:	float	Fortran (701,3401,1,1)	C [1][1][3401][701]
	 From History:
		 n4=1 n3=1 n2=13601 n1=2801 esize=4
	 From Par:
		 n1=701  #f1=0  #j1=4  #min1=0.000000  #max1=3.5
		 n2=3401  #f2=0  #j2=4  #min2=1.000000  #max2=13601
		 n3=1  #f3=0  #j3=1  #min3=0.000000  #max3=0
		 n4=1  #f4=0  #j4=1  #min4=0  #max4=0
	 Make header useful for further processing:
		 n1=701 	 n2=3401 	 n3=1 	 n4=1
		 o1=0.000000 	 o2=1.000000 	 o3=0.000000 	 o4=0.000000
		 d1=0.005 	 d2=4 	 d3=1 	 d4=1
		 label1="none" 	 label2="trace number" 	 label3="" 	 label4=""
		data_format="xdr_float"

hff=-1 o2=0.0 d1=5.000 d2=5.000 label1='z [m]' label2='x [m]' label3='P-WAVE_VELOCITY'


Scp3d:   ettore88@cees-mazama   Sun Nov  5 10:15:24 2017
		sets next: in="/net/server2/homes/sep/gbarnier/code/gpu/Vp_true_elastic.H@"
		copy_intag="Dat/P-WAVE_VELOCITY-true.H"
		data_format="xdr_float"


Clip:   gbarnier@jarvis   Fri Nov  2 11:36:44 2018
		sets next: in="/net/fantastic/scr1/gbarnier/temp0.H@"
		clip=2000
		chop="l"
		to="clip"
		data_format="xdr_float"


Clip:   gbarnier@jarvis   Fri Nov  2 11:36:44 2018
		sets next: in="/net/fantastic/scr1/gbarnier/temp1.H@"
		clip=4000
		chop="g"
		to="clip"
		data_format="xdr_float"


/opt/SEP/SEP7.0/bin/Math:   gbarnier@jarvis   Fri Nov  2 11:36:45 2018
		sets next: in="/net/fantastic/scr1/gbarnier/temp2.H@"
		esize=4
	#Using file1/1000 to produce real data

		#file1="temp1.H"
		data_format="xdr_float"
d1=0.005 d2=0.005 d3=0.005


Interp:   gbarnier@jarvis   Fri Nov  2 11:36:45 2018
		sepstr_ready_out() : sets next in="stdin"
	sinc interpolation with 10 points
	n1=351 o1=0.000000 d1=0.010000
	n2=1701 o2=0.000000 d2=0.010000
	n3=1 o3=0.000000 d3=0.010000
		data_format="xdr_float"


Window3d:   gbarnier@jarvis   Fri Nov  2 11:36:45 2018
		sets next: in="/net/fantastic/scr1/gbarnier/velocityMarmousi.H@"
#Window #f1=0  #j1=1 #n1=350
#Window #f2=0  #j2=1 #n2=1700
#Window #f3=0  #j3=1 #n3=1
		junkME=0
	n2=1700  o2=0.000000  d2=0.010000   label2="x [m]"   unit2="Undefined"
	n3=1  o3=0.000000  d3=0.010000   label3="P-WAVE_VELOCITY"   unit3="Undefined"
		gff="-1"
		hff="-1"
hff=-1 gff=-1  n4=1  n5=1  n6=1  n7=1  n8=1  n9=1 
	n1=350  o1=0.000000  d1=0.010000   label1="z [m]"   unit1="none"
		hff="-1"
		gff="-1"
		esize=4
		data_format="xdr_float"



Cp:   ettore@jarvis   Fri Jan  4 08:54:42 2019
		sets next: in="/net/fantastic/scr1/ettore/velocityMarmousi.H@"
		copy_intag="/homes/sep/gbarnier/code/gpu/gpuDouble/velocityMarmousi.H"
		data_format="xdr_float"
in='/homes/sep/ettore/research/packages/acoustic_isotropic_operators/acoustic_iso_lib/notebooks/velocityMarmousi.H@'
in='./velocityMarmousi.H@'
