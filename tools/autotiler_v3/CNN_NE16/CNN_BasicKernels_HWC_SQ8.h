#ifndef __CNN_BASICKERNELS_HWC_SQ8__
#define __CNN_BASICKERNELS_HWC_SQ8__
#include "Gap.h"
#include "../CNN_Libraries/CNN_Defines.h"
#ifdef GENASM
#ifdef __EMUL__
#define gap_ncore()     8
#define gap_coreid()    __builtin_pulp_CoreId()
#endif
#endif

#define AT_INF_BIASL_SM         0
#define AT_INF_ACTSCALE		0
#define AT_INF_ACTSCALEN	1
#define AT_INF_A0		2
#define AT_INF_B0		3
#define AT_INF_C0		4

#define AT_INF_BIASN		5
#define AT_INF_IN1SCALE		5
#define AT_INF_SCALE		5

#define AT_INF_SCALEN		6
#define AT_INF_IN1SCALEN	6

#define AT_INF_OUTSCALE		7
#define AT_INF_OUTSCALEN	8

#define AT_INF_DIM		9


/******************************************************************************************************************
	Standalone scaling and activation
******************************************************************************************************************/
typedef struct {
	void *__restrict__ In;
	void *__restrict__ Out;
	unsigned short int Feat;
	unsigned short int W;
	unsigned short int H;
	signed char * __restrict__ Infos;
} KerActivation_HWC_SQ8_T;

/******************************************************************************************************************
	Reduction scaling and activation after double precision convolution or linear layer
******************************************************************************************************************/
typedef struct {
	int *__restrict__ In;
	void *__restrict__ Out;
	unsigned short int Feat;
	unsigned short int W;
	unsigned short int H;
	unsigned char * __restrict__ Scale;
	unsigned char * __restrict__ ScaleN;
	signed char * __restrict__ Infos;
} KerConvLinReduct_HWC_SQ8_T;

/******************************************************************************************************************
	Pooling followed by optional scaling and activation
******************************************************************************************************************/
typedef struct {
	unsigned char * __restrict__ In;
	unsigned char * __restrict__ Out;
	unsigned short int Feat;
	unsigned short int W;
	unsigned short int UsedW;
	unsigned short int H;
	unsigned short int UsedH;
	unsigned short FS;		/* Filter Size, x */
	unsigned short FSy;		/* Filter Size, y */
	unsigned char S;		/* Filter Stride, x */
	unsigned char Sy;		/* Filter Stride, y */
	v4s Pad;
	signed char * __restrict__ Infos;
} KerPool_HWC_USQ8_T;

/******************************************************************************************************************
          Stand alone activation. Parallel Feature, Feature Parallel
	  Input is a scaled 8b tensor
	  Output is a scaled 8b tensor, Scale can be different from the one of input
******************************************************************************************************************/

/*
 * Standalone Scaled Activation, Features are evaluated one after the other in parallel
*/
void Ker_ActNone_HWC_SQ8(KerActivation_HWC_SQ8_T *Arg);
void Ker_ReLU_HWC_SQ8(KerActivation_HWC_SQ8_T *Arg);
void Ker_ReLUN_HWC_SQ8(KerActivation_HWC_SQ8_T *Arg);
void Ker_HSigmoid_HWC_SQ8(KerActivation_HWC_SQ8_T *Arg);
void Ker_HSwish_HWC_SQ8(KerActivation_HWC_SQ8_T *Arg);
void Ker_LeakyReLU_HWC_SQ8(KerActivation_HWC_SQ8_T *Arg);

/******************************************************************************************************************
          Input Scaling followed by an optional activation. Parallel Feature, Feature Parallel
	  Input is assumed to be the 32b unnormalized output of a convolution or a linear layer
	  Optional activation is applied to the scaled input and can be optionaly scaled also
	  Output is a scaled 8b quantized tensor
	  Channel Centric (CC)
******************************************************************************************************************/

/*
 * Input Scaling and reduction to 8b then channel centric activation, Out location != In location. Features are evaluated in parallel
*/
void KerParReduct_CC_HWC_SQ8(KerConvLinReduct_HWC_SQ8_T *Arg);
void KerParReduct_CC_NoScale_HWC_SQ8(KerConvLinReduct_HWC_SQ8_T *Arg);
void KerParReduct_CC_ReLU_HWC_SQ8(KerConvLinReduct_HWC_SQ8_T *Arg);
void KerParReduct_CC_ReLUN_HWC_SQ8(KerConvLinReduct_HWC_SQ8_T *Arg);
void KerParReduct_CC_HSigmoid_HWC_SQ8(KerConvLinReduct_HWC_SQ8_T *Arg);
void KerParReduct_CC_HSwish_HWC_SQ8(KerConvLinReduct_HWC_SQ8_T *Arg);
void KerParReduct_CC_LeakyReLU_HWC_SQ8(KerConvLinReduct_HWC_SQ8_T *Arg);

void KerParReduct_CC_NoScale_HWC_USQ8(KerConvLinReduct_HWC_SQ8_T *Arg);
void KerParReduct_CC_HWC_USQ8(KerConvLinReduct_HWC_SQ8_T *Arg);
void KerParReduct_CC_ReLU_HWC_USQ8(KerConvLinReduct_HWC_SQ8_T *Arg);
void KerParReduct_CC_ReLUN_HWC_USQ8(KerConvLinReduct_HWC_SQ8_T *Arg);

void KerParPool_MaxPoolNxMStrideSxSy__HWC_USQ8(KerPool_HWC_USQ8_T *Arg);

#endif
