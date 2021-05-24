/*
 * Copyright (C) 2021 GreenWaves Technologies
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* Matrix transposition */

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wextra"
#pragma GCC diagnostic ignored "-Wpointer-sign"
#pragma GCC diagnostic ignored "-Wsign-compare"

#include "Gap.h"
#include "CNN_CopyBasicKernels.h"

static int CoreCountDynamic = 1;
static int ActiveCore = gap_ncore();

static inline unsigned int __attribute__((always_inline)) ChunkSize(unsigned int X)

{
	unsigned int NCore;
	unsigned int Log2Core;
	unsigned int Chunk;

	if (CoreCountDynamic) NCore = ActiveCore; else NCore = gap_ncore();
	Log2Core = gap_fl1(NCore);
	Chunk = (X>>Log2Core) + ((X&(NCore-1))!=0);
	return Chunk;
}

static void CNN_Transpose_Body_fps(
	signed char *__restrict__ In,
	signed char *__restrict__ Out,
	unsigned int W,
	unsigned int Wo_F,
	unsigned int Wo_L,
	unsigned int H,
	unsigned int Ho_F,
	unsigned int Ho_L
	)
       
{
	int IterL = Ho_L-Ho_F;
	int IterW = Wo_L-Wo_F;
	for (int l=0; l<IterL/4; l++) {
		v4s *pV0 = (v4s *) (In + (Ho_F+4*l+0)*W + Wo_F);
		v4s *pV1 = (v4s *) (In + (Ho_F+4*l+1)*W + Wo_F);
		v4s *pV2 = (v4s *) (In + (Ho_F+4*l+2)*W + Wo_F);
		v4s *pV3 = (v4s *) (In + (Ho_F+4*l+3)*W + Wo_F);
		signed char *pO  = (Out + Ho_F + 4*l + Wo_F*H);
		for (int c=0; c<IterW/4; c++) {
			v4s X, Y;
			v4s A = pV0[c], B = pV1[c], C = pV2[c], D = pV3[c];
			v4s rA, rB, rC, rD;

			X  = __builtin_shuffle(A, B, (v4s) {0,4,1,5});
			Y  = __builtin_shuffle(C, D, (v4s) {0,4,1,5});
			rA = __builtin_shuffle(X, Y, (v4s) {0,1,4,5});
			rB = __builtin_shuffle(X, Y, (v4s) {2,3,6,7});

			X  = __builtin_shuffle(A, B, (v4s) {2,6,3,7});
			Y  = __builtin_shuffle(C, D, (v4s) {2,6,3,7});
			rC = __builtin_shuffle(X, Y, (v4s) {0,1,4,5});
			rD = __builtin_shuffle(X, Y, (v4s) {2,3,6,7});

			*((v4s *) (pO+(4*c+0)*H)) = rA;
			*((v4s *) (pO+(4*c+1)*H)) = rB;
			*((v4s *) (pO+(4*c+2)*H)) = rC;
			*((v4s *) (pO+(4*c+3)*H)) = rD;
		}
		for (int c=4*(IterW/4); c<IterW; c++) {
			int X0 = In[(Ho_F+4*l+0)*W + (Wo_F+c)], X1 = In[(Ho_F+4*l+1)*W + (Wo_F+c)], X2 = In[(Ho_F+4*l+2)*W + (Wo_F+c)], X3 = In[(Ho_F+4*l+3)*W + (Wo_F+c)];
			Out[Ho_F+4*l+0 + (Wo_F+c)*H] = X0; Out[Ho_F+4*l+1 + (Wo_F+c)*H] = X1; Out[Ho_F+4*l+2 + (Wo_F+c)*H] = X2; Out[Ho_F+4*l+3 + (Wo_F+c)*H] = X3;
		}
	}
	for (int l=4*(IterL/4); l<IterL; l++) {
		for (int c=0; c<IterW/2; c++) {
			signed char X0 = In[(Ho_F+l)*W + Wo_F+2*c+0], X1 = In[(Ho_F+l)*W + Wo_F+2*c+1];
			Out[Ho_F+l + (Wo_F+2*c+0)*H] = X0;
			Out[Ho_F+l + (Wo_F+2*c+1)*H] = X1;
		}
		if (IterW&0x1) Out[Ho_F+l+ (Wo_L-1)*H] = In[(Ho_F+l)*W+Wo_L-1];
	}
}

static void CNN_Transpose_Body_fp(
	short int *__restrict__ In,
	short int *__restrict__ Out,
	unsigned int W,
	unsigned int Wo_F,
	unsigned int Wo_L,
	unsigned int H,
	unsigned int Ho_F,
	unsigned int Ho_L
	)
       
{
	int IterL = Ho_L-Ho_F;
	int IterW = Wo_L-Wo_F;
	for (int l=0; l<IterL/2; l++) {
		v2s *pV0 = (v2s *) (In + (Ho_F+2*l+0)*W + Wo_F);
		v2s *pV1 = (v2s *) (In + (Ho_F+2*l+1)*W + Wo_F);
		short int *pO  = (Out + Ho_F + 4*l + Wo_F*H);
		for (int c=0; c<IterW/2; c++) {
			v2s A = pV0[c], B = pV1[c];
			v2s rA, rB;
			rA = __builtin_shuffle(A, B, (v2s) {0,2});
			rB = __builtin_shuffle(A, B, (v2s) {1,3});
			*((v2s *) (pO+(2*c+0)*H)) = rA;
			*((v2s *) (pO+(2*c+1)*H)) = rB;
		}
		if (IterW&0x1) {
			int c = IterW-1;
			int X0 = In[(Ho_F+2*l+0)*W + (Wo_F+c)], X1 = In[(Ho_F+2*l+1)*W + (Wo_F+c)];
			Out[Ho_F+2*l+0 + (Wo_F+c)*H] = X0; Out[Ho_F+2*l+1 + (Wo_F+c)*H] = X1;
		}
	}
	if (IterL&0x1) {
		int l = IterL-1;
		for (int c=0; c<IterW/2; c++) {
			int X0 = In[(Ho_F+l)*W + Wo_F+2*c+0], X1 = In[(Ho_F+l)*W + Wo_F+2*c+1];
			Out[Ho_F+l + (Wo_F+2*c+0)*H] = X0;
			Out[Ho_F+l + (Wo_F+2*c+1)*H] = X1;
		}
		if (IterW&0x1) Out[Ho_F+l+ (Wo_L-1)*H] = In[(Ho_F+l)*W+Wo_L-1];
	}
}

static void CNN_TransposeSxSy_Body_fp(
	short int *__restrict__ In,
	short int *__restrict__ Out,
	unsigned int W,
	unsigned int Wo_F,
	unsigned int Wo_L,
	unsigned int Ho_F,
	unsigned int Ho_L,
	unsigned int Ho,
	unsigned int Sx,
	unsigned int Sy
	)
{
	for (int l=Ho_F; l<Ho_L; l++) {
		int IterW = Wo_L-Wo_F;
		for (int c=0; c<IterW/2; c++) {
			short int X0 = In[Sy*l*W + Sx*(Wo_F+2*c+0)], X1 = In[Sy*l*W + Sx*(Wo_F+2*c+1)];
			Out[l + (Wo_F+2*c+0)*Ho] = X0;
			Out[l + (Wo_F+2*c+1)*Ho] = X1;
		}
		if (IterW&0x1) Out[l+ (Wo_L-1)*Ho] = In[Sy*l*W+Sx*(Wo_L-1)];
	}
}

static void CNN_Transpose2_Body_fps(
	signed char *__restrict__ In,
	signed char *__restrict__ Out,
	unsigned int W,
	unsigned int Wo_F,
	unsigned int Wo_L,
	unsigned int H,
	unsigned int Ho_F,
	unsigned int Ho_L
	)
{
	for (int l=Ho_F; l<Ho_L; l++) {
		int IterW = Wo_L-Wo_F;
		for (int c=0; c<IterW/2; c++) {
			signed char X0 = In[l*W + Wo_F+2*c+0], X1 = In[l*W + Wo_F+2*c+1];
			Out[l + (Wo_F+2*c+0)*H] = X0;
			Out[l + (Wo_F+2*c+1)*H] = X1;
		}
		if (IterW&0x1) Out[l+ (Wo_L-1)*H] = In[l*W+Wo_L-1];
	}
}

static void CNN_TransposeSxSy_Body_fps(
	signed char *__restrict__ In,
	signed char *__restrict__ Out,
	unsigned int W,
	unsigned int Wo_F,
	unsigned int Wo_L,
	unsigned int Ho,
	unsigned int Ho_F,
	unsigned int Ho_L,
	unsigned int Sx,
	unsigned int Sy
	)
{
	for (int l=Ho_F; l<Ho_L; l++) {
		int IterW = Wo_L-Wo_F;
		for (int c=0; c<IterW/2; c++) {
			signed char X0 = In[Sy*l*W + Sx*(Wo_F+2*c+0)], X1 = In[Sy*l*W + Sx*(Wo_F+2*c+1)];
			Out[l + (Wo_F+2*c+0)*Ho] = X0;
			Out[l + (Wo_F+2*c+1)*Ho] = X1;
		}
		if (IterW&0x1) Out[l+ (Wo_L-1)*Ho] = In[Sy*l*W+Sx*(Wo_L-1)];
	}
}

void CNN_ParTranspose_fp(KerMatTranspose_fp_T *Arg)

{
	short int *__restrict__ In = Arg->In;
	short int *__restrict__ Out = Arg->Out;
	unsigned int W = Arg->W;
	unsigned int H = Arg->H;
	unsigned int Feat = Arg->Feat;
        unsigned int CoreId = gap_coreid();
        unsigned int Chunk = ChunkSize(Feat);
        unsigned int First = Chunk*CoreId;
        unsigned int Last = Min(First+Chunk, Feat);

	for (int f=First; f<Last; f++) CNN_Transpose_Body_fp(In+W*H*f, Out+W*H*f, W, 0, W, H, 0, H);
	gap_waitbarrier(0);
}

void CNN_ParTransposeSxSy_fp(KerMatTranspose_fp_T *Arg)

{
	short int *__restrict__ In = Arg->In;
	short int *__restrict__ Out = Arg->Out;
	unsigned int W = Arg->W;
	unsigned int H = Arg->H;
	unsigned int Feat = Arg->Feat;
	unsigned int Sx = Arg->Sx;
	unsigned int Sy = Arg->Sy;
	unsigned int Wo = (W+Sx-1)/Sx;
	unsigned int Ho = (H+Sy-1)/Sy;
        unsigned int CoreId = gap_coreid();
        unsigned int Chunk = ChunkSize(Feat);
        unsigned int First = Chunk*CoreId;
        unsigned int Last = Min(First+Chunk, Feat);

	for (int f=First; f<Last; f++) CNN_TransposeSxSy_Body_fp(In+W*H*f, Out+Wo*Ho*f, W, 0, Wo, Ho, 0, Ho, Sx, Sy);
	gap_waitbarrier(0);
}

void CNN_ParTranspose_fps(KerMatTranspose_fps_T *Arg)

{
	signed char *__restrict__ In = Arg->In;
	signed char *__restrict__ Out = Arg->Out;
	unsigned int W = Arg->W;
	unsigned int H = Arg->H;
	unsigned int Feat = Arg->Feat;
        unsigned int CoreId = gap_coreid();
        unsigned int Chunk = ChunkSize(Feat);
        unsigned int First = Chunk*CoreId;
        unsigned int Last = Min(First+Chunk, Feat);

	for (int f=First; f<Last; f++) CNN_Transpose_Body_fps(In+W*H*f, Out+W*H*f, W, 0, W, H, 0, H);
	gap_waitbarrier(0);
}

void CNN_ParTransposeSxSy_fps(KerMatTranspose_fps_T *Arg)

{
	signed char *__restrict__ In = Arg->In;
	signed char *__restrict__ Out = Arg->Out;
	unsigned int W = Arg->W;
	unsigned int H = Arg->H;
	unsigned int Feat = Arg->Feat;
	unsigned int Sx = Arg->Sx;
	unsigned int Sy = Arg->Sy;
	unsigned int Wo = (W+Sx-1)/Sx;
	unsigned int Ho = (H+Sy-1)/Sy;
        unsigned int CoreId = gap_coreid();
        unsigned int Chunk = ChunkSize(Feat);
        unsigned int First = Chunk*CoreId;
        unsigned int Last = Min(First+Chunk, Feat);

	for (int f=First; f<Last; f++) CNN_TransposeSxSy_Body_fps(In+W*H*f, Out+Wo*Ho*f, W, 0, Wo, Ho, 0, Ho, Sx, Sy);
	gap_waitbarrier(0);
}

void CNN_Transpose_fp(KerMatTranspose_fp_T *Arg)

{
	short int *__restrict__ In = Arg->In;
	short int *__restrict__ Out = Arg->Out;
	unsigned int W = Arg->W;
	unsigned int H = Arg->H;
	unsigned int Wo_F, Wo_L, Ho_F, Ho_L;
	unsigned int Feat = Arg->Feat;
        unsigned int Chunk, CoreId = gap_coreid();

	if (H>W) {
		/* Tile horizontally */
		Chunk = ChunkSize(H); Ho_F = Chunk*CoreId; Ho_L = Min(Ho_F+Chunk, H); Wo_F = 0; Wo_L = W;
	} else {
		/* Tile vertically */
		Chunk = ChunkSize(W); Wo_F = Chunk*CoreId; Wo_L = Min(Wo_F+Chunk, W); Ho_F = 0; Ho_L = H;
	}
	if (Wo_F<Wo_L && Ho_F<Ho_L) {
		for (int f=0; f<Feat; f++) CNN_Transpose_Body_fp(In+W*H*f, Out+W*H*f, W, Wo_F, Wo_L, H, Ho_F, Ho_L); 
	}
	gap_waitbarrier(0);
}

void CNN_TransposeSxSy_fp(KerMatTranspose_fp_T *Arg)

{
	short int *__restrict__ In = Arg->In;
	short int *__restrict__ Out = Arg->Out;
	unsigned int W = Arg->W;
	unsigned int H = Arg->H;
	unsigned int Wo_F, Wo_L, Ho_F, Ho_L;
	unsigned int Sx = Arg->Sx;
	unsigned int Sy = Arg->Sy;
	unsigned int Wo = (W+Sx-1)/Sx;
	unsigned int Ho = (H+Sy-1)/Sy;
	unsigned int Feat = Arg->Feat;
        unsigned int Chunk, CoreId = gap_coreid();

	if (Ho>Wo) {
		/* Tile horizontally */
		Chunk = ChunkSize(Ho); Ho_F = Chunk*CoreId; Ho_L = Min(Ho_F+Chunk, Ho); Wo_F = 0; Wo_L = Wo;
	} else {
		/* Tile vertically */
		Chunk = ChunkSize(Wo); Wo_F = Chunk*CoreId; Wo_L = Min(Wo_F+Chunk, Wo); Ho_F = 0; Ho_L = Ho;
	}
	if (Wo_F<Wo_L && Ho_F<Ho_L) {
		for (int f=0; f<Feat; f++) CNN_TransposeSxSy_Body_fp(In+W*H*f, Out+Wo*Ho*f, W, Wo_F, Wo_L, Ho, Ho_F, Ho_L, Sx, Sy); 
	}
	gap_waitbarrier(0);
}

void CNN_Transpose_fps(KerMatTranspose_fps_T *Arg)

{
	signed char *__restrict__ In = Arg->In;
	signed char *__restrict__ Out = Arg->Out;
	unsigned int W = Arg->W;
	unsigned int H = Arg->H;
	unsigned int Wo_F, Wo_L, Ho_F, Ho_L;
	unsigned int Feat = Arg->Feat;
        unsigned int Chunk, CoreId = gap_coreid();

	if (H>W) {
		/* Tile horizontally */
		Chunk = ChunkSize(H); Ho_F = Chunk*CoreId; Ho_L = Min(Ho_F+Chunk, H); Wo_F = 0; Wo_L = W;
	} else {
		/* Tile vertically */
		Chunk = ChunkSize(W); Wo_F = Chunk*CoreId; Wo_L = Min(Wo_F+Chunk, W); Ho_F = 0; Ho_L = H;
	}
	if (Wo_F<Wo_L && Ho_F<Ho_L) {
		for (int f=0; f<Feat; f++) CNN_Transpose_Body_fps(In+W*H*f, Out+W*H*f, W, Wo_F, Wo_L, H, Ho_F, Ho_L); 
	}
	gap_waitbarrier(0);
}


void CNN_TransposeSxSy_fps(KerMatTranspose_fps_T *Arg)

{
	signed char *__restrict__ In = Arg->In;
	signed char *__restrict__ Out = Arg->Out;
	unsigned int W = Arg->W;
	unsigned int H = Arg->H;
	unsigned int Wo_F, Wo_L, Ho_F, Ho_L;
	unsigned int Sx = Arg->Sx;
	unsigned int Sy = Arg->Sy;
	unsigned int Wo = (W+Sx-1)/Sx;
	unsigned int Ho = (H+Sy-1)/Sy;
	unsigned int Feat = Arg->Feat;
        unsigned int Chunk, CoreId = gap_coreid();

	if (Ho>Wo) {
		/* Tile horizontally */
		Chunk = ChunkSize(Ho); Ho_F = Chunk*CoreId; Ho_L = Min(Ho_F+Chunk, Ho); Wo_F = 0; Wo_L = Wo;
	} else {
		/* Tile vertically */
		Chunk = ChunkSize(Wo); Wo_F = Chunk*CoreId; Wo_L = Min(Wo_F+Chunk, Wo); Ho_F = 0; Ho_L = Ho;
	}
	if (Wo_F<Wo_L && Ho_F<Ho_L) {
		for (int f=0; f<Feat; f++) CNN_TransposeSxSy_Body_fps(In+W*H*f, Out+Wo*Ho*f, W, Wo_F, Wo_L, Ho, Ho_F, Ho_L, Sx, Sy); 
	}
	gap_waitbarrier(0);
}

/* 3D Tensor permutations */

void CNN_MatPermCHW2CWH_fp(KerMatTranspose_fp_T *Arg)

{
	short int *__restrict__ In = Arg->In;
	short int *__restrict__ Out = Arg->Out;
	unsigned int W = Arg->W;
	unsigned int H = Arg->H;
	unsigned int Feat = Arg->Feat, C = Feat;
        unsigned int CoreId = gap_coreid(), Chunk = ChunkSize(Feat), First = Chunk*CoreId, Last = Min(First+Chunk, Feat);

	for (int c=First; c<Last; c++) {
		for (int h=0; h<H; h++) {
			for (int w=0; w<((W/2)*2); w+=2) {
				int V0 = In[c*H*W+h*W+(w+0)], V1 = In[c*H*W+h*W+(w+1)];
				Out[c*H*W + (w+0)*H + h] = V0;
				Out[c*H*W + (w+1)*H + h] = V1;
			}
			if (W&0x1) Out[c*H*W + (W-1)*H + h] = In[c*H*W+h*W+W-1];
		}
	}
	gap_waitbarrier(0);
}

void CNN_MatPermCHW2HWC_fp(KerMatTranspose_fp_T *Arg)

{
	short int *__restrict__ In = Arg->In;
	short int *__restrict__ Out = Arg->Out;
	unsigned int W = Arg->W;
	unsigned int H = Arg->H;
	unsigned int Feat = Arg->Feat, C = Feat;
        unsigned int CoreId = gap_coreid(), Chunk = ChunkSize(Feat), First = Chunk*CoreId, Last = Min(First+Chunk, Feat);

	for (int c=First; c<Last; c++) {
		for (int h=0; h<H; h++) {
			for (int w=0; w<((W/2)*2); w+=2) {
				int V0 = In[c*H*W+h*W+(w+0)], V1 = In[c*H*W+h*W+(w+1)];
				Out[h*W*C + (w+0)*C + c] = V0;
				Out[h*W*C + (w+1)*C + c] = V1;
			}
			if (W&0x1) Out[h*W*C + (W-1)*C + c] = In[c*H*W+h*W+W-1];
		}
	}
	gap_waitbarrier(0);
}

void CNN_MatPermCHW2WHC_fp(KerMatTranspose_fp_T *Arg)

{
	short int *__restrict__ In = Arg->In;
	short int *__restrict__ Out = Arg->Out;
	unsigned int W = Arg->W;
	unsigned int H = Arg->H;
	unsigned int Feat = Arg->Feat, C = Feat;
        unsigned int CoreId = gap_coreid(), Chunk = ChunkSize(Feat), First = Chunk*CoreId, Last = Min(First+Chunk, Feat);

	for (int c=First; c<Last; c++) {
		for (int h=0; h<H; h++) {
			for (int w=0; w<((W/2)*2); w+=2) {
				int V0 = In[c*H*W+h*W+(w+0)], V1 = In[c*H*W+h*W+(w+1)];
				Out[(w+0)*H*C + h*C + c] = V0;
				Out[(w+1)*H*C + h*C + c] = V1;
			}
			if (W&0x1) Out[(W-1)*H*C + h*C + c] = In[c*H*W+h*W+W-1];
		}
	}
	gap_waitbarrier(0);
}

void CNN_MatPermCHW2WCH_fp(KerMatTranspose_fp_T *Arg)

{
	short int *__restrict__ In = Arg->In;
	short int *__restrict__ Out = Arg->Out;
	unsigned int W = Arg->W;
	unsigned int H = Arg->H;
	unsigned int Feat = Arg->Feat, C = Feat;
        unsigned int CoreId = gap_coreid(), Chunk = ChunkSize(Feat), First = Chunk*CoreId, Last = Min(First+Chunk, Feat);

	for (int c=First; c<Last; c++) {
		for (int h=0; h<H; h++) {
			for (int w=0; w<((W/2)*2); w+=2) {
				int V0 = In[c*H*W+h*W+(w+0)], V1 = In[c*H*W+h*W+(w+1)];
				Out[(w+0)*C*H + c*H + h] = V0;
				Out[(w+1)*C*H + c*H + h] = V1;
			}
			if (W&0x1) Out[(W-1)*C*H + c*H + h] = In[c*H*W+h*W+W-1];
		}
	}
	gap_waitbarrier(0);
}

void CNN_MatPermCHW2HCW_fp(KerMatTranspose_fp_T *Arg)

{
	short int *__restrict__ In = Arg->In;
	short int *__restrict__ Out = Arg->Out;
	unsigned int W = Arg->W;
	unsigned int H = Arg->H;
	unsigned int Feat = Arg->Feat, C = Feat;
        unsigned int CoreId = gap_coreid(), Chunk = ChunkSize(Feat), First = Chunk*CoreId, Last = Min(First+Chunk, Feat);

	for (int c=First; c<Last; c++) {
		for (int h=0; h<H; h++) {
			for (int w=0; w<((W/2)*2); w+=2) {
				int V0 = In[c*H*W+h*W+(w+0)], V1 = In[c*H*W+h*W+(w+1)];
				Out[h*C*W + c*W + (w+0)] = V0;
				Out[h*C*W + c*W + (w+1)] = V1;
			}
			if (W&0x1) Out[h*C*W + c*W + (W-1)] = In[c*H*W+h*W+W-1];
		}
	}
	gap_waitbarrier(0);
}


void CNN_MatPermCHW2CWH_fps(KerMatTranspose_fps_T *Arg)

{
	signed char *__restrict__ In = Arg->In;
	signed char *__restrict__ Out = Arg->Out;
	unsigned int W = Arg->W;
	unsigned int H = Arg->H;
	unsigned int Feat = Arg->Feat, C = Feat;
        unsigned int CoreId = gap_coreid(), Chunk = ChunkSize(Feat), First = Chunk*CoreId, Last = Min(First+Chunk, Feat);

	for (int c=First; c<Last; c++) {
		for (int h=0; h<H; h++) {
			for (int w=0; w<((W/2)*2); w+=2) {
				int V0 = In[c*H*W + h*W + (w+0)], V1 = In[c*H*W+h*W+(w+1)];
				Out[c*H*W + (w+0)*H + h] = V0;
				Out[c*H*W + (w+1)*H + h] = V1;
			}
			if (W&0x1) Out[c*H*W + (W-1)*H + h] = In[c*H*W+h*W+W-1];
		}
	}
	gap_waitbarrier(0);
}

void CNN_MatPermCHW2HWC_fps(KerMatTranspose_fps_T *Arg)

{
	signed char *__restrict__ In = Arg->In;
	signed char *__restrict__ Out = Arg->Out;
	unsigned int W = Arg->W;
	unsigned int H = Arg->H;
	unsigned int Feat = Arg->Feat, C = Feat;
        unsigned int CoreId = gap_coreid(), Chunk = ChunkSize(Feat), First = Chunk*CoreId, Last = Min(First+Chunk, Feat);

	for (int c=First; c<Last; c++) {
		for (int h=0; h<H; h++) {
			for (int w=0; w<((W/2)*2); w+=2) {
				int V0 = In[c*H*W+h*W+(w+0)], V1 = In[c*H*W+h*W+(w+1)];
				Out[h*W*C + (w+0)*C + c] = V0;
				Out[h*W*C + (w+1)*C + c] = V1;
			}
			if (W&0x1) Out[h*W*C + (W-1)*C + c] = In[c*H*W+h*W+W-1];
		}
	}
	gap_waitbarrier(0);
}

void CNN_MatPermCHW2WHC_fps(KerMatTranspose_fps_T *Arg)

{
	signed char *__restrict__ In = Arg->In;
	signed char *__restrict__ Out = Arg->Out;
	unsigned int W = Arg->W;
	unsigned int H = Arg->H;
	unsigned int Feat = Arg->Feat, C = Feat;
        unsigned int CoreId = gap_coreid(), Chunk = ChunkSize(Feat), First = Chunk*CoreId, Last = Min(First+Chunk, Feat);

	for (int c=First; c<Last; c++) {
		for (int h=0; h<H; h++) {
			for (int w=0; w<((W/2)*2); w+=2) {
				int V0 = In[c*H*W+h*W+(w+0)], V1 = In[c*H*W+h*W+(w+1)];
				Out[(w+0)*H*C + h*C + c] = V0;
				Out[(w+1)*H*C + h*C + c] = V1;
			}
			if (W&0x1) Out[(W-1)*H*C + h*C + c] = In[c*H*W+h*W+W-1];
		}
	}
	gap_waitbarrier(0);
}

void CNN_MatPermCHW2WCH_fps(KerMatTranspose_fps_T *Arg)

{
	signed char *__restrict__ In = Arg->In;
	signed char *__restrict__ Out = Arg->Out;
	unsigned int W = Arg->W;
	unsigned int H = Arg->H;
	unsigned int Feat = Arg->Feat, C = Feat;
        unsigned int CoreId = gap_coreid(), Chunk = ChunkSize(Feat), First = Chunk*CoreId, Last = Min(First+Chunk, Feat);

	for (int c=First; c<Last; c++) {
		for (int h=0; h<H; h++) {
			for (int w=0; w<((W/2)*2); w+=2) {
				int V0 = In[c*H*W+h*W+(w+0)], V1 = In[c*H*W+h*W+(w+1)];
				Out[(w+0)*C*H + c*H + h] = V0;
				Out[(w+1)*C*H + c*H + h] = V1;
			}
			if (W&0x1) Out[(W-1)*C*H + c*H + h] = In[c*H*W+h*W+W-1];
		}
	}
	gap_waitbarrier(0);
}

void CNN_MatPermCHW2HCW_fps(KerMatTranspose_fps_T *Arg)

{
	signed char *__restrict__ In = Arg->In;
	signed char *__restrict__ Out = Arg->Out;
	unsigned int W = Arg->W;
	unsigned int H = Arg->H;
	unsigned int Feat = Arg->Feat, C = Feat;
        unsigned int CoreId = gap_coreid(), Chunk = ChunkSize(Feat), First = Chunk*CoreId, Last = Min(First+Chunk, Feat);

	for (int c=First; c<Last; c++) {
		for (int h=0; h<H; h++) {
			for (int w=0; w<((W/2)*2); w+=2) {
				int V0 = In[c*H*W+h*W+(w+0)], V1 = In[c*H*W+h*W+(w+1)];
				Out[h*C*W + c*W + (w+0)] = V0;
				Out[h*C*W + c*W + (w+1)] = V1;
			}
			if (W&0x1) Out[h*C*W + c*W + (W-1)] = In[c*H*W+h*W+W-1];
		}
	}
	gap_waitbarrier(0);
}


#define B_CLR(x, bits)	((x)&(~((1<<(bits))-1)))
void CNN_Copy_void(KerCopy_void_T * Arg)
{
	unsigned int Size = Arg->W * Arg->H;
	unsigned int CoreId = gap_coreid();
	unsigned int Chunk = ChunkSize(Size), First = Chunk*CoreId, Last = Min(First+Chunk, Size);
	unsigned int Iter = Max(0, Last-First);
	signed char *__restrict__ From = Arg->In;
	signed char *__restrict__ To = Arg->Out;

	int *pFrom = (int *) (From+First), *pTo = (int *) (To+First);
	for (int i=0; i<Iter/8; i++) {
		int V0 = pFrom[2*i], V1 = pFrom[2*i+1];
		pTo[2*i] = V0; pTo[2*i+1] = V1;
	}
	if (Iter & 0x4) *((int *) (To + First + B_CLR(Iter, 3))) = *((int *) (From + First + B_CLR(Iter, 3)));
	if (Iter & 0x2) *((short int *) (To + First + B_CLR(Iter, 2))) = *((short int *) (From + First + B_CLR(Iter, 2)));
	if (Iter & 0x1) *((signed char *) (To + First + Iter - 1)) = *((signed char *) (From + First + Iter - 1));
}

#pragma GCC diagnostic pop
