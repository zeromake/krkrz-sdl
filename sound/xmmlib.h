//---------------------------------------------------------------------------
/*
	Risa [りさ]      alias 吉里吉里3 [kirikiri-3]
	 stands for "Risa Is a Stagecraft Architecture"
	Copyright (C) 2000 W.Dee <dee@kikyou.info> and contributors

	See details of license at "license.txt"
*/
//---------------------------------------------------------------------------
//! @file
//! @brief XMMユーティリティ
//---------------------------------------------------------------------------

/*
	and

	the original copyright:
*/


/********************************************************************
 *                                                                  *
 * THIS FILE IS PART OF THE OggVorbis SOFTWARE CODEC SOURCE CODE.   *
 * USE, DISTRIBUTION AND REPRODUCTION OF THIS LIBRARY SOURCE IS     *
 * GOVERNED BY A BSD-STYLE SOURCE LICENSE INCLUDED WITH THIS SOURCE *
 * IN 'COPYING'. PLEASE READ THESE TERMS BEFORE DISTRIBUTING.       *
 *                                                                  *
 * THE OggVorbis SOURCE CODE IS (C) COPYRIGHT 1994-2003             *
 * by the XIPHOPHORUS Company http://www.xiph.org/                  *
 *                                                                  *
 ********************************************************************

 function: Header of SSE Function Library
 last mod: $Id: xmmlib.h,v 1.3 2005-07-08 15:00:00+09 blacksword Exp $

 ********************************************************************/

/*
	Based on a file from Ogg Vorbis Optimization Project:
	 http://homepage3.nifty.com/blacksword/
*/

#ifndef _XMMLIB_H_INCLUDED
#define _XMMLIB_H_INCLUDED

#include "tjsCommHead.h"
#include <memory.h>

//---------------------------------------------------------------------------


#if	defined(__GNUC__)||defined(_MSC_VER)
#else
#error "Not supported System."
#endif



#if __cplusplus >= 201703L
#define register
#endif

/* We need type definitions from the XMM header file.  */
#if defined(__vita__) || defined(__SWITCH__)
#include <simde/simde/simde-common.h>
#undef SIMDE_HAVE_FENV_H
#endif
#include <simde/x86/sse.h>
#include <simde/x86/sse2.h>

#ifdef __GNUC__
#define _SALIGN16(x) static x __attribute__((aligned(16)))
#define _ALIGN16(x) x __attribute__((aligned(16)))
#else
#define _SALIGN16(x) __declspec(align(16)) static x
#define _ALIGN16(x) __declspec(align(16)) x
#endif
#define PM64(x)		(*(simde__m64*)(x))
#define PM128(x)	(*(simde__m128*)(x))
#define PM128I(x)	(*(simde__m128i*)(x))

typedef union {
	unsigned char	si8[8];
	unsigned short	si16[4];
	unsigned long	si32[2];
	char			ssi8[8];
	short			ssi16[4];
	long			ssi32[2];
	simde__m64			pi64;
} __m64x;

#if		defined(_MSC_VER)
typedef union __declspec(intrin_type) __declspec(align(16)) __m128x{
	unsigned long	si32[4];
	float			sf[4];
	simde__m64			pi64[2];
	simde__m128			ps;
	simde__m128i			pi;
	simde__m128d			pd;
} __m128x;

#elif	defined(__GNUC__)
typedef union {
	unsigned long	si32[4];
	float			sf[4];
	simde__m64			pi64[2];
	simde__m128			ps;
	simde__m128i			pi;
	simde__m128d			pd;
} __m128x __attribute__((aligned(16)));

#endif

extern _ALIGN16(const tjs_uint32) PCS_NNRN[4];
extern _ALIGN16(const tjs_uint32) PCS_NNRR[4];
extern _ALIGN16(const tjs_uint32) PCS_NRNN[4];
extern _ALIGN16(const tjs_uint32) PCS_NRNR[4];
extern _ALIGN16(const tjs_uint32) PCS_NRRN[4];
extern _ALIGN16(const tjs_uint32) PCS_NRRR[4];
extern _ALIGN16(const tjs_uint32) PCS_RNNN[4];
extern _ALIGN16(const tjs_uint32) PCS_RNRN[4];
extern _ALIGN16(const tjs_uint32) PCS_RNRR[4];
extern _ALIGN16(const tjs_uint32) PCS_RRNN[4];
extern _ALIGN16(const tjs_uint32) PCS_RNNR[4];
extern _ALIGN16(const tjs_uint32) PCS_RRRR[4];
extern _ALIGN16(const tjs_uint32) PCS_NNNR[4];
extern _ALIGN16(const tjs_uint32) PABSMASK[4];
extern _ALIGN16(const tjs_uint32) PSTARTEDGEM1[4];
extern _ALIGN16(const tjs_uint32) PSTARTEDGEM2[4];
extern _ALIGN16(const tjs_uint32) PSTARTEDGEM3[4];
extern _ALIGN16(const tjs_uint32) PENDEDGEM1[4];
extern _ALIGN16(const tjs_uint32) PENDEDGEM2[4];
extern _ALIGN16(const tjs_uint32) PENDEDGEM3[4];
extern _ALIGN16(const tjs_uint32) PMASKTABLE[16*4];

extern _ALIGN16(const float) PFV_0[4];
extern _ALIGN16(const float) PFV_1[4];
extern _ALIGN16(const float) PFV_2[4];
extern _ALIGN16(const float) PFV_4[4];
extern _ALIGN16(const float) PFV_8[4];
extern _ALIGN16(const float) PFV_INIT[4];
extern _ALIGN16(const float) PFV_0P5[4];

inline simde__m128 simde_mm_untnorm_ps(simde__m128 x)
{
	_SALIGN16(const tjs_uint32)	PIV0[4]	 = {
		0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000
	};
	register simde__m128 r;
	r	 = simde_mm_and_ps(x, PM128(PCS_RRRR));
	r	 = simde_mm_or_ps(x, PM128(PIV0));
	return	r;
}

inline float simde_mm_add_horz(simde__m128 x)
{
	simde__m128	y;
	y	 = simde_mm_movehl_ps(y, x);
	x	 = simde_mm_add_ps(x, y);
	y	 = x;
	y	 = simde_mm_shuffle_ps(y, y, SIMDE_MM_SHUFFLE(1,1,1,1));
	x	 = simde_mm_add_ss(x, y);
	return simde_mm_cvtss_f32(x);
}

inline simde__m128 simde_mm_add_horz_ss(simde__m128 x)
{
	simde__m128	y;
	y	 = simde_mm_movehl_ps(y, x);
	x	 = simde_mm_add_ps(x, y);
	y	 = x;
	y	 = simde_mm_shuffle_ps(y, y, SIMDE_MM_SHUFFLE(1,1,1,1));
	x	 = simde_mm_add_ss(x, y);
	return x;
}

inline float simde_mm_max_horz(simde__m128 x)
{
	simde__m128	y;
	y	 = simde_mm_movehl_ps(y, x);
	x	 = simde_mm_max_ps(x, y);
	y	 = x;
	y	 = simde_mm_shuffle_ps(y, y, SIMDE_MM_SHUFFLE(1,1,1,1));
	x	 = simde_mm_max_ss(x, y);
	return simde_mm_cvtss_f32(x);
}

inline float simde_mm_min_horz(simde__m128 x)
{
	simde__m128	y;
	y	 = simde_mm_movehl_ps(y, x);
	x	 = simde_mm_min_ps(x, y);
	y	 = x;
	y	 = simde_mm_shuffle_ps(y, y, SIMDE_MM_SHUFFLE(1,1,1,1));
	x	 = simde_mm_min_ss(x, y);
	return simde_mm_cvtss_f32(x);
}



/**
 * 128ビット境界にポインタがアラインメントされているかどうか
 * @param p	ポインタ
 * @return	128ビット境界にポインタがアラインメントされているかどうか
 */
inline bool IsAlignedTo128bits(const void * p)
{
	return !(reinterpret_cast<tjs_offset>(p) & 0xf);
}


//---------------------------------------------------------------------------

#endif /* _XMMLIB_H_INCLUDED */
