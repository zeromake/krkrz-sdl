
#ifndef __SIMD_DEF_X86_X64_H__
#define __SIMD_DEF_X86_X64_H__


#if defined(__vita__) || defined(__SWITCH__)
#include <simde/simde/simde-common.h>
#undef SIMDE_HAVE_FENV_H
#endif
#include <simde/x86/sse.h>
#include <simde/x86/sse2.h>
#include <simde/x86/sse3.h>
#include <simde/x86/ssse3.h>
#include <simde/x86/avx.h>
#include <simde/x86/avx2.h>

#ifdef _MSC_VER
#ifndef simde_mm_srli_pi64
#define simde_mm_srli_pi64 simde_mm_srli_si64
#endif
#ifndef simde_mm_slli_pi64
#define simde_mm_slli_pi64 simde_mm_slli_si64
#endif
#pragma warning(push)
#pragma warning(disable : 4799)	// ignore simde_mm_empty request.
#ifndef simde_mm_cvtsi64_m64
__inline simde__m64 simde_mm_cvtsi64_m64( __int64 v ) { simde__m64 ret; ret.m64_i64 = v; return ret; }
#endif
#ifndef simde_mm_cvtm64_si64
__inline __int64 simde_mm_cvtm64_si64( simde__m64 v ) { return v.m64_i64; }
#endif
#pragma warning(pop)
#endif

#ifdef _MSC_VER // visual c++
# define ALIGN16_BEG __declspec(align(16))
# define ALIGN16_END 
# define ALIGN32_BEG __declspec(align(32))
# define ALIGN32_END 
#else // gcc or icc
# define ALIGN16_BEG
# define ALIGN16_END __attribute__((aligned(16)))
# define ALIGN32_BEG
# define ALIGN32_END __attribute__((aligned(32)))
#endif


#define _PS_CONST128(Name, Val)    \
const ALIGN16_BEG float WeightValuesSSE::##Name[4] ALIGN16_END = { Val, Val, Val, Val }

#define _PI32_CONST128(Name, Val)  \
const ALIGN16_BEG tjs_uint32 WeightValuesSSE::##Name[4] ALIGN16_END = { Val, Val, Val, Val }


#define _PS_CONST256(Name, Val)    \
const ALIGN32_BEG float WeightValuesAVX::##Name[8] ALIGN32_END = { Val, Val, Val, Val, Val, Val, Val, Val }

#define _PI32_CONST256(Name, Val)  \
const ALIGN32_BEG tjs_uint32 WeightValuesAVX::##Name[8] ALIGN32_END = { Val, Val, Val, Val, Val, Val, Val, Val }

#define _PS_CONST_TYPE256(Name, Type, Val)                                 \
static const ALIGN32_BEG Type m256_ps_##Name[8] ALIGN32_END = { Val, Val, Val, Val, Val, Val, Val, Val }

#endif // __SIMD_DEF_X86_X64_H__
