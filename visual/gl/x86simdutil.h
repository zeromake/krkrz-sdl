

#ifndef __X86_SIMD_UTIL_H__
#define __X86_SIMD_UTIL_H__

#if defined(__vita__) || defined(__SWITCH__)
#include <simde/simde/simde-common.h>
#undef SIMDE_HAVE_FENV_H
#endif
#include <simde/x86/sse.h>
#include <simde/x86/sse2.h>
#include <simde/x86/sse3.h>
#include <simde/x86/avx.h>
#include <simde/x86/avx2.h>
#include <simde/x86/fma.h>

// SIMD版数学系関数 ( SSE+SSE2使用 ) 4要素一気に計算する
extern simde__m128 log_ps(simde__m128 x);
extern simde__m128 exp_ps(simde__m128 x);
extern simde__m128 sin_ps(simde__m128 x);
extern simde__m128 cos_ps(simde__m128 x);
extern void sincos_ps(simde__m128 x, simde__m128 *s, simde__m128 *c);

// SIMD版数学系関数 ( AVX+AVX2使用 ) 8要素一気に計算する
extern simde__m256 mm256_sin_ps(simde__m256 x);
extern simde__m256 mm256_exp_ps(simde__m256 x);
extern simde__m256 mm256_cos_ps(simde__m256 x);

// exp(y*log(x)) for pow(x, y)
inline simde__m128 pow_ps( simde__m128 x, simde__m128 y ) {
	return exp_ps( simde_mm_mul_ps( y, log_ps( x ) ) );
}

/**
 * 22bit 精度で逆数を求める(4要素版)
 * aの逆数 = 2 * a - rcpa * a * a を用いる
 */
inline simde__m128 m128_rcp_22bit_ps( const simde__m128& a ) {
	simde__m128 xm0 = a;
	simde__m128 xm1 = simde_mm_rcp_ps(xm0);
	xm0 = simde_mm_mul_ps( simde_mm_mul_ps( xm0, xm1 ), xm1 );
	xm1 = simde_mm_add_ps( xm1, xm1 );
	return simde_mm_sub_ps( xm1, xm0 );
}
/**
 * 22bit 精度で逆数を求める(1要素版)
 */
inline simde__m128 m128_rcp_22bit_ss( const simde__m128& a ) {
	simde__m128 xm0 = a;
	simde__m128 xm1 = simde_mm_rcp_ss(xm0);
	xm0 = simde_mm_mul_ss( simde_mm_mul_ss( xm0, xm1 ), xm1 );
	xm1 = simde_mm_add_ss( xm1, xm1 );
	return simde_mm_sub_ss( xm1, xm0 );
}
/**
 * 22bit 精度で逆数を求める(float版)
 */
inline float rcp_sse( float a ) {
	float  ret;
	simde__m128 xm0 = simde_mm_set_ss(a);
	simde_mm_store_ss( &ret, m128_rcp_22bit_ss(xm0) );
	return ret;
}

/**
 * 22bit 精度で逆数を求める(8要素版)
 * aの逆数 = 2 * a - rcpa * a * a を用いる
 */
inline simde__m256 m256_rcp_22bit_ps( const simde__m256& a ) {
	simde__m256 xm0 = a;
	simde__m256 xm1 = simde_mm256_rcp_ps(xm0);
	xm0 = simde_mm256_mul_ps( simde_mm256_mul_ps( xm0, xm1 ), xm1 );
	xm1 = simde_mm256_add_ps( xm1, xm1 );
	return simde_mm256_sub_ps( xm1, xm0 );
}

/**
 * 22bit 精度で逆数を求める(8要素版)
 * aの逆数 = 2 * a - rcpa * a * a を用いる
 * FMA3 を使って、少し速くする 未確認
 */
inline simde__m256 m256_rcp_22bit_fma_ps( const simde__m256& a ) {
	simde__m256 xm0 = a;
	simde__m256 xm1 = simde_mm256_rcp_ps(xm0);
	return simde_mm256_fnmadd_ps( simde_mm256_mul_ps( xm0, xm1 ), xm1, simde_mm256_add_ps( xm1, xm1 ) );
}

/**
 * SSEで4要素の合計値を求める
 * 合計値は全要素に入る
 */
inline simde__m128 m128_hsum_sse1_ps( simde__m128 sum ) {
	simde__m128 tmp = sum;
	sum = simde_mm_shuffle_ps( sum, tmp, SIMDE_MM_SHUFFLE(1,0,3,2) );
	sum = simde_mm_add_ps( sum, tmp );
	tmp = sum;
	sum = simde_mm_shuffle_ps( sum, tmp, SIMDE_MM_SHUFFLE(2,3,0,1) );
	return simde_mm_add_ps( sum, tmp );
}

/**
 * SSEで4要素の合計値を求める
 * 合計値は全要素に入る
 * SSE3 だと少ない命令で実行できる
 */
inline simde__m128 m128_hsum_sse3_ps( simde__m128 sum ) {
	sum = simde_mm_hadd_ps( sum, sum );
	return simde_mm_hadd_ps( sum, sum );
}

/**
 * AVXで8要素の合計値を求める
 * 合計値は全要素に入る
 */
inline simde__m256 m256_hsum_avx_ps( simde__m256 sum ) {
	sum = simde_mm256_hadd_ps( sum, sum );
	sum = simde_mm256_hadd_ps( sum, sum );
	simde__m256 rsum = simde_mm256_permute2f128_ps(sum, sum, 0 << 4 | 1 );
	sum = simde_mm256_unpacklo_ps( sum, rsum );
	sum = simde_mm256_hadd_ps( sum, sum );
	return sum;
}
/**
 * AVX2で16要素の合計値を求める
 * 合計値は全要素に入る
 */
/*
inline simde__m256 m256_hsum_avx_epi16( simde__m256 sum ) {
	sum = simde_mm256_hadd_ps( sum, sum );
	sum = simde_mm256_hadd_ps( sum, sum );
	return simde_mm256_hadd_ps( sum, sum );
}
*/
#endif
