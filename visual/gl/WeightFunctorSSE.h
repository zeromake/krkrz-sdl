/******************************************************************************/
/**
 * 各種フィルターのウェイトを計算する SSE 版
 * ----------------------------------------------------------------------------
 * 	Copyright (C) T.Imoto <http://www.kaede-software.com>
 * ----------------------------------------------------------------------------
 * @author		T.Imoto
 * @date		2014/04/05
 * @note
 *****************************************************************************/


#ifndef __WEIGHT_FUNCTOR_SSE_H__
#define __WEIGHT_FUNCTOR_SSE_H__

#include "WeightFunctor.h"

/**
 * バイリニア
 */
struct BilinearWeightSSE : public BilinearWeight {
	const simde__m128 absmask;
	const simde__m128 one;
	inline BilinearWeightSSE() : absmask( simde_mm_castsi128_ps(simde_mm_set1_epi32(0x7fffffff)) ), one(simde_mm_set1_ps(1.0f)){}
	inline simde__m128 operator() ( simde__m128 distance ) const {
		simde__m128 x = simde_mm_and_ps( distance, absmask );	// 絶対値
		// 1.0f 以上は0を設定するためのマスク
		//simde__m128 zeromask = simde_mm_cmp_ps( x, one, SIMDE_CMP_LT_OS ); // x < 1.0f
		simde__m128 zeromask = simde_mm_cmplt_ps( x, one ); // x < 1.0f
		x = simde_mm_sub_ps( one, x ); // 1.0f - x
		return simde_mm_and_ps( x, zeromask );	// x >= 1.0f は0へ
	}
};
/**
 * バイキュービック
 */
struct BicubicWeightSSE : public BicubicWeight {
	const simde__m128 M128_P0;
	const simde__m128 M128_P1;
	const simde__m128 M128_P2;
	const simde__m128 M128_P3;
	const simde__m128 M128_P4;
	const simde__m128 M128_COEFF;
	const simde__m128 absmask;
	const simde__m128 one;
	const simde__m128 two;

	/**
	 * @param c : シャープさ。小さくなるにしたがって強くなる
	 */
	inline BicubicWeightSSE( float c = -1 ) : BicubicWeight(c),
		M128_P0( simde_mm_set1_ps( c + 3.0f ) ),
		M128_P1( simde_mm_set1_ps( c + 2.0f ) ),
		M128_P2( simde_mm_set1_ps( -c*4.0f ) ),
		M128_P3( simde_mm_set1_ps( c*8.0f ) ),
		M128_P4( simde_mm_set1_ps( c*5.0f ) ),
		M128_COEFF( simde_mm_set1_ps( c ) ),
		absmask( simde_mm_castsi128_ps(simde_mm_set1_epi32(0x7fffffff)) ),
		one(simde_mm_set1_ps(1.0f)),
		two(simde_mm_set1_ps(2.0f))
	{}

	// 素直に実装すると遅いかも
	inline simde__m128 operator() ( simde__m128 distance ) const {
		simde__m128 x = simde_mm_and_ps( distance, absmask );	// 絶対値
		simde__m128 x2 = simde_mm_mul_ps(x, x);
		simde__m128 x3 = simde_mm_mul_ps(x, x2);

		// 1.0以下の時の値
		simde__m128 t2 = simde_mm_mul_ps( M128_P0, x2 );
		simde__m128 t3 = simde_mm_mul_ps( M128_P1, x3 );
		simde__m128 onev = simde_mm_add_ps( simde_mm_sub_ps( one, t2 ), t3 );

		// 2.0以下の時の値
		t2 = simde_mm_mul_ps( M128_P3, x );
		t3 = simde_mm_mul_ps( M128_P4, x2 );
		simde__m128 t4 = simde_mm_mul_ps( M128_COEFF, x3 );
		simde__m128 twov = simde_mm_add_ps( simde_mm_sub_ps( simde_mm_add_ps( M128_P2, t2 ), t3 ), t4 );

		//simde__m128 onemask = simde_mm_cmp_ps( x, one, SIMDE_CMP_LE_OS ); // x <= 1.0f
		simde__m128 onemask = simde_mm_cmple_ps( x, one ); // x <= 1.0f
		//simde__m128 twomask = simde_mm_cmp_ps( x, two, SIMDE_CMP_LE_OS ); // x <= 2.0f
		simde__m128 twomask = simde_mm_cmple_ps( x, two ); // x <= 2.0f
		twomask = simde_mm_andnot_ps( onemask, twomask );	// x > 1.0f && x <= 2.0f
		onev = simde_mm_and_ps( onev, onemask );	// 1.0以下の時の値
		twov = simde_mm_and_ps( twov, twomask );	// 1.0より大きく、2.0以下の時の値
		return simde_mm_or_ps( onev, twov );	// 2.0以下の値を or して、それ以外は0に
	}
};
/**
 * Lanczos
 */
template<int TTap>
struct LanczosWeightSSE : public LanczosWeight<TTap> {
	const simde__m128 M128_TAP;
	const simde__m128 M128_PI;
	const simde__m128 M128_EPSILON;
	const simde__m128 M128_1_0;
	inline LanczosWeightSSE() : M128_TAP(simde_mm_set1_ps((float)TTap)), M128_PI(simde_mm_set1_ps((float)M_PI)),
		M128_EPSILON(simde_mm_set1_ps(FLT_EPSILON)), M128_1_0(simde_mm_set1_ps(1.0f)) {}
	inline simde__m128 operator() ( simde__m128 dist ) const {
		simde__m128 pidist = simde_mm_mul_ps( dist, M128_PI );	// M_PI * dist
		simde__m128 result = sin_ps( pidist ); // std::sin(M_PI*phase)
		simde__m128 tap = M128_TAP;
		simde__m128 rtap = m128_rcp_22bit_ps( tap );
		result = simde_mm_mul_ps( result, sin_ps( simde_mm_mul_ps( pidist, rtap ) ) ); // std::sin(M_PI*phase)*std::sin(M_PI*phase/TTap)
		pidist = simde_mm_mul_ps( pidist, pidist );
		pidist = simde_mm_mul_ps( pidist, rtap );// M_PI*M_PI*phase*phase/TTap
		pidist = m128_rcp_22bit_ps( pidist );
		result = simde_mm_mul_ps( result, pidist );	// std::sin(M_PI*phase)*std::sin(M_PI*phase/TTap)/(M_PI*M_PI*phase*phase/TTap)
		
		// EPSILON より小さい場合は、1を設定
		simde__m128 onemask = simde_mm_cmplt_ps( dist, M128_EPSILON );
		result = simde_mm_or_ps( simde_mm_andnot_ps( onemask, result ), simde_mm_and_ps( M128_1_0, onemask ) );
		// TAP 以上はゼロを設定
		simde__m128 zeromask = simde_mm_cmpge_ps( dist, tap );
		result = simde_mm_or_ps( simde_mm_andnot_ps( zeromask, result ), simde_mm_and_ps( simde_mm_setzero_ps(), zeromask ) );
		return result;
	}
};

/**
 * Spline16 用ウェイト関数
 */
struct Spline16WeightSSE : public Spline16Weight {
	const simde__m128 one;
	const simde__m128 two;
	const simde__m128 absmask;
	const simde__m128 M128_9_0__5_0;
	const simde__m128 M128_1_0__5_0;
	const simde__m128 M128_1_0__3_0;
	const simde__m128 M128_46_0__15_0;
	const simde__m128 M128_8_0__5_0;
	inline Spline16WeightSSE() : absmask(simde_mm_castsi128_ps(simde_mm_set1_epi32(0x7fffffff))), one(simde_mm_set1_ps(1.0f)), two(simde_mm_set1_ps(2.0f)),
		M128_9_0__5_0(simde_mm_set1_ps(9.0f/5.0f)), M128_1_0__5_0(simde_mm_set1_ps(1.0f/ 5.0f)),
		M128_1_0__3_0(simde_mm_set1_ps(1.0f/3.0f)), M128_46_0__15_0(simde_mm_set1_ps(46.0f/15.0f)),
		M128_8_0__5_0(simde_mm_set1_ps(8.0f/5.0f))
	{}
	inline simde__m128 operator() ( simde__m128 distance ) const {
		simde__m128 x = simde_mm_and_ps( distance, absmask );	// 絶対値
		simde__m128 x2 = simde_mm_mul_ps( x, x );
		simde__m128 x3 = simde_mm_mul_ps( x, x2 );

		// 1以下の時
		const simde__m128 t2_95 = M128_9_0__5_0;
		simde__m128 t2 = simde_mm_mul_ps( x2, t2_95 );	// x*x*9/5
		simde__m128 t3 = simde_mm_mul_ps( x3, M128_1_0__5_0 );	// x*1/5
		simde__m128 onev = simde_mm_add_ps( simde_mm_sub_ps( simde_mm_sub_ps( x3, t2 ), t3 ), one );

		// 2以下の時
		simde__m128 t1 = simde_mm_mul_ps( x3, M128_1_0__3_0 );	// x*x*x*1/3
		t2 = simde_mm_mul_ps( x2, t2_95 );	// x*x*9/5
		t3 = simde_mm_mul_ps( x, M128_46_0__15_0 );	// x*46.0f/15.0f
		simde__m128 twov = simde_mm_add_ps( simde_mm_sub_ps( simde_mm_sub_ps( t2, t1 ), t3 ), M128_8_0__5_0 );

		// 結果を合成
		simde__m128 onemask = simde_mm_cmple_ps( x, one ); // x <= 1.0f
		simde__m128 twomask = simde_mm_cmple_ps( x, two );
		twomask = simde_mm_andnot_ps( onemask, twomask );	// x > 1.0f && x <= 2.0f
		onev = simde_mm_and_ps( onev, onemask );	// 1.0以下の時の値
		twov = simde_mm_and_ps( twov, twomask );	// 1.0より大きく、2.0以下の時の値
		return simde_mm_or_ps( onev, twov );	// 2.0以下の値を or して、それ以外は0に
	}
};

/**
 * Spline36 用ウェイト関数
 */
struct Spline36WeightSSE : public Spline36Weight {
	const simde__m128 one;
	const simde__m128 two;
	const simde__m128 three;
	const simde__m128 absmask;
	const simde__m128 M128_13_0__11_0;
	const simde__m128 M128_453_0__209_0;
	const simde__m128 M128_3_0__209_0;
	const simde__m128 M128_6_0__11_0;
	const simde__m128 M128_612_0__209_0;
	const simde__m128 M128_1038_0__209_0;
	const simde__m128 M128_540_0__209_0;
	const simde__m128 M128_1_0__11_0;
	const simde__m128 M128_159_0__209_0;
	const simde__m128 M128_434_0__209_0;
	const simde__m128 M128_384_0__209_0;
	inline Spline36WeightSSE() : absmask(simde_mm_castsi128_ps(simde_mm_set1_epi32(0x7fffffff))),
		one(simde_mm_set1_ps(1.0f)), two(simde_mm_set1_ps(2.0f)), three(simde_mm_set1_ps(3.0f)),
		M128_13_0__11_0(simde_mm_set1_ps(13.0f/11.0f)), 
		M128_453_0__209_0(simde_mm_set1_ps(453.0f/209.0f)), 
		M128_3_0__209_0(simde_mm_set1_ps(3.0f/209.0f)), 
		M128_6_0__11_0(simde_mm_set1_ps(6.0f/11.0f)), 
		M128_612_0__209_0(simde_mm_set1_ps(612.0f/209.0f)), 
		M128_1038_0__209_0(simde_mm_set1_ps(1038.0f/209.0f)), 
		M128_540_0__209_0(simde_mm_set1_ps(540.0f/209.0f)), 
		M128_1_0__11_0(simde_mm_set1_ps(1.0f/11.0f)), 
		M128_159_0__209_0(simde_mm_set1_ps(159.0f/209.0f)), 
		M128_434_0__209_0(simde_mm_set1_ps(434.0f/209.0f)), 
		M128_384_0__209_0(simde_mm_set1_ps(384.0f/209.0f)) {}
	inline simde__m128 operator() ( simde__m128 distance ) const {
		simde__m128 x = simde_mm_and_ps( distance, absmask );	// 絶対値
		simde__m128 x2 = simde_mm_mul_ps( x, x );
		simde__m128 x3 = simde_mm_mul_ps( x, x2 );

		// 1.0以下の時
		simde__m128 t1 = simde_mm_mul_ps( x3, M128_13_0__11_0 );
		simde__m128 t2 = simde_mm_mul_ps( x2, M128_453_0__209_0 );
		simde__m128 t3 = simde_mm_mul_ps( x, M128_3_0__209_0 );
		simde__m128 onev = simde_mm_add_ps( simde_mm_sub_ps( simde_mm_sub_ps( t1, t2 ), t3 ), one );

		// 2.0以下の時
		t1 = simde_mm_mul_ps( x3, M128_6_0__11_0 );
		t2 = simde_mm_mul_ps( x2, M128_612_0__209_0 );
		t3 = simde_mm_mul_ps( x, M128_1038_0__209_0 );
		simde__m128 twov = simde_mm_add_ps( simde_mm_sub_ps( simde_mm_sub_ps( t2, t1 ), t3 ), M128_540_0__209_0 );

		// 3.0以下の時
		t1 = simde_mm_mul_ps( x3, M128_1_0__11_0 );
		t2 = simde_mm_mul_ps( x2, M128_159_0__209_0 );
		t3 = simde_mm_mul_ps( x, M128_434_0__209_0 );
		simde__m128 thrv = simde_mm_sub_ps( simde_mm_add_ps( simde_mm_sub_ps( t1, t2 ), t3 ), M128_384_0__209_0 );

		// 結果を合成
		simde__m128 onemask = simde_mm_cmple_ps( x, one ); // x <= 1.0f
		simde__m128 twomask = simde_mm_cmple_ps( x, two ); // x <= 2.0f
		simde__m128 thrmask = simde_mm_cmple_ps( x, three ); // x <= 3.0f
		thrmask = simde_mm_andnot_ps( twomask, thrmask );	// x > 2.0f && x <= 3.0f
		twomask = simde_mm_andnot_ps( onemask, twomask );	// x > 1.0f && x <= 2.0f
		onev = simde_mm_and_ps( onev, onemask );	// 1.0以下の時の値
		twov = simde_mm_and_ps( twov, twomask );	// 1.0より大きく、2.0以下の時の値
		thrv = simde_mm_and_ps( thrv, thrmask );	// 2.0より大きく、3.0以下の時の値
		return simde_mm_or_ps( simde_mm_or_ps( onev, twov ), thrv );
	}
};

/**
 * ガウス関数
 */
struct GaussianWeightSSE : public GaussianWeight {
	const simde__m128 M128_SQ2PI;
	const simde__m128 M128_M2_0;
	inline GaussianWeightSSE() : M128_SQ2PI(simde_mm_set1_ps(std::sqrt( 2.0f/(float)M_PI ))), M128_M2_0(simde_mm_set1_ps(-2.0f)) {}
	inline simde__m128 operator() ( simde__m128 x ) const {
		simde__m128 x2 = simde_mm_mul_ps( x, x );
		x2 = simde_mm_mul_ps( x2, M128_M2_0 );
		return simde_mm_mul_ps( exp_ps( x2 ), M128_SQ2PI );
	}
};

/**
 * Blackman-Sinc 関数
 */
struct BlackmanSincWeightSSE : public BlackmanSincWeight {
	const simde__m128 absmask;
	const simde__m128 M128_COEFF;
	const simde__m128 M128_EPSILON;
	const simde__m128 M128_1_0;
	const simde__m128 M128_PI;
	const simde__m128 M128_0_5;
	const simde__m128 M128_2_0;
	const simde__m128 M128_0_08;
	const simde__m128 M128_0_42;
	inline BlackmanSincWeightSSE( float c = RANGE ) : BlackmanSincWeight(c), absmask(simde_mm_castsi128_ps(simde_mm_set1_epi32(0x7fffffff))),
		M128_COEFF(simde_mm_set1_ps(1.0f/c)),
		M128_EPSILON(simde_mm_set1_ps(FLT_EPSILON)),
		M128_1_0(simde_mm_set1_ps(1.0f)),
		M128_PI(simde_mm_set1_ps((float)M_PI)),
		M128_0_5(simde_mm_set1_ps(0.5f)),
		M128_2_0(simde_mm_set1_ps(2.0f)),
		M128_0_08(simde_mm_set1_ps(0.08f)),
		M128_0_42(simde_mm_set1_ps(0.42f)) {}
	inline simde__m128 operator() ( simde__m128 distance ) const {
		simde__m128 x = simde_mm_and_ps( distance, absmask );	// 絶対値
		simde__m128 pix = simde_mm_mul_ps( x, M128_PI );	// PI * x

		simde__m128 coeff = M128_COEFF;
		coeff = simde_mm_mul_ps( coeff, pix );	// coeff * PI * x

		simde__m128 t2 = cos_ps( coeff );	// cos(coeff * PI * x)
		t2 = simde_mm_mul_ps( t2, M128_0_5 );	// 0.5 * cos(coeff * PI * x)

		simde__m128 t3 = simde_mm_mul_ps( coeff, M128_2_0 );	// 2.0 * coeff * PI * x
		t3 = cos_ps( t3 );	// cos(2.0 * coeff * PI * x)
		t3 = simde_mm_mul_ps( t3, M128_0_08 );	// 0.08 * cos(2.0 * coeff * PI * x)

		// 0.42 + 0.5 * cos(PI * x * coeff) + 0.08 * cos(2.0 * PI * x * coeff)
		simde__m128 result = simde_mm_add_ps( simde_mm_add_ps( t2, t3 ), M128_0_42 );	// + 0.42

		simde__m128 sincpix = sin_ps( pix );	// sin(PI * x)
		pix  = m128_rcp_22bit_ps( pix );	// 1.0 / (PI * x)
		sincpix = simde_mm_mul_ps( sincpix, pix );	// sin(PI * x) / (PI * x)
		result = simde_mm_mul_ps( result, sincpix );	// result

		simde__m128 one = M128_1_0;
		simde__m128 onemask = simde_mm_cmplt_ps( x, M128_EPSILON );
		one = simde_mm_and_ps( one, onemask );	// x < epsilon ? 1.0
		onemask = simde_mm_andnot_ps( onemask, result );	// x >= epsilon ? result
		return simde_mm_or_ps( one, onemask );
	}
};

#endif // __WEIGHT_FUNCTOR_SSE_H__
