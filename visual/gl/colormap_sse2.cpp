

#include "tjsCommHead.h"
#include "tvpgl.h"
#include "tvpgl_ia32_intf.h"
#include "tvpgl_mathutil.h"
#include "simd_def_x86x64.h"
#include <string.h>

extern "C" {
extern unsigned char TVPNegativeMulTable65[65*256];
extern unsigned char TVPOpacityOnOpacityTable65[65*256];
};

// tshift = 6 : 65
// tshift = 8 : normal(255)
template<int tshift>
struct sse2_apply_color_map_xx_functor {
	simde__m128i mc_;
	simde__m128i color_;
	const simde__m128i zero_;
	inline sse2_apply_color_map_xx_functor( tjs_uint32 color ) : zero_(simde_mm_setzero_si128()) {
		mc_ = simde_mm_cvtsi32_si128( color );
		mc_ = simde_mm_shuffle_epi32( mc_, SIMDE_MM_SHUFFLE( 0, 0, 0, 0 )  );
		color_ = mc_;
		mc_ = simde_mm_unpacklo_epi8( mc_, zero_ );
	}
	inline tjs_uint32 operator()( tjs_uint32 d, tjs_uint8 s ) const {
		simde__m128i md = simde_mm_cvtsi32_si128( d );
		md = simde_mm_unpacklo_epi8( md, zero_ );
		simde__m128i mo = simde_mm_cvtsi32_si128( s );
		mo = simde_mm_shufflelo_epi16( mo, SIMDE_MM_SHUFFLE( 0, 0, 0, 0 )  );	// 00oo00oo00oo00oo
		simde__m128i mc = mc_;
		mc = simde_mm_sub_epi16( mc, md );	// c - d
		mc = simde_mm_mullo_epi16( mc, mo );	// c *= opa
		mc = simde_mm_srai_epi16( mc, tshift );	// c >>= tshift
		md = simde_mm_add_epi16( md, mc );	// d += c
		md = simde_mm_packus_epi16( md, zero_ );
		return simde_mm_cvtsi128_si32( md );
	}
	inline simde__m128i operator()( simde__m128i md1, simde__m128i mo1 ) const {
		simde__m128i md2 = md1;
		md1 = simde_mm_unpacklo_epi8( md1, zero_ );
		simde__m128i mo2 = mo1;
		mo1 = simde_mm_unpacklo_epi16( mo1, mo1 );
		mo2 = simde_mm_unpackhi_epi16( mo2, mo2 );

		simde__m128i mc = mc_;
		mc = simde_mm_sub_epi16( mc, md1 );		// c - d
		mc = simde_mm_mullo_epi16( mc, mo1 );	// c *= opa
		mc = simde_mm_srai_epi16( mc, tshift );	// c >>= tshift
		md1 = simde_mm_add_epi16( md1, mc );		// d += c

		mc = mc_;
		md2 = simde_mm_unpackhi_epi8( md2, zero_ );
		mc = simde_mm_sub_epi16( mc, md2 );		// c - d
		mc = simde_mm_mullo_epi16( mc, mo2 );	// c *= opa
		mc = simde_mm_srai_epi16( mc, tshift );	// c >>= tshift
		md2 = simde_mm_add_epi16( md2, mc );		// d += c

		return simde_mm_packus_epi16( md1, md2 );
	}
};
struct sse2_apply_color_map256_functor {
	simde__m128i mc_;
	simde__m128i color_;
	const simde__m128i zero_;
	inline sse2_apply_color_map256_functor( tjs_uint32 color ) : zero_(simde_mm_setzero_si128()) {
		mc_ = simde_mm_cvtsi32_si128( color );
		mc_ = simde_mm_shuffle_epi32( mc_, SIMDE_MM_SHUFFLE( 0, 0, 0, 0 )  );
		color_ = mc_;
		mc_ = simde_mm_unpacklo_epi8( mc_, zero_ );
	}
	inline tjs_uint32 operator()( tjs_uint32 d, tjs_uint8 s ) const {
		simde__m128i md = simde_mm_cvtsi32_si128( d );
		md = simde_mm_unpacklo_epi8( md, zero_ );
		simde__m128i mo = simde_mm_cvtsi32_si128( s );
		mo = simde_mm_shufflelo_epi16( mo, SIMDE_MM_SHUFFLE( 0, 0, 0, 0 )  );	// 00oo00oo00oo00oo
		simde__m128i mc = mc_;
		mc = simde_mm_sub_epi16( mc, md );	// c - d
		mc = simde_mm_mullo_epi16( mc, mo );	// c *= opa
		mc = simde_mm_srli_epi16( mc, 8 );	// c >>= tshift
		md = simde_mm_add_epi8( md, mc );	// d += c
		md = simde_mm_packus_epi16( md, zero_ );
		return simde_mm_cvtsi128_si32( md );
	}
	inline simde__m128i operator()( simde__m128i md1, simde__m128i mo1 ) const {
		simde__m128i md2 = md1;
		md1 = simde_mm_unpacklo_epi8( md1, zero_ );
		simde__m128i mo2 = mo1;
		mo1 = simde_mm_unpacklo_epi16( mo1, mo1 );
		mo2 = simde_mm_unpackhi_epi16( mo2, mo2 );

		simde__m128i mc = mc_;
		mc = simde_mm_sub_epi16( mc, md1 );		// c - d
		mc = simde_mm_mullo_epi16( mc, mo1 );	// c *= opa
		mc = simde_mm_srli_epi16( mc, 8 );	// c >>= tshift
		md1 = simde_mm_add_epi8( md1, mc );		// d += c

		mc = mc_;
		md2 = simde_mm_unpackhi_epi8( md2, zero_ );
		mc = simde_mm_sub_epi16( mc, md2 );		// c - d
		mc = simde_mm_mullo_epi16( mc, mo2 );	// c *= opa
		mc = simde_mm_srli_epi16( mc, 8 );	// c >>= tshift
		md2 = simde_mm_add_epi8( md2, mc );		// d += c

		return simde_mm_packus_epi16( md1, md2 );
	}
};

template<typename tbase>
struct sse2_apply_color_map_xx_o_functor : tbase {
	tjs_int opa32_;
	simde__m128i opa_;
	const simde__m128i zero_;
	inline sse2_apply_color_map_xx_o_functor( tjs_uint32 color, tjs_int opa ) : zero_(simde_mm_setzero_si128()), tbase(color), opa32_(opa) {
		opa_ = simde_mm_cvtsi32_si128( opa );
		opa_ = simde_mm_shufflelo_epi16( opa_, SIMDE_MM_SHUFFLE( 0, 0, 0, 0 )  );
	}
	inline tjs_uint32 operator()( tjs_uint32 d, tjs_uint8 s ) const {
		return tbase::operator()( d, (tjs_uint8)( (s*opa32_)>>8) );
	}
	inline simde__m128i operator()( simde__m128i md1, tjs_uint32 s ) const {
		simde__m128i mo = simde_mm_cvtsi32_si128( s );
		mo = simde_mm_unpacklo_epi8( mo, zero_ );	// 0000 0o 0o 0o 0o
		mo = simde_mm_mullo_epi16( mo, opa_ );
		mo = simde_mm_srli_epi16( mo, 8 );
		mo = simde_mm_unpacklo_epi16( mo, mo );	// 1 1 2 2 3 3 4 4
		return tbase::operator()( md1, mo );
	}
};
template<typename tbase>
struct sse2_apply_color_map_xx_straight_functor : tbase {
	const simde__m128i zero_;
	inline sse2_apply_color_map_xx_straight_functor( tjs_uint32 color ) : zero_(simde_mm_setzero_si128()), tbase(color) {}
	inline tjs_uint32 operator()( tjs_uint32 d, tjs_uint8 s ) const {
		return tbase::operator()( d, s );
	}
	inline simde__m128i operator()( simde__m128i md1, tjs_uint32 s ) const {
		simde__m128i mo = simde_mm_cvtsi32_si128( s );
		mo = simde_mm_unpacklo_epi8( mo, zero_ );	// 0000 0o 0o 0o 0o
		mo = simde_mm_unpacklo_epi16( mo, mo );		// 1 1 2 2 3 3 4 4
		return tbase::operator()( md1, mo );
	}
};
typedef sse2_apply_color_map_xx_straight_functor<sse2_apply_color_map_xx_functor<6> > sse2_apply_color_map65_functor;
typedef sse2_apply_color_map_xx_straight_functor<sse2_apply_color_map256_functor > sse2_apply_color_map_functor;
typedef sse2_apply_color_map_xx_o_functor<sse2_apply_color_map_xx_functor<6> > sse2_apply_color_map65_o_functor;
typedef sse2_apply_color_map_xx_o_functor<sse2_apply_color_map256_functor > sse2_apply_color_map_o_functor;

struct sse2_apply_color_map65_d_functor {
	simde__m128i mc_;
	simde__m128i color_;
	const simde__m128i zero_;
	const simde__m128i colormask_;
	inline sse2_apply_color_map65_d_functor( tjs_uint32 color ) : zero_(simde_mm_setzero_si128()), colormask_(simde_mm_set1_epi32(0x00ffffff)) {
		mc_ = simde_mm_cvtsi32_si128( color|0xff000000 );
		mc_ = simde_mm_shuffle_epi32( mc_, SIMDE_MM_SHUFFLE( 0, 0, 0, 0 )  );
		color_ = mc_;
		mc_ = simde_mm_unpacklo_epi8( mc_, zero_ );
	}
	inline tjs_uint32 operator()( tjs_uint32 d, tjs_uint8 s ) const {
		if( s == 0 ) return d;
		simde__m128i mc = mc_;	// color
		simde__m128i md = simde_mm_cvtsi32_si128( d );
		md = simde_mm_unpacklo_epi8( md, zero_ );
		tjs_int addr = (s << 8) | (d >> 24);
		tjs_uint32 dopa = TVPNegativeMulTable65[addr];		// dest opa table
		tjs_uint32 o = TVPOpacityOnOpacityTable65[addr];	// blend opa table
		dopa <<= 24;
		simde__m128i mo = simde_mm_cvtsi32_si128( o );
		mo = simde_mm_shufflelo_epi16( mo, SIMDE_MM_SHUFFLE( 0, 0, 0, 0 )  );	// 00oo00oo00oo00oo
		mc = simde_mm_sub_epi16( mc, md );	// mc -= md
		mc = simde_mm_mullo_epi16( mc, mo );	// mc *= mo
		md = simde_mm_slli_epi16( md, 8 );	// md <<= 8
		md = simde_mm_add_epi16( md, mc );	// md += mc
		md = simde_mm_srli_epi16( md, 8 );	// md >>= 8
		md = simde_mm_packus_epi16( md, zero_ );
		tjs_uint32 ret = simde_mm_cvtsi128_si32( md );
		return (ret&0x00ffffff)|dopa;
	}
	inline simde__m128i operator()( simde__m128i md1, tjs_uint32 s ) const {
		simde__m128i da = md1;
		da = simde_mm_srli_epi32( da, 24 );	// >>= 24
		simde__m128i opa = simde_mm_cvtsi32_si128( s );
		opa = simde_mm_unpacklo_epi8( opa, zero_ );	// 0000 | 0 s 0 s 0 s 0 s
		simde__m128i dopa = opa;	// 4倍して飽和してって処理長いからテーブルの方がいいかも…
		dopa = simde_mm_slli_epi16( dopa, 2 );			// *= 4
		dopa = simde_mm_packus_epi16( dopa, zero_ );		// dopa > 255 ? 255 : dopa
		dopa = simde_mm_unpacklo_epi8( dopa, zero_ );	// 0000 | 0 s 0 s 0 s 0 s
		dopa = simde_mm_unpacklo_epi16( dopa, zero_ );	// 000s 000s 000s 000s
		dopa = simde_mm_slli_epi32( dopa, 8 );			// <<= 8
		dopa = simde_mm_or_si128( dopa, da );

		opa = simde_mm_unpacklo_epi16( opa, zero_ );	// 000s 000s 000s 000s
		opa = simde_mm_slli_epi32( opa, 8 );			// <<= 8
		da = simde_mm_or_si128( da, opa );
		simde__m128i ma1 = simde_mm_set_epi32(
			TVPOpacityOnOpacityTable65[simde_mm_extract_epi16(da,6)],
			TVPOpacityOnOpacityTable65[simde_mm_extract_epi16(da,4)],
			TVPOpacityOnOpacityTable65[simde_mm_extract_epi16(da,2)],
			TVPOpacityOnOpacityTable65[simde_mm_extract_epi16(da,0)]);

		ma1 = simde_mm_packs_epi32( ma1, ma1 );		// 0 1 2 3 0 1 2 3
		ma1 = simde_mm_unpacklo_epi16( ma1, ma1 );	// 0 0 1 1 2 2 3 3
		simde__m128i ma2 = ma1;
		ma1 = simde_mm_unpacklo_epi16( ma1, ma1 );	// 0 0 0 0 1 1 1 1

		simde__m128i mc = mc_;
		simde__m128i md2 = md1;
		md1 = simde_mm_unpacklo_epi8( md1, zero_ );
		mc = simde_mm_sub_epi16( mc, md1 );	// mc -= md
		mc = simde_mm_mullo_epi16( mc, ma1 );// mc *= mo
		md1 = simde_mm_slli_epi16( md1, 8 );	// md <<= 8
		md1 = simde_mm_add_epi16( md1, mc );	// md += mc
		md1 = simde_mm_srli_epi16( md1, 8 );	// md >>= 8

		mc = mc_;
		ma2 = simde_mm_unpackhi_epi16( ma2, ma2 );	// 2 2 2 2 3 3 3 3
		md2 = simde_mm_unpackhi_epi8( md2, zero_ );
		mc = simde_mm_sub_epi16( mc, md2 );	// mc -= md
		mc = simde_mm_mullo_epi16( mc, ma2 );// mc *= mo
		md2 = simde_mm_slli_epi16( md2, 8 );	// md <<= 8
		md2 = simde_mm_add_epi16( md2, mc );	// md += mc
		md2 = simde_mm_srli_epi16( md2, 8 );	// md >>= 8

		md1 = simde_mm_packus_epi16( md1, md2 );

		// ( 255 - (255-a)*(255-b)/ 255 ); 
		simde__m128i mask = colormask_;
		mask = simde_mm_srli_epi32( mask, 8 );	// 0x00ffffff >> 8 = 0x0000ffff
		dopa = simde_mm_xor_si128( dopa, mask );	// (a = 255-a, b = 255-b) : ^=xor : 普通に8bit単位で引いても一緒か……
		simde__m128i mtmp = dopa;
		dopa = simde_mm_slli_epi32( dopa, 8 );		// 00ff|ff00	上位 << 8
		mtmp = simde_mm_slli_epi16( mtmp, 8 );		// 0000|ff00	下位 << 8
		mtmp = simde_mm_slli_epi32( mtmp, 8 );		// 00ff|0000
		dopa = simde_mm_mullo_epi16( dopa, mtmp );	// 上位で演算、下位部分はごみ
		dopa = simde_mm_srli_epi32( dopa, 16 );		// addr >> 16 | 下位を捨てる
		dopa = simde_mm_andnot_si128( dopa, mask );	// ~addr&0x0000ffff
		dopa = simde_mm_srli_epi16( dopa, 8 );		// addr>>8
		dopa = simde_mm_slli_epi32( dopa, 24 );		// アルファ位置へ

		md1 = simde_mm_and_si128( md1, colormask_ );
		return simde_mm_or_si128( md1, dopa );
	}
};
template<int tshift>
struct sse2_apply_color_map_xx_a_functor {
	simde__m128i mc_;
	simde__m128i color_;
	const simde__m128i zero_;
	inline sse2_apply_color_map_xx_a_functor( tjs_uint32 color ) : zero_(simde_mm_setzero_si128()) {
		mc_ = simde_mm_cvtsi32_si128( color&0x00ffffff );
		simde__m128i tmp = simde_mm_cvtsi32_si128( 0x100 );
		tmp = simde_mm_slli_epi64( tmp, 48 );	// << 48
		mc_ = simde_mm_unpacklo_epi8( mc_, zero_ );
		mc_ = simde_mm_or_si128( mc_, tmp );		// 01 00 00 co 00 co 00 co
		mc_ = simde_mm_shuffle_epi32( mc_, SIMDE_MM_SHUFFLE( 1, 0, 1, 0 )  );
		color_ = simde_mm_cvtsi32_si128( color|0xff000000 );
		color_ = simde_mm_shuffle_epi32( color_, SIMDE_MM_SHUFFLE( 0, 0, 0, 0 )  );
	}
	inline tjs_uint32 operator()( tjs_uint32 d, tjs_uint8 s ) const {
		simde__m128i mo = simde_mm_cvtsi32_si128( s );
		mo = simde_mm_shufflelo_epi16( mo, SIMDE_MM_SHUFFLE( 0, 0, 0, 0 )  );	// 00 Sa 00 Sa 00 Sa 00 Sa
		simde__m128i ms = mo;
		ms = simde_mm_mullo_epi16( ms, mc_ );		// alpha * color
		ms = simde_mm_srli_epi16( ms, tshift );		// 00 Sa 00 Si 00 Si 00 Si
		simde__m128i md = simde_mm_cvtsi32_si128( d );	// (DaDiDiDi)
		md = simde_mm_unpacklo_epi8( md, zero_ );	// 00 Da 00 Di 00 Di 00 Di
		simde__m128i mds = md;
		mds = simde_mm_mullo_epi16( mds, mo );		// dest * alpha
		mds = simde_mm_srli_epi16( mds, tshift );	// 00 SaDa 00 SaDi 00 SaDi 00 SaDi
		md = simde_mm_sub_epi16( md, mds );			// Di - SaDi
		md = simde_mm_add_epi16( md, ms );			// Di - SaDi + Si
		md = simde_mm_packus_epi16( md, zero_ );
		return simde_mm_cvtsi128_si32( md );
	}
	inline simde__m128i operator()( simde__m128i md1, simde__m128i mo1 ) const {
		simde__m128i md2 = md1;
		md1 = simde_mm_unpacklo_epi8( md1, zero_ );	// 00 Da 00 Di 00 Di 00 Di
		simde__m128i mo2 = mo1;
		mo1 = simde_mm_unpacklo_epi16( mo1, mo1 );
		mo2 = simde_mm_unpackhi_epi16( mo2, mo2 );

		simde__m128i ms = mo1;
		ms = simde_mm_mullo_epi16( ms, mc_ );		// alpha * color
		ms = simde_mm_srli_epi16( ms, tshift );		// 00 Sa 00 Si 00 Si 00 Si
		simde__m128i mds = md1;
		mds = simde_mm_mullo_epi16( mds, mo1 );		// dest * alpha
		mds = simde_mm_srli_epi16( mds, tshift );			// 00 SaDa 00 SaDi 00 SaDi 00 SaDi
		md1 = simde_mm_sub_epi16( md1, mds );		// Di - SaDi
		md1 = simde_mm_add_epi16( md1, ms );			// Di - SaDi + Si
		
		md2 = simde_mm_unpackhi_epi8( md2, zero_ );	// 00 Da 00 Di 00 Di 00 Di
		ms = mo2;
		ms = simde_mm_mullo_epi16( ms, mc_ );		// alpha * color
		ms = simde_mm_srli_epi16( ms, tshift );		// 00 Sa 00 Si 00 Si 00 Si
		mds = md2;
		mds = simde_mm_mullo_epi16( mds, mo2 );		// dest * alpha
		mds = simde_mm_srli_epi16( mds, tshift );	// 00 SaDa 00 SaDi 00 SaDi 00 SaDi
		md2 = simde_mm_sub_epi16( md2, mds );		// Di - SaDi
		md2 = simde_mm_add_epi16( md2, ms );			// Di - SaDi + Si

		return simde_mm_packus_epi16( md1, md2 );
	}
};

typedef sse2_apply_color_map_xx_straight_functor<sse2_apply_color_map_xx_a_functor<6> > sse2_apply_color_map65_a_functor;
typedef sse2_apply_color_map_xx_straight_functor<sse2_apply_color_map_xx_a_functor<8> > sse2_apply_color_map_a_functor;
typedef sse2_apply_color_map_xx_o_functor<sse2_apply_color_map_xx_a_functor<6> > sse2_apply_color_map65_ao_functor;
typedef sse2_apply_color_map_xx_o_functor<sse2_apply_color_map_xx_a_functor<8> > sse2_apply_color_map_ao_functor;


template<typename functor,tjs_uint32 topaque>
static inline void apply_color_map_branch_func_sse2( tjs_uint32 *dest, const tjs_uint8 *src, tjs_int len, const functor& func ) {
	if( len <= 0 ) return;

	tjs_int count = (tjs_int)((size_t)dest & 0xF);
	if( count ) {
		count = (16 - count)>>2;
		count = count > len ? len : count;
		tjs_uint32* limit = dest + count;
		while( dest < limit ) {
			*dest = func( *dest, *src );
			dest++; src++;
		}
		len -= count;
	}
	tjs_uint32 rem = (len>>2)<<2;
	tjs_uint32* limit = dest + rem;
	while( dest < limit ) {
		tjs_uint32 s = *(const tjs_uint32*)src;
		if( s == topaque ) { // completely opaque
			simde_mm_store_si128( (simde__m128i*)dest, func.color_ );
		} else if( s != 0 ) {
			simde__m128i md = simde_mm_load_si128( (simde__m128i const*)dest );
			simde_mm_store_si128( (simde__m128i*)dest, func( md, s ) );
		} // else { // completely transparent
		dest+=4; src+=4;
	}
	limit += (len-rem);
	while( dest < limit ) {
		*dest = func( *dest, *src );
		dest++; src++;
	}
}

template<typename functor>
static inline void apply_color_map_func_sse2( tjs_uint32 *dest, const tjs_uint8 *src, tjs_int len, const functor& func ) {
	if( len <= 0 ) return;

	tjs_int count = (tjs_int)((size_t)dest & 0xF);
	if( count ) {
		count = (16 - count)>>2;
		count = count > len ? len : count;
		tjs_uint32* limit = dest + count;
		while( dest < limit ) {
			*dest = func( *dest, *src );
			dest++; src++;
		}
		len -= count;
	}
	tjs_uint32 rem = (len>>2)<<2;
	tjs_uint32* limit = dest + rem;
	while( dest < limit ) {
		tjs_uint32 s = *(const tjs_uint32*)src;
		simde__m128i md = simde_mm_load_si128( (simde__m128i const*)dest );
		simde_mm_store_si128( (simde__m128i*)dest, func( md, s ) );
		dest+=4; src+=4;
	}
	limit += (len-rem);
	while( dest < limit ) {
		*dest = func( *dest, *src );
		dest++; src++;
	}
}

void TVPApplyColorMap65_sse2_c(tjs_uint32 *dest, const tjs_uint8 *src, tjs_int len, tjs_uint32 color) {
	sse2_apply_color_map65_functor func(color);
	apply_color_map_branch_func_sse2<sse2_apply_color_map65_functor,0x40404040>( dest, src, len , func );
}
void TVPApplyColorMap_sse2_c(tjs_uint32 *dest, const tjs_uint8 *src, tjs_int len, tjs_uint32 color) {
	sse2_apply_color_map_functor func(color);
	apply_color_map_branch_func_sse2<sse2_apply_color_map_functor,0xffffffff>( dest, src, len , func );
}
void TVPApplyColorMap65_d_sse2_c(tjs_uint32 *dest, const tjs_uint8 *src, tjs_int len, tjs_uint32 color) {
	sse2_apply_color_map65_d_functor func( color );
	apply_color_map_branch_func_sse2<sse2_apply_color_map65_d_functor,0x40404040>( dest, src, len , func );
}
void TVPApplyColorMap65_a_sse2_c(tjs_uint32 *dest, const tjs_uint8 *src, tjs_int len, tjs_uint32 color) {
	sse2_apply_color_map65_a_functor func( color );
	apply_color_map_branch_func_sse2<sse2_apply_color_map65_a_functor,0x40404040>( dest, src, len , func );
}
void TVPApplyColorMap_a_sse2_c(tjs_uint32 *dest, const tjs_uint8 *src, tjs_int len, tjs_uint32 color) {
	sse2_apply_color_map_a_functor func( color );
	apply_color_map_branch_func_sse2<sse2_apply_color_map_a_functor,0xffffffff>( dest, src, len , func );
}
void TVPApplyColorMap65_o_sse2_c(tjs_uint32 *dest, const tjs_uint8 *src, tjs_int len, tjs_uint32 color, tjs_int opa) {
	sse2_apply_color_map65_o_functor func(color,opa);
	apply_color_map_func_sse2( dest, src, len , func );
}
void TVPApplyColorMap_o_sse2_c(tjs_uint32 *dest, const tjs_uint8 *src, tjs_int len, tjs_uint32 color, tjs_int opa) {
	sse2_apply_color_map_o_functor func(color,opa);
	apply_color_map_func_sse2( dest, src, len , func );
}
void TVPApplyColorMap65_ao_sse2_c(tjs_uint32 *dest, const tjs_uint8 *src, tjs_int len, tjs_uint32 color, tjs_int opa) {
	sse2_apply_color_map65_ao_functor func(color,opa);
	apply_color_map_func_sse2( dest, src, len , func );
}
void TVPApplyColorMap_ao_sse2_c(tjs_uint32 *dest, const tjs_uint8 *src, tjs_int len, tjs_uint32 color, tjs_int opa) {
	sse2_apply_color_map_ao_functor func(color,opa);
	apply_color_map_func_sse2( dest, src, len , func );
}


/*
void TVPApplyColorMap(tjs_uint32 *dest, const tjs_uint8 *src, tjs_int len, tjs_uint32 color); // has SSE2
void TVPApplyColorMap_o(tjs_uint32 *dest, const tjs_uint8 *src, tjs_int len, tjs_uint32 color, tjs_int opa); // has SSE2
void TVPApplyColorMap65(tjs_uint32 *dest, const tjs_uint8 *src, tjs_int len, tjs_uint32 color); // has SSE2
void TVPApplyColorMap65_o(tjs_uint32 *dest, const tjs_uint8 *src, tjs_int len, tjs_uint32 color, tjs_int opa); // has SSE2
void TVPApplyColorMap_HDA(tjs_uint32 *dest, const tjs_uint8 *src, tjs_int len, tjs_uint32 color);
void TVPApplyColorMap_HDA_o(tjs_uint32 *dest, const tjs_uint8 *src, tjs_int len, tjs_uint32 color, tjs_int opa);
void TVPApplyColorMap65_HDA(tjs_uint32 *dest, const tjs_uint8 *src, tjs_int len, tjs_uint32 color);
void TVPApplyColorMap65_HDA_o(tjs_uint32 *dest, const tjs_uint8 *src, tjs_int len, tjs_uint32 color, tjs_int opa);
void TVPApplyColorMap_d(tjs_uint32 *dest, const tjs_uint8 *src, tjs_int len, tjs_uint32 color);
void TVPApplyColorMap65_d(tjs_uint32 *dest, const tjs_uint8 *src, tjs_int len, tjs_uint32 color); // has SSE2
void TVPApplyColorMap_a(tjs_uint32 *dest, const tjs_uint8 *src, tjs_int len, tjs_uint32 color); // has SSE2
void TVPApplyColorMap65_a(tjs_uint32 *dest, const tjs_uint8 *src, tjs_int len, tjs_uint32 color); // has SSE2
void TVPApplyColorMap_do(tjs_uint32 *dest, const tjs_uint8 *src, tjs_int len, tjs_uint32 color, tjs_int opa);
void TVPApplyColorMap65_do(tjs_uint32 *dest, const tjs_uint8 *src, tjs_int len, tjs_uint32 color, tjs_int opa);
void TVPApplyColorMap_ao(tjs_uint32 *dest, const tjs_uint8 *src, tjs_int len, tjs_uint32 color, tjs_int opa); // has SSE2
void TVPApplyColorMap65_ao(tjs_uint32 *dest, const tjs_uint8 *src, tjs_int len, tjs_uint32 color, tjs_int opa); // has SSE2
*/


struct sse2_ch_blur_mul_copy_functor {
	const simde__m128i zero_;
	const simde__m128i levello_;
	const simde__m128i levelhi_;
	const tjs_int level32_;
	inline sse2_ch_blur_mul_copy_functor( tjs_int level )
		: zero_(simde_mm_setzero_si128()), level32_(level)
		, levello_(simde_mm_set1_epi16(level&0xffff)), levelhi_(simde_mm_set1_epi16(level>>16))
	{}
	inline tjs_uint8 operator()( tjs_uint8 d, tjs_uint8 s ) const {
		tjs_int a = (s * level32_ >> 18);
		if(a>=255) a = 255;
		return (tjs_uint8)a;
	}
	inline tjs_uint32 operator()( tjs_uint32 d, tjs_uint32 s ) const {
		simde__m128i ms1 = simde_mm_cvtsi32_si128(s);
		ms1 = simde_mm_unpacklo_epi8( ms1, zero_ );
		simde__m128i ms1l = ms1;
		ms1l = simde_mm_mulhi_epu16( ms1l, levello_ );
		ms1 = simde_mm_mullo_epi16( ms1, levelhi_ );
		ms1 = simde_mm_adds_epu16( ms1, ms1l );
		ms1 = simde_mm_srli_epi16( ms1, 2 );
		ms1 = simde_mm_packus_epi16( ms1, zero_ );
		return simde_mm_cvtsi128_si32( ms1 );
	}
	inline simde__m128i operator()( simde__m128i md1, simde__m128i ms1 ) const {
		simde__m128i ms2 = ms1;
		ms1 = simde_mm_unpacklo_epi8( ms1, zero_ );
		ms2 = simde_mm_unpackhi_epi8( ms2, zero_ );
		simde__m128i ms1l = ms1;
		ms1l = simde_mm_mulhi_epu16( ms1l, levello_ );
		ms1 = simde_mm_mullo_epi16( ms1, levelhi_ );
		ms1 = simde_mm_adds_epu16( ms1, ms1l );
		ms1 = simde_mm_srli_epi16( ms1, 2 );

		simde__m128i ms2l = ms2;
		ms2l = simde_mm_mulhi_epu16( ms2l, levello_ );
		ms2 = simde_mm_mullo_epi16( ms2, levelhi_ );
		ms2 = simde_mm_adds_epu16( ms2, ms2l );
		ms2 = simde_mm_srli_epi16( ms2, 2 );
		return simde_mm_packus_epi16( ms1, ms2 );
	}
};
struct sse2_ch_blur_add_mul_copy_functor {
	const simde__m128i zero_;
	const simde__m128i levello_;
	const simde__m128i levelhi_;
	const tjs_int level32_;
	inline sse2_ch_blur_add_mul_copy_functor( tjs_int level )
		: zero_(simde_mm_setzero_si128()), level32_(level)
		, levello_(simde_mm_set1_epi16(level&0xffff)), levelhi_(simde_mm_set1_epi16(level>>16)){}
	inline tjs_uint8 operator()( tjs_uint8 d, tjs_uint8 s ) const {
		tjs_int a = d+(s * level32_ >> 18);
		if(a>=255) a = 255;
		return (tjs_uint8)a;
	}
	inline tjs_uint32 operator()( tjs_uint32 d, tjs_uint32 s ) const {
		simde__m128i ms1 = simde_mm_cvtsi32_si128(s);
		ms1 = simde_mm_unpacklo_epi8( ms1, zero_ );
		simde__m128i md1 = simde_mm_cvtsi32_si128(d);
		md1 = simde_mm_unpacklo_epi8( md1, zero_ );
		simde__m128i ms1l = ms1;
		ms1l = simde_mm_mulhi_epu16( ms1l, levello_ );
		ms1 = simde_mm_mullo_epi16( ms1, levelhi_ );
		ms1 = simde_mm_adds_epu16( ms1, ms1l );
		ms1 = simde_mm_srli_epi16( ms1, 2 );
		md1 = simde_mm_adds_epu16( md1, ms1 );
		md1 = simde_mm_packus_epi16( md1, zero_ );
		return simde_mm_cvtsi128_si32( md1 );
	}
	inline simde__m128i operator()( simde__m128i md1, simde__m128i ms1 ) const {
		simde__m128i ms2 = ms1;
		simde__m128i md2 = md1;
		ms1 = simde_mm_unpacklo_epi8( ms1, zero_ );
		ms2 = simde_mm_unpackhi_epi8( ms2, zero_ );
		md1 = simde_mm_unpacklo_epi8( md1, zero_ );
		md2 = simde_mm_unpackhi_epi8( md2, zero_ );

		simde__m128i ms1l = ms1;
		ms1l = simde_mm_mulhi_epu16( ms1l, levello_ );
		ms1 = simde_mm_mullo_epi16( ms1, levelhi_ );
		ms1 = simde_mm_adds_epu16( ms1, ms1l );
		ms1 = simde_mm_srli_epi16( ms1, 2 );

		simde__m128i ms2l = ms2;
		ms2l = simde_mm_mulhi_epu16( ms2l, levello_ );
		ms2 = simde_mm_mullo_epi16( ms2, levelhi_ );
		ms2 = simde_mm_adds_epu16( ms2, ms2l );
		ms2 = simde_mm_srli_epi16( ms2, 2 );

		md1 = simde_mm_adds_epu16( md1, ms1 );
		md2 = simde_mm_adds_epu16( md2, ms2 );
		return simde_mm_packus_epi16( md1, md2 );
	}
};
struct sse2_ch_blur_mul_copy65_functor {
	const simde__m128i zero_;
	const simde__m128i levello_;
	const simde__m128i levelhi_;
	const simde__m128i m64_;
	const tjs_int level32_;
	inline sse2_ch_blur_mul_copy65_functor( tjs_int level )
		: zero_(simde_mm_setzero_si128()), m64_(simde_mm_set1_epi16(64)), level32_(level)
		, levello_(simde_mm_set1_epi16(level&0xffff)), levelhi_(simde_mm_set1_epi16(level>>16)) {}
	inline tjs_uint8 operator()( tjs_uint8 d, tjs_uint8 s ) const {
		tjs_int a = (s * level32_ >> 18);
		if(a>=64) a = 64;
		return (tjs_uint8)a;
	}
	inline tjs_uint32 operator()( tjs_uint32 d, tjs_uint32 s ) const {
		simde__m128i ms1 = simde_mm_cvtsi32_si128(s);
		ms1 = simde_mm_unpacklo_epi8( ms1, zero_ );
		simde__m128i ms1l = ms1;
		ms1l = simde_mm_mulhi_epu16( ms1l, levello_ );
		ms1 = simde_mm_mullo_epi16( ms1, levelhi_ );
		ms1 = simde_mm_adds_epu16( ms1, ms1l );
		ms1 = simde_mm_srli_epi16( ms1, 2 );

		simde__m128i ms1mask = simde_mm_cmplt_epi16( ms1, m64_ );	// d < 64 ? 0xffff : 0
		ms1 = simde_mm_and_si128(ms1, ms1mask);
		ms1mask = simde_mm_andnot_si128(ms1mask, m64_ );	// d >= 64 ? 64 : 0
		ms1 = simde_mm_or_si128( ms1, ms1mask );
		ms1 = simde_mm_packus_epi16( ms1, zero_ );
		return simde_mm_cvtsi128_si32( ms1 );
	}
	inline simde__m128i operator()( simde__m128i md, simde__m128i ms1 ) const {
		simde__m128i ms2 = ms1;
		ms1 = simde_mm_unpacklo_epi8( ms1, zero_ );
		ms2 = simde_mm_unpackhi_epi8( ms2, zero_ );
		simde__m128i ms1l = ms1;
		ms1l = simde_mm_mulhi_epu16( ms1l, levello_ );
		ms1 = simde_mm_mullo_epi16( ms1, levelhi_ );
		ms1 = simde_mm_adds_epu16( ms1, ms1l );
		ms1 = simde_mm_srli_epi16( ms1, 2 );

		simde__m128i ms2l = ms2;
		ms2l = simde_mm_mulhi_epu16( ms2l, levello_ );
		ms2 = simde_mm_mullo_epi16( ms2, levelhi_ );
		ms2 = simde_mm_adds_epu16( ms2, ms2l );
		ms2 = simde_mm_srli_epi16( ms2, 2 );

		simde__m128i ms1mask = simde_mm_cmplt_epi16( ms1, m64_ );	// d < 64 ? 0xffff : 0
		simde__m128i ms2mask = simde_mm_cmplt_epi16( ms2, m64_ );	// d < 64 ? 0xffff : 0
		ms1 = simde_mm_and_si128(ms1, ms1mask);
		ms2 = simde_mm_and_si128(ms2, ms2mask);
		ms1mask = simde_mm_andnot_si128(ms1mask, m64_ );	// d >= 64 ? 64 : 0
		ms2mask = simde_mm_andnot_si128(ms2mask, m64_ );	// d >= 64 ? 64 : 0
		ms1 = simde_mm_or_si128( ms1, ms1mask );
		ms2 = simde_mm_or_si128( ms2, ms2mask );
		return simde_mm_packus_epi16( ms1, ms2 );
	}
};
struct sse2_ch_blur_add_mul_copy65_functor {
	const simde__m128i zero_;
	const simde__m128i levello_;
	const simde__m128i levelhi_;
	const simde__m128i m64_;
	const tjs_int level32_;
	inline sse2_ch_blur_add_mul_copy65_functor( tjs_int level )
		: zero_(simde_mm_setzero_si128()), m64_(simde_mm_set1_epi16(64)), level32_(level)
		, levello_(simde_mm_set1_epi16(level&0xffff)), levelhi_(simde_mm_set1_epi16(level>>16)) {}
	inline tjs_uint8 operator()( tjs_uint8 d, tjs_uint8 s ) const {
		tjs_int a = d+(s * level32_ >> 18);
		if(a>=64) a = 64;
		return (tjs_uint8)a;
	}
	inline tjs_uint32 operator()( tjs_uint32 d, tjs_uint32 s ) const {
		simde__m128i ms1 = simde_mm_cvtsi32_si128(s);
		ms1 = simde_mm_unpacklo_epi8( ms1, zero_ );
		simde__m128i md1 = simde_mm_cvtsi32_si128(d);
		md1 = simde_mm_unpacklo_epi8( md1, zero_ );
		simde__m128i ms1l = ms1;
		ms1l = simde_mm_mulhi_epu16( ms1l, levello_ );
		ms1 = simde_mm_mullo_epi16( ms1, levelhi_ );
		ms1 = simde_mm_adds_epu16( ms1, ms1l );
		ms1 = simde_mm_srli_epi16( ms1, 2 );
		md1 = simde_mm_adds_epu16( md1, ms1 );
		simde__m128i md1mask = simde_mm_cmplt_epi16( md1, m64_ );	// d < 64 ? 0xffff : 0
		md1 = simde_mm_and_si128(md1, md1mask);
		md1mask = simde_mm_andnot_si128(md1mask, m64_ );	// d >= 64 ? 64 : 0
		md1 = simde_mm_or_si128( md1, md1mask );
		md1 = simde_mm_packus_epi16( md1, zero_ );
		return simde_mm_cvtsi128_si32( md1 );
	}
	inline simde__m128i operator()( simde__m128i md1, simde__m128i ms1 ) const {
		simde__m128i ms2 = ms1;
		simde__m128i md2 = md1;
		ms1 = simde_mm_unpacklo_epi8( ms1, zero_ );
		ms2 = simde_mm_unpackhi_epi8( ms2, zero_ );
		md1 = simde_mm_unpacklo_epi8( md1, zero_ );
		md2 = simde_mm_unpackhi_epi8( md2, zero_ );

		simde__m128i ms1l = ms1;
		ms1l = simde_mm_mulhi_epu16( ms1l, levello_ );
		ms1 = simde_mm_mullo_epi16( ms1, levelhi_ );
		ms1 = simde_mm_adds_epu16( ms1, ms1l );
		ms1 = simde_mm_srli_epi16( ms1, 2 );

		simde__m128i ms2l = ms2;
		ms2l = simde_mm_mulhi_epu16( ms2l, levello_ );
		ms2 = simde_mm_mullo_epi16( ms2, levelhi_ );
		ms2 = simde_mm_adds_epu16( ms2, ms2l );
		ms2 = simde_mm_srli_epi16( ms2, 2 );

		md1 = simde_mm_adds_epu16( md1, ms1 );
		md2 = simde_mm_adds_epu16( md2, ms2 );

		simde__m128i md1mask = simde_mm_cmplt_epi16( md1, m64_ );	// d < 64 ? 0xffff : 0
		simde__m128i md2mask = simde_mm_cmplt_epi16( md2, m64_ );	// d < 64 ? 0xffff : 0
		md1 = simde_mm_and_si128(md1, md1mask);
		md2 = simde_mm_and_si128(md2, md2mask);
		md1mask = simde_mm_andnot_si128(md1mask, m64_ );	// d >= 64 ? 64 : 0
		md2mask = simde_mm_andnot_si128(md2mask, m64_ );	// d >= 64 ? 64 : 0
		md1 = simde_mm_or_si128( md1, md1mask );
		md2 = simde_mm_or_si128( md2, md2mask );
		return simde_mm_packus_epi16( md1, md2 );
	}
};

template<typename functor>
static inline void blend_func_sse2( tjs_uint8 * __restrict dest, const tjs_uint8 * __restrict src, tjs_int len, const functor& func ) {
	tjs_uint32 rem = (len>>4)<<4;
	tjs_uint32 rem32 = (len>>2)<<2;
	tjs_uint8* limit = dest + rem;
	tjs_uint8* limit32 = dest + rem32;
	tjs_uint8* end = dest + len;
	while( dest < limit ) {
		simde__m128i md = simde_mm_loadu_si128( (simde__m128i const*)dest );
		simde__m128i ms = simde_mm_loadu_si128( (simde__m128i const*)src );
		simde_mm_storeu_si128( (simde__m128i*)dest, func( md, ms ) );
		dest+=16; src+=16;
	}
	while( dest < limit32 ) {
		tjs_uint32 d = *(tjs_uint32 const*)dest;
		tjs_uint32 s = *(tjs_uint32 const*)src;
		*(tjs_uint32*)dest = func( d, s );
		dest+=4; src+=4;
	}
	while( dest < end ) {
		*dest = func( *dest, *src );
		dest++; src++;
	}
}

void TVPChBlurAddMulCopy65_sse2_c( tjs_uint8 *dest, const tjs_uint8 *src, tjs_int len, tjs_int opa ) {
	sse2_ch_blur_add_mul_copy65_functor func(opa);
	blend_func_sse2( dest, src, len, func );
}
void TVPChBlurAddMulCopy_sse2_c( tjs_uint8 *dest, const tjs_uint8 *src, tjs_int len, tjs_int opa ) {
	sse2_ch_blur_add_mul_copy_functor func(opa);
	blend_func_sse2( dest, src, len, func );
}
void TVPChBlurMulCopy65_sse2_c( tjs_uint8 *dest, const tjs_uint8 *src, tjs_int len, tjs_int opa ) {
	sse2_ch_blur_mul_copy65_functor func(opa);
	blend_func_sse2( dest, src, len, func );
}
void TVPChBlurMulCopy_sse2_c( tjs_uint8 *dest, const tjs_uint8 *src, tjs_int len, tjs_int opa ) {
	sse2_ch_blur_mul_copy_functor func(opa);
	blend_func_sse2( dest, src, len, func );
}

/* simple blur for character data */
/* shuld be more optimized */
template<typename functor>
static inline void sse2_ch_blur_copy_func( tjs_uint8 *dest, tjs_int destpitch, tjs_int destwidth, tjs_int destheight, const tjs_uint8 * src, tjs_int srcpitch, tjs_int srcwidth, tjs_int srcheight, tjs_int blurwidth, tjs_int blurlevel ) {
	/* clear destination */
	memset(dest, 0, destpitch*destheight);

	/* compute filter level */
	tjs_int lvsum = 0;
	for(tjs_int y = -blurwidth; y <= blurwidth; y++) {
		for(tjs_int x = -blurwidth; x <= blurwidth; x++) {
			tjs_int len = fast_int_hypot(x, y);
			if(len <= blurwidth)
				lvsum += (blurwidth - len +1);
		}
	}

	if(lvsum) lvsum = (1<<18)/lvsum; else lvsum=(1<<18);

	/* apply */
	for(tjs_int y = -blurwidth; y <= blurwidth; y++) {
		for(tjs_int x = -blurwidth; x <= blurwidth; x++) {
			tjs_int len = fast_int_hypot(x, y);
			if(len <= blurwidth) {
				len = blurwidth - len +1;
				len *= lvsum;
				len *= blurlevel;
				len >>= 8;
				if( len >= (1<<10) ) {
					functor func(len);
					for(tjs_int sy = 0; sy < srcheight; sy++) {
						blend_func_sse2( dest + (y + sy + blurwidth)*destpitch + x + blurwidth, src + sy * srcpitch, srcwidth, func );
					}
				}
			}
		}
	}
}
extern void TVP_ch_blur_copy65( tjs_uint8 *dest, tjs_int destpitch, tjs_int destwidth, tjs_int destheight, const tjs_uint8 * src, tjs_int srcpitch, tjs_int srcwidth, tjs_int srcheight, tjs_int blurwidth, tjs_int blurlevel );
extern void TVP_ch_blur_copy( tjs_uint8 *dest, tjs_int destpitch, tjs_int destwidth, tjs_int destheight, const tjs_uint8 * src, tjs_int srcpitch, tjs_int srcwidth, tjs_int srcheight, tjs_int blurwidth, tjs_int blurlevel );
void TVPChBlurCopy65_sse2_c( tjs_uint8 *dest, tjs_int destpitch, tjs_int destwidth, tjs_int destheight, const tjs_uint8 * src, tjs_int srcpitch, tjs_int srcwidth, tjs_int srcheight, tjs_int blurwidth, tjs_int blurlevel ) {
	tjs_uint64 vmax = ((1ULL<<18)*(blurwidth+1)*blurlevel)>>8;
	if( vmax >= (1<<24) ) {
		TVP_ch_blur_copy65( dest, destpitch, destwidth, destheight, src, srcpitch, srcwidth, srcheight, blurwidth, blurlevel );
	} else {
		sse2_ch_blur_copy_func<sse2_ch_blur_add_mul_copy65_functor>( dest, destpitch, destwidth, destheight, src, srcpitch, srcwidth, srcheight, blurwidth, blurlevel );
	}
}
void TVPChBlurCopy_sse2_c( tjs_uint8 *dest, tjs_int destpitch, tjs_int destwidth, tjs_int destheight, const tjs_uint8 * src, tjs_int srcpitch, tjs_int srcwidth, tjs_int srcheight, tjs_int blurwidth, tjs_int blurlevel ) {
	tjs_uint64 vmax = ((1ULL<<18)*(blurwidth+1)*blurlevel)>>8;
	if( vmax >= (1<<24) ) {
		TVP_ch_blur_copy( dest, destpitch, destwidth, destheight, src, srcpitch, srcwidth, srcheight, blurwidth, blurlevel );
	} else {
		sse2_ch_blur_copy_func<sse2_ch_blur_add_mul_copy_functor>( dest, destpitch, destwidth, destheight, src, srcpitch, srcwidth, srcheight, blurwidth, blurlevel );
	}
}
//TVPChBlurMulCopy65 = TVPChBlurMulCopy65_sse2_c;
//TVPChBlurAddMulCopy65 = TVPChBlurAddMulCopy65_sse2_c;
//TVPChBlurMulCopy = TVPChBlurMulCopy_sse2_c;
//TVPChBlurAddMulCopy = TVPChBlurAddMulCopy_sse2_c;
//TVPChBlurCopy65 = TVPChBlurCopy65_sse2_c;
//TVPChBlurCopy = TVPChBlurCopy_sse2_c;

struct sse2_bind_mask_to_main_functor {
	const simde__m128i zero_;
	const simde__m128i colormask_;
	inline sse2_bind_mask_to_main_functor() : zero_(simde_mm_setzero_si128()), colormask_(simde_mm_set1_epi32(0x00ffffff)) {}
	inline tjs_uint32 operator()( tjs_uint32 c, tjs_uint8 a ) const {
		return (c&0xffffff) | (a<<24);
	}
	inline simde__m128i operator()( simde__m128i md, tjs_uint32 s ) const {
		simde__m128i mo = simde_mm_cvtsi32_si128( s );
		mo = simde_mm_unpacklo_epi8( mo, zero_ );	// 0000 0a 0a 0a 0a
		mo = simde_mm_unpacklo_epi16( mo, zero_ );	// 000a 000a 000a 000a
		mo = simde_mm_slli_epi32( mo, 24 );
		md = simde_mm_and_si128( md, colormask_ );
		return simde_mm_or_si128( md, mo );
	}
};
void TVPBindMaskToMain_sse2_c(tjs_uint32 *main, const tjs_uint8 *mask, tjs_int len) {
	sse2_bind_mask_to_main_functor func;
	apply_color_map_func_sse2( main, mask, len , func );
}

