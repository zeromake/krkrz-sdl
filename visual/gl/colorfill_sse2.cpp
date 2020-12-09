
#include "tjsCommHead.h"
#include "tvpgl.h"
#include "tvpgl_ia32_intf.h"
#include "simd_def_x86x64.h"

extern "C" {
extern unsigned char TVPOpacityOnOpacityTable[256*256];
extern unsigned char TVPNegativeMulTable[256*256];
};

//--------------------------------------------------------------------
void TVPFillARGB_sse2_c( tjs_uint32 *dest, tjs_int len, tjs_uint32 value ) {
	if( len <= 0 ) return;
	tjs_int count = (tjs_int)((size_t)dest & 0xF);
	if( count ) {
		count = (16 - count)>>2;
		count = count > len ? len : count;
		tjs_uint32* limit = dest + count;
		while( dest < limit ) {
			*dest = value;
			dest++;
		}
		len -= count;
	}
	tjs_uint32 rem = (len>>4)<<4;
	tjs_uint32* limit = dest + rem;
	{	// 16pixelごと
		const simde__m128i mvalue = simde_mm_set1_epi32( value );
		while( dest < limit ) {
			simde_mm_store_si128( (simde__m128i*)(dest+ 0), mvalue );
			simde_mm_store_si128( (simde__m128i*)(dest+ 4), mvalue );
			simde_mm_store_si128( (simde__m128i*)(dest+ 8), mvalue );
			simde_mm_store_si128( (simde__m128i*)(dest+12), mvalue );
			dest+=16;
		}
	}
	limit += (len-rem);
	while( dest < limit ) {
		*dest = value;
		dest++;
	}
}
//--------------------------------------------------------------------
void TVPFillARGB_NC_sse2_c( tjs_uint32 *dest, tjs_int len, tjs_uint32 value ) {
	if( len <= 0 ) return;
	tjs_int count = (tjs_int)((size_t)dest & 0xF);
	if( count ) {
		count = (16 - count)>>2;
		count = count > len ? len : count;
		tjs_uint32* limit = dest + count;
		while( dest < limit ) {
			*dest = value;
			dest++;
		}
		len -= count;
	}
	tjs_uint32 rem = (len>>4)<<4;
	tjs_uint32* limit = dest + rem;
	{	// 16pixelごと
		const simde__m128i mvalue = simde_mm_set1_epi32( value );
		while( dest < limit ) {
			simde_mm_stream_si128( (simde__m128i*)(dest+ 0), mvalue );
			simde_mm_stream_si128( (simde__m128i*)(dest+ 4), mvalue );
			simde_mm_stream_si128( (simde__m128i*)(dest+ 8), mvalue );
			simde_mm_stream_si128( (simde__m128i*)(dest+12), mvalue );
			dest+=16;
		}
	}
	limit += (len-rem);
	while( dest < limit ) {
		*dest = value;
		dest++;
	}
}

struct sse2_const_color_copy_functor {
	const tjs_uint32 color32_;
	const simde__m128i color_;
	const simde__m128i alphamask_;
	inline sse2_const_color_copy_functor( tjs_uint32 color )
	: color32_(color&0x00ffffff), color_(simde_mm_set1_epi32(color&0x00ffffff)), alphamask_(simde_mm_set1_epi32(0xff000000)) {}
	inline tjs_uint32 operator()( tjs_uint32 d ) const {
		return (d&0xff000000) + color32_;
	}
	inline simde__m128i operator()( simde__m128i md1 ) const {
		md1 = simde_mm_and_si128( md1, alphamask_ );
		return simde_mm_or_si128( md1, color_ );
	}
};
struct sse2_const_alpha_copy_functor {
	const tjs_uint32 alpha32_;
	const simde__m128i alpha_;
	const simde__m128i colormask_;
	inline sse2_const_alpha_copy_functor( tjs_uint32 mask )
	: alpha32_(mask<<24), alpha_(simde_mm_set1_epi32(mask<<24)), colormask_(simde_mm_set1_epi32(0x00ffffff)) {}
	inline tjs_uint32 operator()( tjs_uint32 d ) const {
		return (d&0x00ffffff) + alpha32_;
	}
	inline simde__m128i operator()( simde__m128i md1 ) const {
		md1 = simde_mm_and_si128( md1, colormask_ );
		return simde_mm_or_si128( md1, alpha_ );
	}
};
template<typename functor>
inline void sse2_const_color_copy_unroll( tjs_uint32 *dest, tjs_int len, const functor& func ) {
	if( len <= 0 ) return;
	tjs_int count = (tjs_int)((size_t)dest & 0xF);
	if( count ) {
		count = (16 - count)>>2;
		count = count > len ? len : count;
		tjs_uint32* limit = dest + count;
		while( dest < limit ) {
			*dest = func( *dest );
			dest++;
		}
		len -= count;
	}
	tjs_uint32 rem = (len>>4)<<4;
	tjs_uint32* limit = dest + rem;
	{	// 16pixelごと
		while( dest < limit ) {
			simde__m128i md0 = simde_mm_load_si128( (simde__m128i const*)(dest + 0) );
			simde__m128i md1 = simde_mm_load_si128( (simde__m128i const*)(dest + 4) );
			simde__m128i md2 = simde_mm_load_si128( (simde__m128i const*)(dest + 8) );
			simde__m128i md3 = simde_mm_load_si128( (simde__m128i const*)(dest + 12) );
			simde_mm_store_si128( (simde__m128i*)(dest+ 0), func( md0 ) );
			simde_mm_store_si128( (simde__m128i*)(dest+ 4), func( md1 ) );
			simde_mm_store_si128( (simde__m128i*)(dest+ 8), func( md2 ) );
			simde_mm_store_si128( (simde__m128i*)(dest+12), func( md3 ) );
			dest+=16;
		}
	}
	limit += (len-rem);
	while( dest < limit ) {
		*dest = func( *dest );
		dest++;
	}
}

void TVPFillColor_sse2_c(tjs_uint32 *dest, tjs_int len, tjs_uint32 color) {
	sse2_const_color_copy_functor func(color);
	sse2_const_color_copy_unroll( dest, len, func );
}
void TVPFillMask_sse2_c(tjs_uint32 *dest, tjs_int len, tjs_uint32 mask) {
	sse2_const_alpha_copy_functor func(mask);
	sse2_const_color_copy_unroll( dest, len, func );
}

struct sse2_const_alpha_fill_blend_functor {
	const simde__m128i invopa_;
	const simde__m128i zero_;
	simde__m128i color_;
	const simde__m128i alphamask_;
	const simde__m128i colormask_;
	inline sse2_const_alpha_fill_blend_functor( tjs_int32 opa, tjs_int32 color )
	: zero_(simde_mm_setzero_si128()), invopa_(simde_mm_set1_epi16((short)(255-opa))),
	  alphamask_(simde_mm_set1_epi32(0xff000000)), colormask_(simde_mm_set1_epi32(0x00ffffff)) {
		simde__m128i mopa = simde_mm_set1_epi16((short)opa);
		color_ = simde_mm_cvtsi32_si128( color );
		color_ = simde_mm_shuffle_epi32( color_, SIMDE_MM_SHUFFLE( 0, 0, 0, 0 )  );
		color_ = simde_mm_unpacklo_epi8( color_, zero_ );
		color_ = simde_mm_mullo_epi16( color_, mopa );
	}
	inline tjs_uint32 operator()( tjs_uint32 d ) const {
		simde__m128i md = simde_mm_cvtsi32_si128( d );
		simde__m128i ma = md;
		ma = simde_mm_and_si128( ma, alphamask_ );
		md = simde_mm_unpacklo_epi8( md, zero_ );// 00 dd 00 dd 00 dd 00 dd
		md = simde_mm_mullo_epi16( md, invopa_ );// md *= invopa
		md = simde_mm_adds_epu16( md, color_ );	// md += color
		md = simde_mm_srli_epi16( md, 8 );		// md >>= 8
		md = simde_mm_packus_epi16( md, zero_ );	// pack
		md = simde_mm_and_si128( md, colormask_ );// mask
		md = simde_mm_or_si128( md, ma );		// bind dest opa
		return simde_mm_cvtsi128_si32( md );		// store
	}
	inline simde__m128i operator()( simde__m128i md1 ) const {
		simde__m128i ma = md1;
		simde__m128i md2 = md1;
		ma = simde_mm_and_si128( ma, alphamask_ );

		md1 = simde_mm_unpacklo_epi8( md1, zero_ );	// 00 dd 00 dd 00 dd 00 dd
		md1 = simde_mm_mullo_epi16( md1, invopa_ );	// md *= invopa
		md1 = simde_mm_adds_epu16( md1, color_ );	// md += color
		md1 = simde_mm_srli_epi16( md1, 8 );			// md >>= 8

		md2 = simde_mm_unpackhi_epi8( md2, zero_ );	// 00 dd 00 dd 00 dd 00 dd
		md2 = simde_mm_mullo_epi16( md2, invopa_ );	// md *= invopa
		md2 = simde_mm_adds_epu16( md2, color_ );	// md += color
		md2 = simde_mm_srli_epi16( md2, 8 );			// md >>= 8

		md1 = simde_mm_packus_epi16( md1, md2 );
		md1 = simde_mm_and_si128( md1, colormask_ );// mask
		return simde_mm_or_si128( md1, ma );		// bind dest opa
	}
};

struct sse2_const_alpha_fill_blend_d_functor {
	const tjs_int32 opa32_;
	const simde__m128i colormask_;
	const simde__m128i zero_;
	const simde__m128i opa_;
	simde__m128i color_;
	inline sse2_const_alpha_fill_blend_d_functor( tjs_int32 opa, tjs_int32 color )
		: colormask_(simde_mm_set1_epi32(0x00ffffff)), opa32_(opa<<8), zero_(simde_mm_setzero_si128()), opa_(simde_mm_set1_epi32(opa<<8)) {
		color_ = simde_mm_cvtsi32_si128( color&0x00ffffff );
		color_ = simde_mm_shuffle_epi32( color_, SIMDE_MM_SHUFFLE( 0, 0, 0, 0 )  );
		color_ = simde_mm_unpacklo_epi8( color_, zero_ );
	}
	inline tjs_uint32 operator()( tjs_uint32 d ) const {
		tjs_uint32 addr = opa32_ + (d>>24);
		tjs_uint32 sopa = TVPOpacityOnOpacityTable[addr];
		simde__m128i ma = simde_mm_cvtsi32_si128( sopa );
		ma = simde_mm_shufflelo_epi16( ma, SIMDE_MM_SHUFFLE( 0, 0, 0, 0 )  );	// 0000000000000000 00oo00oo00oo00oo
		simde__m128i md = simde_mm_cvtsi32_si128( d );
		simde__m128i mc = color_;
		md = simde_mm_unpacklo_epi8( md, zero_ );// 00 dd 00 dd 00 dd 00 dd
		mc = simde_mm_sub_epi16( mc, md );		// mc -= md
		mc = simde_mm_mullo_epi16( mc, ma );		// ms *= ma
		md = simde_mm_slli_epi16( md, 8 );		// md <<= 8
		mc = simde_mm_add_epi8( mc, md );		// mc += md
		mc = simde_mm_srli_epi16( mc, 8 );		// mc >>= 8
		mc = simde_mm_packus_epi16( mc, zero_ );	// pack
		tjs_uint32 ret = simde_mm_cvtsi128_si32( mc );		// store
		addr = TVPNegativeMulTable[addr] << 24;
		return (ret&0x00ffffff) | addr;
	}
	inline simde__m128i operator()( simde__m128i md1 ) const {
		simde__m128i da = md1;
		da = simde_mm_srli_epi32( da, 24 );
		simde__m128i maddr = simde_mm_add_epi32( opa_, da );
		simde__m128i dopa = maddr;
		simde__m128i ma1 = simde_mm_set_epi32(
			TVPOpacityOnOpacityTable[simde_mm_extract_epi16(maddr,6)],
			TVPOpacityOnOpacityTable[simde_mm_extract_epi16(maddr,4)],
			TVPOpacityOnOpacityTable[simde_mm_extract_epi16(maddr,2)],
			TVPOpacityOnOpacityTable[simde_mm_extract_epi16(maddr,0)]);

		ma1 = simde_mm_packs_epi32( ma1, ma1 );		// 0 1 2 3 0 1 2 3
		ma1 = simde_mm_unpacklo_epi16( ma1, ma1 );	// 0 0 1 1 2 2 3 3
		simde__m128i ma2 = ma1;
		ma1 = simde_mm_unpacklo_epi16( ma1, ma1 );	 // 0 0 0 0 1 1 1 1

		simde__m128i md2 = md1;
		simde__m128i mc1 = color_;
		md1 = simde_mm_unpacklo_epi8( md1, zero_ );
		mc1 = simde_mm_sub_epi16( mc1, md1 );	// c -= d
		mc1 = simde_mm_mullo_epi16( mc1, ma1 );	// c *= a
		md1 = simde_mm_slli_epi16( md1, 8 );		// d <<= 8
		mc1 = simde_mm_add_epi16( mc1, md1 );		// c += d
		mc1 = simde_mm_srli_epi16( mc1, 8 );		// c >>= 8

		simde__m128i mc2 = color_;
		ma2 = simde_mm_unpackhi_epi16( ma2, ma2 );	// 2 2 2 2 3 3 3 3
		md2 = simde_mm_unpackhi_epi8( md2, zero_ );
		mc2 = simde_mm_sub_epi16( mc2, md2 );		// s -= d
		mc2 = simde_mm_mullo_epi16( mc2, ma2 );		// s *= a
		md2 = simde_mm_slli_epi16( md2, 8 );			// d <<= 8
		mc2 = simde_mm_add_epi16( mc2, md2 );			// d += s
		mc2 = simde_mm_srli_epi16( mc2, 8 );			// s >>= 8
		mc1 = simde_mm_packus_epi16( mc1, mc2 );

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

		mc1 = simde_mm_and_si128( mc1, colormask_ );
		return simde_mm_or_si128( mc1, dopa );
	}
};
// Di = Di - SaDi + Si
// Da = Da - SaDa + Sa
struct sse2_const_alpha_fill_blend_a_functor {
	simde__m128i mo_;
	simde__m128i mc_;
	const simde__m128i zero_;
	inline sse2_const_alpha_fill_blend_a_functor( tjs_int32 opa, tjs_int32 color ) : zero_(simde_mm_setzero_si128()) {
		opa += opa>>7;// adjust opacity
		mo_ = simde_mm_set1_epi16((short)(opa));

		simde__m128i msa = simde_mm_cvtsi32_si128( opa<<24 );
		msa = simde_mm_shuffle_epi32( msa, SIMDE_MM_SHUFFLE( 0, 0, 0, 0 )  );
		msa = simde_mm_unpacklo_epi8( msa, zero_ );	// 00 Sa 00 00 00 00 00 00

		mc_ = simde_mm_cvtsi32_si128( color&0x00ffffff );
		mc_ = simde_mm_shuffle_epi32( mc_, SIMDE_MM_SHUFFLE( 0, 0, 0, 0 )  );
		mc_ = simde_mm_unpacklo_epi8( mc_, zero_ );	// mc = 00 00 00 Si 00 Si 00 Si
		mc_ = simde_mm_mullo_epi16( mc_, mo_ );	// mc *= mo
		mc_ = simde_mm_srli_epi16( mc_, 8 );	// mc >>= 8
		mc_ = simde_mm_or_si128( mc_, msa );	// mc = 00 Sa 00 Si 00 Si 00 Si
	}
	inline tjs_uint32 operator()( tjs_uint32 d ) const {
		simde__m128i md = simde_mm_cvtsi32_si128( d );	// (DaDiDiDi)
		md = simde_mm_unpacklo_epi8( md, zero_ );	// 00 Da 00 Di 00 Di 00 Di
		simde__m128i md2 = md;
		md2 = simde_mm_mullo_epi16( md2, mo_ );		// d *= opa
		md2 = simde_mm_srli_epi16( md2, 8 );			// 00 SaDa 00 SaDi 00 SaDi 00 SaDi
		md = simde_mm_sub_epi16( md, md2 );			// d -= sadi
		md = simde_mm_add_epi16( md, mc_ );			// d += color
		md = simde_mm_packus_epi16( md, zero_ );
		return simde_mm_cvtsi128_si32( md );
	}
	inline simde__m128i operator()( simde__m128i md1 ) const {
		simde__m128i md2 = md1;

		md1 = simde_mm_unpacklo_epi8( md1, zero_ );	// 00 Da 00 Di 00 Di 00 Di
		simde__m128i mt = md1;
		mt = simde_mm_mullo_epi16( mt, mo_ );		// d *= opa
		mt = simde_mm_srli_epi16( mt, 8 );			// 00 SaDa 00 SaDi 00 SaDi 00 SaDi
		md1 = simde_mm_sub_epi16( md1, mt );			// d -= sadi
		md1 = simde_mm_add_epi16( md1, mc_ );		// d += color

		md2 = simde_mm_unpackhi_epi8( md2, zero_ );	// 00 Da 00 Di 00 Di 00 Di
		mt = md2;
		mt = simde_mm_mullo_epi16( mt, mo_ );		// d *= opa
		mt = simde_mm_srli_epi16( mt, 8 );			// 00 SaDa 00 SaDi 00 SaDi 00 SaDi
		md2 = simde_mm_sub_epi16( md2, mt );			// d -= sadi
		md2 = simde_mm_add_epi16( md2, mc_ );		// d += color

		return simde_mm_packus_epi16( md1, md2 );
	}
};
// TVPFillColor_sse2_c も以下のバリエーションとして実装可能だけど、↑のアンロール版使うことにする
template<typename functor>
inline void sse2_const_color_alpha_blend( tjs_uint32 *dest, tjs_int len, const functor& func ) {
	if( len <= 0 ) return;
	tjs_int count = (tjs_int)((size_t)dest & 0xF);
	if( count ) {
		count = (16 - count)>>2;
		count = count > len ? len : count;
		tjs_uint32* limit = dest + count;
		while( dest < limit ) {
			*dest = func( *dest );
			dest++;
		}
		len -= count;
	}
	tjs_uint32 rem = (len>>2)<<2;
	tjs_uint32* limit = dest + rem;
	{	// 4pixelごと
		while( dest < limit ) {
			simde__m128i md = simde_mm_load_si128( (simde__m128i const*)dest );
			simde_mm_store_si128( (simde__m128i*)dest, func( md ) );
			dest+=4;
		}
	}
	limit += (len-rem);
	while( dest < limit ) {
		*dest = func( *dest );
		dest++;
	}
}


void TVPConstColorAlphaBlend_sse2_c(tjs_uint32 *dest, tjs_int len, tjs_uint32 color, tjs_int opa) {
	sse2_const_alpha_fill_blend_functor func( opa, color );
	sse2_const_color_alpha_blend( dest, len, func );
}
void TVPConstColorAlphaBlend_d_sse2_c(tjs_uint32 *dest, tjs_int len, tjs_uint32 color, tjs_int opa) {
	sse2_const_alpha_fill_blend_d_functor func( opa, color );
	sse2_const_color_alpha_blend( dest, len, func );
}
void TVPConstColorAlphaBlend_a_sse2_c(tjs_uint32 *dest, tjs_int len, tjs_uint32 color, tjs_int opa) {
	sse2_const_alpha_fill_blend_a_functor func( opa, color );
	sse2_const_color_alpha_blend( dest, len, func );
}
struct sse2_alpha_color_mat_functor {
	const simde__m128i zero_;
	const simde__m128i color_;
	const simde__m128i alphamask_;
	inline sse2_alpha_color_mat_functor(  tjs_int32 color )
	: zero_(simde_mm_setzero_si128()), alphamask_(simde_mm_set1_epi32(0xff000000)), color_(simde_mm_unpacklo_epi8( simde_mm_set1_epi32(color), zero_)){}
	inline tjs_uint32 operator()( tjs_uint32 s ) const {
		simde__m128i md = color_;
		simde__m128i ms = simde_mm_cvtsi32_si128( s );
		simde__m128i ma = ms;
		ma = simde_mm_srli_epi32( ma, 24 );
		ma = simde_mm_shufflelo_epi16( ma, SIMDE_MM_SHUFFLE( 0, 0, 0, 0 )  );	// 下位のみ
		
		ms = simde_mm_unpacklo_epi8( ms, zero_ );
		ms = simde_mm_sub_epi16( ms, md );		// ms -= md
		ms = simde_mm_mullo_epi16( ms, ma );		// ms *= ma
		ms = simde_mm_srli_epi16( ms, 8 );		// ms >>= 8
		ms = simde_mm_add_epi8( ms, md );		// md += ms : d + ((s-d)*sopa)>>8
		ms = simde_mm_packus_epi16( ms, zero_ );	// pack
		return simde_mm_cvtsi128_si32( ms ) | 0xff000000;		// store
	}
	inline simde__m128i operator()( simde__m128i ms1 ) const {
		simde__m128i md = color_;
		simde__m128i ms2 = ms1;
		simde__m128i ma1 = ms1;
		ma1 = simde_mm_srli_epi32( ma1, 24 );
		ma1 = simde_mm_packs_epi32( ma1, ma1 );		// 0 1 2 3 0 1 2 3
		ma1 = simde_mm_unpacklo_epi16( ma1, ma1 );	// 0 0 1 1 2 2 3 3
		simde__m128i ma2 = ma1;
		ma1 = simde_mm_unpacklo_epi16( ma1, ma1 );	// 0 0 0 0 1 1 1 1
		ma2 = simde_mm_unpackhi_epi16( ma2, ma2 );	// 2 2 2 2 3 3 3 3
		
		ms1 = simde_mm_unpacklo_epi8( ms1, zero_ );
		ms1 = simde_mm_sub_epi16( ms1, md );		// s -= d
		ms1 = simde_mm_mullo_epi16( ms1, ma1 );	// s *= a
		ms1 = simde_mm_srli_epi16( ms1, 8 );		// s >>= 8
		ms1 = simde_mm_add_epi8( ms1, md );		// d += s
		
		ms2 = simde_mm_unpackhi_epi8( ms2, zero_ );
		ms2 = simde_mm_sub_epi16( ms2, md );		// s -= d
		ms2 = simde_mm_mullo_epi16( ms2, ma2 );	// s *= a
		ms2 = simde_mm_srli_epi16( ms2, 8 );		// s >>= 8
		ms2 = simde_mm_add_epi8( ms2, md );		// d += s
		ms1 = simde_mm_packus_epi16( ms1, ms2 );
		return simde_mm_or_si128( ms1, alphamask_ );
	}
};

void TVPAlphaColorMat_sse2_c(tjs_uint32 *dest, tjs_uint32 color, tjs_int len ) {
	sse2_alpha_color_mat_functor func( color );
	sse2_const_color_alpha_blend( dest, len, func );
}



