
#ifndef __BLEND_FUNCTOR_SSE2_H__
#define __BLEND_FUNCTOR_SSE2_H__


extern "C" {
extern unsigned char TVPOpacityOnOpacityTable[256*256];
extern unsigned char TVPNegativeMulTable[256*256];
};

// ソースのアルファを使う
template<typename blend_func>
struct sse2_variation : public blend_func {
	inline tjs_uint32 operator()( tjs_uint32 d, tjs_uint32 s ) const {
		tjs_uint32 a = (s>>24);
		return blend_func::operator()( d, s, a );
	}
	inline simde__m128i operator()( simde__m128i d, simde__m128i s ) const {
		simde__m128i a = s;
		a = simde_mm_srli_epi32( a, 24 );
		return blend_func::operator()( d, s, a );
	}
};

// ソースのアルファとopacity値を使う
template<typename blend_func>
struct sse2_variation_opa : public blend_func {
	const tjs_int32 opa_;
	const simde__m128i opa128_;
	inline sse2_variation_opa( tjs_int32 opa ) : opa_(opa), opa128_(simde_mm_set1_epi32(opa)) {}
	inline tjs_uint32 operator()( tjs_uint32 d, tjs_uint32 s ) const {
		//tjs_uint32 a = ((s>>24)*opa_) >> 8;
		tjs_uint32 a = (tjs_uint32)( ((tjs_uint64)s*(tjs_uint64)opa_) >> 32 );	// 最適化でうまくmulの上位ビット入るはず
		return blend_func::operator()( d, s, a );
	}
	inline simde__m128i operator()( simde__m128i d, simde__m128i s ) const {
		simde__m128i a = s;
		a = simde_mm_srli_epi32( a, 24 );
		a = simde_mm_mullo_epi16( a, opa128_ );
		a = simde_mm_srli_epi32( a, 8 );
		return blend_func::operator()( d, s, a );
	}
};

// ソースとデスティネーションのアルファを使う
//template<typename blend_func>
struct sse2_alpha_blend_d_functor {
	const simde__m128i colormask_;
	const simde__m128i zero_;
	inline sse2_alpha_blend_d_functor() : colormask_(simde_mm_set1_epi32(0x00ffffff)), zero_(simde_mm_setzero_si128()){}
	inline tjs_uint32 operator()( tjs_uint32 d, tjs_uint32 s ) const {
		tjs_uint32 addr = ((s >> 16) & 0xff00) + (d>>24);
		tjs_uint32 sopa = TVPOpacityOnOpacityTable[addr];

		simde__m128i ma = simde_mm_cvtsi32_si128( sopa );
		ma = simde_mm_shufflelo_epi16( ma, SIMDE_MM_SHUFFLE( 0, 0, 0, 0 )  );	// 00oo00oo00oo00oo
		simde__m128i ms = simde_mm_cvtsi32_si128( s );
		ms = simde_mm_unpacklo_epi8( ms, zero_ );// 00 ss 00 ss 00 ss 00 ss
		simde__m128i md = simde_mm_cvtsi32_si128( d );
		md = simde_mm_unpacklo_epi8( md, zero_ );// 00 dd 00 dd 00 dd 00 dd

		ms = simde_mm_sub_epi16( ms, md );		// ms -= md
		ms = simde_mm_mullo_epi16( ms, ma );		// ms *= ma
		md = simde_mm_slli_epi16( md, 8 );		// md <<= 8
		md = simde_mm_add_epi16( md, ms );		// md += ms
		md = simde_mm_srli_epi16( md, 8 );		// ms >>= 8
		md = simde_mm_packus_epi16( md, md );
		tjs_uint32 ret = simde_mm_cvtsi128_si32( md );

#if 0
		// TVPNegativeMulTable[addr] = (unsigned char)( 255 - (255-a)*(255-b)/ 255 );
		addr ^= 0xffff; // (a = 255-a, b = 255-b) : ^=xor
		tjs_uint32 tmp = addr;
		addr = (addr&0xff) * (tmp>>8);	// (255-a)*(255-b)
		addr = ~addr;	// result = 255-result : ビット反転
		addr = (addr&0xff00) << 16;	// /255 : 最下位ではなく、一つ上を選択することで/255と同等
#endif
		addr = TVPNegativeMulTable[addr] << 24;
		return (ret&0x00ffffff) | addr;
	}
	inline simde__m128i operator()( simde__m128i md1, simde__m128i ms1 ) const {
		//if( simde_mm_testc_si128( s, alphamask_ ) ) { return s; }
		//if( simde_mm_testz_si128( s, alphamask_ ) ) { return d; }
		simde__m128i sa = ms1;
		sa  = simde_mm_srli_epi32( sa, 24 );
		simde__m128i da = md1;
		da = simde_mm_srli_epi32( da, 24 );
		simde__m128i maddr = simde_mm_add_epi32( simde_mm_slli_epi32(sa, 8), da );
		simde__m128i dopa = maddr;
#if 0
		tjs_uint32 addr;
		addr = simde_mm_cvtsi128_si32( maddr );
		tjs_uint32 opa = TVPOpacityOnOpacityTable[addr];
		simde__m128i mopa = simde_mm_cvtsi32_si128( opa );

		maddr = simde_mm_shuffle_epi32( maddr, SIMDE_MM_SHUFFLE( 0, 3, 2, 1 ) );
		addr = simde_mm_cvtsi128_si32( maddr );
		opa = TVPOpacityOnOpacityTable[addr];
		simde__m128i mopa2 = simde_mm_cvtsi32_si128( opa );
		mopa = simde_mm_unpacklo_epi32( mopa, mopa2 );

		maddr = simde_mm_shuffle_epi32( maddr, SIMDE_MM_SHUFFLE( 0, 3, 2, 1 ) );
		addr = simde_mm_cvtsi128_si32( maddr );
		opa = TVPOpacityOnOpacityTable[addr];
		mopa2 = simde_mm_cvtsi32_si128( opa );

		maddr = simde_mm_shuffle_epi32( maddr, SIMDE_MM_SHUFFLE( 0, 3, 2, 1 ) );
		addr = simde_mm_cvtsi128_si32( maddr );
		opa = TVPOpacityOnOpacityTable[addr];
		simde__m128i mopa3 = simde_mm_cvtsi32_si128( opa );
		mopa2 = simde_mm_unpacklo_epi32( mopa2, mopa3 );

		mopa = simde_mm_unpacklo_epi64( mopa, mopa2 );
#else	// 以下のようにコンパイラ任せの方がいいかも
		simde__m128i ma1 = simde_mm_set_epi32(
			TVPOpacityOnOpacityTable[simde_mm_extract_epi16(maddr,6)],
			TVPOpacityOnOpacityTable[simde_mm_extract_epi16(maddr,4)],
			TVPOpacityOnOpacityTable[simde_mm_extract_epi16(maddr,2)],
			TVPOpacityOnOpacityTable[simde_mm_extract_epi16(maddr,0)]);
#endif
#if 0
		tjs_uint32 sopa = ma1.m128i_u32[0];
		tjs_uint32 d = md1.m128i_u32[0];
		tjs_uint32 s = ms1.m128i_u32[0];
		tjs_uint32 d1 = d & 0xff00ff;
		d1 = (d1 + (((s & 0xff00ff) - d1) * sopa >> 8)) & 0xff00ff;
		d &= 0xff00;
		s &= 0xff00;
		tjs_uint32 dest = d1 + ((d + ((s - d) * sopa >> 8)) & 0xff00);
#endif
		simde__m128i ms2 = ms1;
		simde__m128i md2 = md1;
		ms1 = simde_mm_unpacklo_epi8( ms1, zero_ );
		md1 = simde_mm_unpacklo_epi8( md1, zero_ );
		ma1 = simde_mm_packs_epi32( ma1, ma1 );		// 0 1 2 3 0 1 2 3
		ma1 = simde_mm_unpacklo_epi16( ma1, ma1 );	// 0 0 1 1 2 2 3 3
		simde__m128i ma2 = ma1;
		ma1 = simde_mm_unpacklo_epi16( ma1, ma1 );	 // 0 0 0 0 1 1 1 1
		ms1 = simde_mm_sub_epi16( ms1, md1 );	// s -= d
		ms1 = simde_mm_mullo_epi16( ms1, ma1 );	// s *= a
		md1 = simde_mm_slli_epi16( md1, 8 );		// d <<= 8
		md1 = simde_mm_add_epi16( md1, ms1 );	// d += s
		md1 = simde_mm_srli_epi16( md1, 8 );		// d >>= 8

		ms2 = simde_mm_unpackhi_epi8( ms2, zero_ );
		md2 = simde_mm_unpackhi_epi8( md2, zero_ );
		ma2 = simde_mm_unpackhi_epi16( ma2, ma2 );	// 2 2 2 2 3 3 3 3
		ms2 = simde_mm_sub_epi16( ms2, md2 );		// s -= d
		ms2 = simde_mm_mullo_epi16( ms2, ma2 );		// s *= a
		md2 = simde_mm_slli_epi16( md2, 8 );			// d <<= 8
		md2 = simde_mm_add_epi16( md2, ms2 );		// d += s
		md2 = simde_mm_srli_epi16( md2, 8 );			// d >>= 8
		md1 = simde_mm_packus_epi16( md1, md2 );

		const simde__m128i mask = simde_mm_set1_epi32( 0x0000ffff );
		dopa = simde_mm_xor_si128( dopa, mask );	// (a = 255-a, b = 255-b) : ^=xor
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

template<typename blend_func>
struct sse2_variation_hda : public blend_func {
	simde__m128i alphamask_;
	simde__m128i colormask_;
	inline sse2_variation_hda() {
		alphamask_ = simde_mm_set1_epi32( 0xFF000000 );
		colormask_ = simde_mm_set1_epi32( 0x00FFFFFF );
	}
	inline sse2_variation_hda( tjs_int32 opa ) : blend_func(opa) {
		alphamask_ = simde_mm_set1_epi32( 0xFF000000 );
		colormask_ = simde_mm_set1_epi32( 0x00FFFFFF );
	}
	inline tjs_uint32 operator()( tjs_uint32 d, tjs_uint32 s ) const {
		tjs_uint32 dstalpha = d&0xff000000;
		tjs_uint32 ret = blend_func::operator()( d, s );
		return (ret&0x00ffffff)|dstalpha;
	}
	inline simde__m128i operator()( simde__m128i d, simde__m128i s ) const {
		simde__m128i dstalpha = simde_mm_and_si128( d, alphamask_ );
		simde__m128i ret = blend_func::operator()( d, s );
		ret = simde_mm_and_si128( ret, colormask_ );
		return simde_mm_or_si128( ret, dstalpha );
	}
};

struct sse2_alpha_blend_func {
	const simde__m128i zero_;
	inline sse2_alpha_blend_func() : zero_( simde_mm_setzero_si128() ) {}
	inline tjs_uint32 one( simde__m128i md, simde__m128i ms, simde__m128i ma ) const {
		ms = simde_mm_unpacklo_epi8( ms, zero_ );// unpack 00 ss 00 ss 00 ss 00 ss
		ma = simde_mm_unpacklo_epi16( ma, ma );	// unpack 00 00 00 00 00 aa 00 aa
		md = simde_mm_unpacklo_epi8( md, zero_ );// unpack 00 dd 00 dd 00 dd 00 dd
		ma = simde_mm_unpacklo_epi32( ma, ma );	// unpack 00 aa 00 aa 00 aa 00 aa
		ms = simde_mm_sub_epi16( ms, md );		// ms -= md
		ms = simde_mm_mullo_epi16( ms, ma );		// ms *= ma
		ms = simde_mm_srli_epi16( ms, 8 );		// ms >>= 8
		md = simde_mm_add_epi8( md, ms );			// md += ms : d + ((s-d)*sopa)>>8
		md = simde_mm_packus_epi16( md, zero_ );	// pack
		return simde_mm_cvtsi128_si32( md );		// store
	}
	inline simde__m128i operator()( simde__m128i md1, simde__m128i ms1, simde__m128i ma1 ) const {
		simde__m128i ms2 = ms1;
		simde__m128i md2 = md1;

		ms1 = simde_mm_unpacklo_epi8( ms1, zero_ );
		md1 = simde_mm_unpacklo_epi8( md1, zero_ );
		ma1 = simde_mm_packs_epi32( ma1, ma1 );		// 0 1 2 3 0 1 2 3
		ma1 = simde_mm_unpacklo_epi16( ma1, ma1 );	// 0 0 1 1 2 2 3 3
		simde__m128i ma2 = ma1;
		ma1 = simde_mm_unpacklo_epi16( ma1, ma1 );	 // 0 0 0 0 1 1 1 1
		ms1 = simde_mm_sub_epi16( ms1, md1 );		// s -= d
		ms1 = simde_mm_mullo_epi16( ms1, ma1 );	// s *= a
		ms1 = simde_mm_srli_epi16( ms1, 8 );		// s >>= 8
		md1 = simde_mm_add_epi8( md1, ms1 );		// d += s

		ms2 = simde_mm_unpackhi_epi8( ms2, zero_ );
		md2 = simde_mm_unpackhi_epi8( md2, zero_ );
		ma2 = simde_mm_unpackhi_epi16( ma2, ma2 );	// 2 2 2 2 3 3 3 3
		ms2 = simde_mm_sub_epi16( ms2, md2 );		// s -= d
		ms2 = simde_mm_mullo_epi16( ms2, ma2 );		// s *= a
		ms2 = simde_mm_srli_epi16( ms2, 8 );			// s >>= 8
		md2 = simde_mm_add_epi8( md2, ms2 );		// d += s
		return simde_mm_packus_epi16( md1, md2 );
	}
};

struct sse2_alpha_blend : public sse2_alpha_blend_func {
	inline tjs_uint32 operator()( tjs_uint32 d, tjs_uint32 s, tjs_uint32 a ) const {
		simde__m128i ma = simde_mm_cvtsi32_si128( a );
		simde__m128i ms = simde_mm_cvtsi32_si128( s );
		simde__m128i md = simde_mm_cvtsi32_si128( d );
		return sse2_alpha_blend_func::one( md, ms, ma );
	}
	inline simde__m128i operator()( simde__m128i md1, simde__m128i ms1, simde__m128i ma1 ) const {
		return sse2_alpha_blend_func::operator()( md1, ms1, ma1 );
	}
};


struct sse2_variation_straight : public sse2_alpha_blend {
	const simde__m128i colormask_;
	const simde__m128i alphamask_;
	inline sse2_variation_straight() : colormask_(simde_mm_set1_epi32(0x00ffffff)), alphamask_(simde_mm_set1_epi32(0xff000000)) {}
	inline tjs_uint32 operator()( tjs_uint32 d, tjs_uint32 s ) const {
		if( s >= 0xff000000 ) {
			return s;
		}
		tjs_uint32 a = (s>>24);
		return sse2_alpha_blend::operator()( d, s, a );
	}
	inline simde__m128i operator()( simde__m128i d, simde__m128i s ) const {
		// SSE4.1
		/*
		if( simde_mm_testc_si128( s, alphamask_ ) ) {
			return s;	// totally opaque
		}
		if( simde_mm_testz_si128( s, alphamask_ ) ) {
			return d;	// totally transparent
		}
		*/
		simde__m128i a = s;
		a = simde_mm_srli_epi32( a, 24 );
		return sse2_alpha_blend::operator()( d, s, a );
	}
};

// もっともシンプルなコピー dst = src
struct sse2_const_copy_functor {
	inline tjs_uint32 operator()( tjs_uint32 d, tjs_uint32 s ) const { return s; }
	inline simde__m128i operator()( simde__m128i md1, simde__m128i ms1 ) const { return ms1; }
};
// 単純コピーだけど alpha をコピーしない(HDAと同じ)
struct sse2_color_copy_functor {
	const simde__m128i colormask_;
	const simde__m128i alphamask_;
	inline sse2_color_copy_functor() : colormask_(simde_mm_set1_epi32(0x00ffffff)), alphamask_(simde_mm_set1_epi32(0xff000000)) {}
	inline tjs_uint32 operator()( tjs_uint32 d, tjs_uint32 s ) const {
		return (d&0xff000000) + (s&0x00ffffff);
	}
	inline simde__m128i operator()( simde__m128i md1, simde__m128i ms1 ) const {
		ms1 = simde_mm_and_si128( ms1, colormask_ );
		md1 = simde_mm_and_si128( md1, alphamask_ );
		return simde_mm_or_si128( md1, ms1 );
	}
};
// alphaだけコピーする : color_copy の src destを反転しただけ
struct sse2_alpha_copy_functor : public sse2_color_copy_functor {
	inline tjs_uint32 operator()( tjs_uint32 d, tjs_uint32 s ) const {
		return sse2_color_copy_functor::operator()( s, d );
	}
	inline simde__m128i operator()( simde__m128i md1, simde__m128i ms1 ) const {
		return sse2_color_copy_functor::operator()( ms1, md1 );
	}
};
// このままコピーするがアルファを0xffで埋める dst = 0xff000000 | src
struct sse2_color_opaque_functor {
	const simde__m128i alphamask_;
	inline sse2_color_opaque_functor() : alphamask_(simde_mm_set1_epi32(0xff000000)) {}
	inline tjs_uint32 operator()( tjs_uint32 d, tjs_uint32 s ) const { return 0xff000000 | s; }
	inline simde__m128i operator()( simde__m128i md1, simde__m128i ms1 ) const { return simde_mm_or_si128( alphamask_, ms1 ); }
};
// 矩形版未実装
struct sse2_alpha_blend_a_functor {
	const simde__m128i mask_;
	const simde__m128i zero_;
	inline sse2_alpha_blend_a_functor() : zero_(simde_mm_setzero_si128()), mask_(simde_mm_set_epi32(0x0000ffff,0xffffffff,0x0000ffff,0xffffffff)) {}
	inline tjs_uint32 operator()( tjs_uint32 d, tjs_uint32 s ) const {
		simde__m128i ms = simde_mm_cvtsi32_si128( s );
		simde__m128i mo = ms;
		mo = simde_mm_srli_epi32( mo, 24 );			// >> 24
		simde__m128i msa = mo;
		msa = simde_mm_slli_epi64( msa, 48 );		// << 48
		mo = simde_mm_shufflelo_epi16( mo, SIMDE_MM_SHUFFLE( 0, 0, 0, 0 )  );	// 00oo00oo00oo00oo
		ms = simde_mm_unpacklo_epi8( ms, zero_ );	// 00Sa00Si00Si00Si
		ms = simde_mm_mullo_epi16( ms, mo );			// src * sopa
		ms = simde_mm_srli_epi16( ms, 8 );			// src * sopa >> 8
		ms = simde_mm_and_si128( ms, mask_ );		// drop alpha
		ms = simde_mm_or_si128( ms, msa );			// set original alpha

		simde__m128i md = simde_mm_cvtsi32_si128( d );
		md = simde_mm_unpacklo_epi8( md, zero_ );	// 00Da00Di00Di00Di
		simde__m128i md2 = md;
		md2 = simde_mm_mullo_epi16( md2, mo );	// d * sopa
		md2 = simde_mm_srli_epi16( md2, 8 );		// 00 SaDa 00 SaDi 00 SaDi 00 SaDi
		md = simde_mm_sub_epi16( md, md2 );		// d - d*sopa
		md = simde_mm_add_epi16( md, ms );		// (d-d*sopa) + s
		md = simde_mm_packus_epi16( md, zero_ );
		return simde_mm_cvtsi128_si32( md );
	}
	inline simde__m128i operator()( simde__m128i md, simde__m128i ms ) const {
		simde__m128i mo0 = ms;
		mo0 = simde_mm_srli_epi32( mo0, 24 );
		simde__m128i msa = mo0;
		msa = simde_mm_slli_epi32( msa, 24 );		// << 24
		mo0 = simde_mm_packs_epi32( mo0, mo0 );		// 0 1 2 3 0 1 2 3
		mo0 = simde_mm_unpacklo_epi16( mo0, mo0 );	// 0 0 1 1 2 2 3 3
		simde__m128i mo1 = mo0;
		mo1 = simde_mm_unpacklo_epi16( mo1, mo1 );	// 0 0 0 0 1 1 1 1 o[1]
		mo0 = simde_mm_unpackhi_epi16( mo0, mo0 );	// 2 2 2 2 3 3 3 3 o[0]
		
		simde__m128i ms1 = ms;
		ms = simde_mm_unpackhi_epi8( ms, zero_ );
		ms = simde_mm_mullo_epi16( ms, mo0 );		// src * sopa
		ms = simde_mm_srli_epi16( ms, 8 );			// src * sopa >> 8
		ms = simde_mm_and_si128( ms, mask_ );		// drop alpha
		simde__m128i msa1 = msa;
		msa = simde_mm_unpackhi_epi8( msa, zero_ );
		ms = simde_mm_or_si128( ms, msa );			// set original alpha
		
		ms1 = simde_mm_unpacklo_epi8( ms1, zero_ );
		ms1 = simde_mm_srli_epi16( ms1, 8 );			// src * sopa >> 8
		ms1 = simde_mm_and_si128( ms1, mask_ );		// drop alpha
		msa1 = simde_mm_unpacklo_epi8( msa1, zero_ );
		ms1 = simde_mm_or_si128( ms1, msa1 );		// set original alpha

		simde__m128i md1 = md;
		md = simde_mm_unpackhi_epi8( md, zero_ );// 00dd00dd00dd00dd d[0]
		simde__m128i md02 = md;
		md02 = simde_mm_mullo_epi16( md02, mo0 );	// d * sopa | d[0]
		md02 = simde_mm_srli_epi16( md02, 8 );	// 00 SaDa 00 SaDi 00 SaDi 00 SaDi | d[0]
		md = simde_mm_sub_epi16( md, md02 );		// d - d*sopa | d[0]
		md = simde_mm_add_epi16( md, ms );		// d - d*sopa + s | d[0]

		md1 = simde_mm_unpacklo_epi8( md1, zero_ );// 00dd00dd00dd00dd d[1]
		simde__m128i md12 = md1;
		md12 = simde_mm_mullo_epi16( md12, mo1 );// d * sopa | d[1]
		md12 = simde_mm_srli_epi16( md12, 8 );	// 00 SaDa 00 SaDi 00 SaDi 00 SaDi | d[1]
		md1 = simde_mm_sub_epi16( md1, md12 );	// d - d*sopa | d[1]
		md1 = simde_mm_add_epi16( md1, ms1 );	// d - d*sopa + s | d[1]

		return simde_mm_packus_epi16( md1, md );
	}
};

//typedef sse2_variation<sse2_alpha_blend>							sse2_alpha_blend_functor;
typedef sse2_variation_straight										sse2_alpha_blend_functor;
typedef sse2_variation_opa<sse2_alpha_blend>						sse2_alpha_blend_o_functor;
typedef sse2_variation_hda<sse2_variation<sse2_alpha_blend> >		sse2_alpha_blend_hda_functor;
typedef sse2_variation_hda<sse2_variation_opa<sse2_alpha_blend> >	sse2_alpha_blend_hda_o_functor;
//typedef sse2_variation_d<sse2_alpha_blend>							sse2_alpha_blend_d_functor;
// sse2_alpha_blend_d_functor
// sse2_alpha_blend_a_functor

struct sse2_premul_alpha_blend_functor {
	const simde__m128i zero_;
	inline sse2_premul_alpha_blend_functor() : zero_( simde_mm_setzero_si128() ) {}
	inline tjs_uint32 operator()( tjs_uint32 d, tjs_uint32 s ) const {
		simde__m128i ms = simde_mm_cvtsi32_si128( s );
		tjs_int32 sopa = s >> 24;
		simde__m128i mo = simde_mm_cvtsi32_si128( sopa );
		simde__m128i md = simde_mm_cvtsi32_si128( d );
		mo = simde_mm_unpacklo_epi16( mo, mo );	// 0000000000oo00oo
		mo = simde_mm_unpacklo_epi32( mo, mo );	// 00oo00oo00oo00oo
		//mo = simde_mm_shufflelo_epi16( mo, SIMDE_MM_SHUFFLE( 0, 0, 0, 0 )  );	// 0000000000000000 00oo00oo00oo00oo にできるか
		md = simde_mm_unpacklo_epi8( md, zero_ );// 00dd00dd00dd00dd
		simde__m128i md2 = md;
		md = simde_mm_mullo_epi16( md, mo );	// md * sopa
		md = simde_mm_srli_epi16( md, 8 );		// md >>= 8
		md2 = simde_mm_sub_epi16( md2, md );		// d - (d*sopa)>>8
		md2 = simde_mm_packus_epi16( md2, zero_ );	// pack
		md2 = simde_mm_adds_epu8( md2, ms );		// d - ((d*sopa)>>8) + src
		return simde_mm_cvtsi128_si32( md2 );
	}
	inline simde__m128i operator()( simde__m128i d, simde__m128i s ) const {
		simde__m128i ma1 = s;
		ma1 = simde_mm_srli_epi32( ma1, 24 );		// s >> 24
		ma1 = simde_mm_packs_epi32( ma1, ma1 );		// 0 1 2 3 0 1 2 3
		ma1 = simde_mm_unpacklo_epi16( ma1, ma1 );	// 0 0 1 1 2 2 3 3
		simde__m128i ma2 = ma1;
		ma1 = simde_mm_unpacklo_epi16( ma1, ma1 );	 // 0 0 0 0 1 1 1 1

		simde__m128i md2 = d;
		d = simde_mm_unpacklo_epi8( d, zero_ );
		simde__m128i md1 = d;
		d = simde_mm_mullo_epi16( d, ma1 );	// md * sopa
		d = simde_mm_srli_epi16( d, 8 );		// md >>= 8
		md1 = simde_mm_sub_epi16( md1, d );	// d - (d*sopa)>>8

		ma2 = simde_mm_unpackhi_epi16( ma2, ma2 );	// 2 2 2 2 3 3 3 3
		md2 = simde_mm_unpackhi_epi8( md2, zero_ );
		simde__m128i md2t = md2;
		md2 = simde_mm_mullo_epi16( md2, ma2 );	// md * sopa
		md2 = simde_mm_srli_epi16( md2, 8 );		// md >>= 8
		md2t = simde_mm_sub_epi16( md2t, md2 );	// d - (d*sopa)>>8

		md1 = simde_mm_packus_epi16( md1, md2t );
		return simde_mm_adds_epu8( md1, s );		// d - ((d*sopa)>>8) + src
	}
};
//--------------------------------------------------------------------
// di = di - di*a*opa + si*opa
//              ~~~~~Df ~~~~~~ Sf
//           ~~~~~~~~Ds
//      ~~~~~~~~~~~~~Dq
// additive alpha blend with opacity
struct sse2_premul_alpha_blend_o_functor {
	const simde__m128i zero_;
	const simde__m128i opa_;
	inline sse2_premul_alpha_blend_o_functor( tjs_int opa ) : zero_( simde_mm_setzero_si128() ), opa_(simde_mm_set1_epi16((short)opa)) {}
	inline tjs_uint32 operator()( tjs_uint32 d, tjs_uint32 s ) const {
		simde__m128i ms = simde_mm_cvtsi32_si128( s );
		simde__m128i md = simde_mm_cvtsi32_si128( d );
		ms = simde_mm_unpacklo_epi8( ms, zero_ );	// 00ss00ss00ss00ss
		md = simde_mm_unpacklo_epi8( md, zero_ );	// 00dd00dd00dd00dd
		ms = simde_mm_mullo_epi16( ms, opa_ );	// 00Sf00Sf00Sf00Sf s * opa
		simde__m128i md2 = md;
		ms = simde_mm_srli_epi16( ms, 8 );		// s >> 8
		simde__m128i ms2 = ms;
		ms2 = simde_mm_srli_epi64( ms2, 48 );		// s >> 48 | sopa
		ms2 = simde_mm_unpacklo_epi16( ms2, ms2 );
		ms2 = simde_mm_unpacklo_epi16( ms2, ms2 );// 00Df00Df00Df00Df

		md = simde_mm_mullo_epi16( md, ms2 );	// 00Ds00Ds00Ds00Ds
		md = simde_mm_srli_epi16( md, 8 );	// d >> 8
		md2 = simde_mm_sub_epi16( md2, md );	// 00Dq00Dq00Dq00Dq
		md2 = simde_mm_add_epi16( md2, ms );	// d + s
		md2 = simde_mm_packus_epi16( md2, zero_ );
		return simde_mm_cvtsi128_si32( md2 );
	}
	inline simde__m128i operator()( simde__m128i d, simde__m128i s ) const {
		simde__m128i ms = s;
		simde__m128i md = d;
		ms = simde_mm_unpackhi_epi8( ms, zero_ );
		md = simde_mm_unpackhi_epi8( md, zero_ );
		ms = simde_mm_mullo_epi16( ms, opa_ );	// 00Sf00Sf00Sf00Sf s * opa
		simde__m128i md2 = md;
		ms = simde_mm_srli_epi16( ms, 8 );		// s >> 8
		simde__m128i ma = ms;
		ma = simde_mm_srli_epi64( ma, 48 );		// s >> 48 : sopa 0 0 0 1 0 0 0 2
		ma = simde_mm_shuffle_epi32( ma, SIMDE_MM_SHUFFLE( 2, 2, 0, 0 ) );	// 0 1 0 1 0 2 0 2
		ma = simde_mm_packs_epi32( ma, ma );		// 1 1 2 2 1 1 2 2
		ma = simde_mm_unpacklo_epi32( ma, ma );	// 1 1 1 1 2 2 2 2
		// 00Df00Df00Df00Df

		md = simde_mm_mullo_epi16( md, ma );	// 00Ds00Ds00Ds00Ds
		md = simde_mm_srli_epi16( md, 8 );	// d >> 8
		md2 = simde_mm_sub_epi16( md2, md );	// 00Dq00Dq00Dq00Dq
		md2 = simde_mm_add_epi16( md2, ms );	// d + s

		s = simde_mm_unpacklo_epi8( s, zero_ );
		d = simde_mm_unpacklo_epi8( d, zero_ );
		s = simde_mm_mullo_epi16( s, opa_ );	// 00Sf00Sf00Sf00Sf s * opa
		simde__m128i md1 = d;
		s = simde_mm_srli_epi16( s, 8 );		// s >> 8
		ma = s;
		ma = simde_mm_srli_epi64( ma, 48 );		// s >> 48 | sopa
		ma = simde_mm_shuffle_epi32( ma, SIMDE_MM_SHUFFLE( 2, 2, 0, 0 ) );	// 0 1 0 1 0 2 0 2
		ma = simde_mm_packs_epi32( ma, ma );		// 1 1 2 2 1 1 2 2
		ma = simde_mm_unpacklo_epi32( ma, ma );	// 1 1 1 1 2 2 2 2
		// 00Df00Df00Df00Df

		d = simde_mm_mullo_epi16( d, ma );	// 00Ds00Ds00Ds00Ds
		d = simde_mm_srli_epi16( d, 8 );		// d >> 8
		md1 = simde_mm_sub_epi16( md1, d );	// 00Dq00Dq00Dq00Dq
		md1 = simde_mm_add_epi16( md1, s );	// d + s
		return simde_mm_packus_epi16( md1, md2 );
	}
};
/*
	Di = sat(Si, (1-Sa)*Di)
	Da = Sa + Da - SaDa
*/
// additive alpha blend holding destination alpha
struct sse2_premul_alpha_blend_hda_functor {
	const simde__m128i zero_;
	const simde__m128i alphamask_;
	const simde__m128i colormask_;
	inline sse2_premul_alpha_blend_hda_functor() : zero_(simde_mm_setzero_si128()),
		alphamask_(simde_mm_set_epi32(0x0000ffff,0xffffffff,0x0000ffff,0xffffffff)), colormask_(simde_mm_set1_epi32(0x00FFFFFF)) {}
//		alphamask_ = 0x0000ffffffffffffLL
	inline tjs_uint32 operator()( tjs_uint32 d, tjs_uint32 s ) const {
		simde__m128i ms = simde_mm_cvtsi32_si128( s );
		simde__m128i mo = simde_mm_cvtsi32_si128( s >> 24 );
		simde__m128i md = simde_mm_cvtsi32_si128( d );
		mo = simde_mm_unpacklo_epi16( mo, mo );		// 0000000000oo00oo
		ms = simde_mm_and_si128( ms, colormask_ );	// 0000000000ssssss
		mo = simde_mm_unpacklo_epi16( mo, mo );		// 00oo00oo00oo00oo
		md = simde_mm_unpacklo_epi8( md, zero_ );	// 00dd00dd00dd00dd
		simde__m128i md2 = md;
		md = simde_mm_mullo_epi16( md, mo );		// d * opa
		md = simde_mm_srli_epi16( md, 8 );		// d >> 8
		md = simde_mm_and_si128( md, alphamask_ );// d & 0x00ffffff
		md2 = simde_mm_sub_epi16( md2, md );		// d - d*opa
		md2 = simde_mm_packus_epi16( md2, zero_ );
		md2 = simde_mm_adds_epu8( md2, ms );		// d + src
		return simde_mm_cvtsi128_si32( md2 );
	}
	inline simde__m128i operator()( simde__m128i md, simde__m128i s ) const {
		simde__m128i mo0 = s;
		mo0 = simde_mm_srli_epi32( mo0, 24 );
		mo0 = simde_mm_packs_epi32( mo0, mo0 );		// 0 1 2 3 0 1 2 3
		mo0 = simde_mm_unpacklo_epi16( mo0, mo0 );	// 0 0 1 1 2 2 3 3
		simde__m128i mo1 = mo0;
		mo1 = simde_mm_unpacklo_epi16( mo1, mo1 );	// 0 0 0 0 1 1 1 1 o[1]
		mo0 = simde_mm_unpackhi_epi16( mo0, mo0 );	// 2 2 2 2 3 3 3 3 o[0]

		simde__m128i md0 = md;
		md = simde_mm_unpacklo_epi8( md, zero_ );	// 00dd00dd00dd00dd d[1]
		simde__m128i md12 = md;
		md = simde_mm_mullo_epi16( md, mo1 );		// d[1] * o[1]
		md = simde_mm_srli_epi16( md, 8 );			// d[1] >> 8
		md = simde_mm_and_si128( md, alphamask_);	// d[1] & 0x00ffffff
		md12 = simde_mm_sub_epi16( md12, md );		// d[1] - d[1]*opa

		md0 = simde_mm_unpackhi_epi8( md0, zero_ );	// 00dd00dd00dd00dd d[0]
		simde__m128i md02 = md0;
		md0 = simde_mm_mullo_epi16( md0, mo0 );		// d[0] * o[0]
		md0 = simde_mm_srli_epi16( md0, 8 );			// d[0] >> 8
		md0 = simde_mm_and_si128( md0, alphamask_); 	// d[0] & 0x00ffffff
		md02 = simde_mm_sub_epi16( md02, md0 );		// d[0] - d[0]*opa
		md02 = simde_mm_packus_epi16( md12, md02 );	// pack( d[1], d[0] )

		s = simde_mm_and_si128( s, colormask_);		// s & 0x00ffffff00ffffff
		return simde_mm_adds_epu8( md02, s );			// d + s
	}
};
// additive alpha blend on additive alpha
struct sse2_premul_alpha_blend_a_functor {
	const simde__m128i mask_;
	const simde__m128i zero_;
	inline sse2_premul_alpha_blend_a_functor() : zero_(simde_mm_setzero_si128()), mask_(simde_mm_set1_epi32(0x00ffffff)) {}
	inline tjs_uint32 operator()( tjs_uint32 d, tjs_uint32 s ) const {
		simde__m128i ms = simde_mm_cvtsi32_si128( s );
		simde__m128i mo = ms;
		mo = simde_mm_srli_epi64( mo, 24 );			// sopa
		ms = simde_mm_unpacklo_epi8( ms, zero_ );	// 00Sa00Si00Si00Si
		mo = simde_mm_unpacklo_epi16( mo, mo );		// 0000000000Sa00Sa
		simde__m128i md = simde_mm_cvtsi32_si128( d );
		mo = simde_mm_unpacklo_epi16( mo, mo );		// 00Sa00Sa00Sa00Sa
		md = simde_mm_unpacklo_epi8( md, zero_ );	// 00Da00Di00Di00Di
		simde__m128i md2 = md;
		md2 = simde_mm_mullo_epi16( md2, mo );	// d * sopa
		md2 = simde_mm_srli_epi16( md2, 8 );		// 00 SaDa 00 SaDi 00 SaDi 00 SaDi
		md = simde_mm_sub_epi16( md, md2 );		// d - d*sopa
		md = simde_mm_add_epi16( md, ms );		// (d-d*sopa) + s
		md = simde_mm_packus_epi16( md, zero_ );
		return simde_mm_cvtsi128_si32( md );
	}
	inline simde__m128i operator()( simde__m128i md, simde__m128i ms ) const {
		simde__m128i mo0 = ms;
		mo0 = simde_mm_srli_epi32( mo0, 24 );
		mo0 = simde_mm_packs_epi32( mo0, mo0 );		// 0 1 2 3 0 1 2 3
		mo0 = simde_mm_unpacklo_epi16( mo0, mo0 );	// 0 0 1 1 2 2 3 3
		simde__m128i mo1 = mo0;
		mo1 = simde_mm_unpacklo_epi16( mo1, mo1 );	// 0 0 0 0 1 1 1 1 o[1]
		mo0 = simde_mm_unpackhi_epi16( mo0, mo0 );	// 2 2 2 2 3 3 3 3 o[0]

		simde__m128i md1 = md;
		simde__m128i ms1 = ms;
		md = simde_mm_unpackhi_epi8( md, zero_ );// 00dd00dd00dd00dd d[0]
		simde__m128i md02 = md;
		ms = simde_mm_unpackhi_epi8( ms, zero_ );
		md02 = simde_mm_mullo_epi16( md02, mo0 );	// d * sopa | d[0]
		md02 = simde_mm_srli_epi16( md02, 8 );	// 00 SaDa 00 SaDi 00 SaDi 00 SaDi | d[0]
		md = simde_mm_sub_epi16( md, md02 );		// d - d*sopa | d[0]
		md = simde_mm_add_epi16( md, ms );		// d - d*sopa + s | d[0]

		md1 = simde_mm_unpacklo_epi8( md1, zero_ );// 00dd00dd00dd00dd d[1]
		simde__m128i md12 = md1;
		ms1 = simde_mm_unpacklo_epi8( ms1, zero_ );
		md12 = simde_mm_mullo_epi16( md12, mo1 );// d * sopa | d[1]
		md12 = simde_mm_srli_epi16( md12, 8 );	// 00 SaDa 00 SaDi 00 SaDi 00 SaDi | d[1]
		md1 = simde_mm_sub_epi16( md1, md12 );	// d - d*sopa | d[1]
		md1 = simde_mm_add_epi16( md1, ms1 );	// d - d*sopa + s | d[1]

		return simde_mm_packus_epi16( md1, md );
	}
};

// opacity値を使う
struct sse2_const_alpha_blend_functor {
	const simde__m128i opa_;
	const simde__m128i zero_;
	inline sse2_const_alpha_blend_functor( tjs_int32 opa ) : zero_(simde_mm_setzero_si128()), opa_(simde_mm_set1_epi16((short)opa)) {}
	inline tjs_uint32 operator()( tjs_uint32 d, tjs_uint32 s ) const {
		simde__m128i ms = simde_mm_cvtsi32_si128( s );
		simde__m128i md = simde_mm_cvtsi32_si128( d );
		ms = simde_mm_unpacklo_epi8( ms, zero_ );// unpack 00 ss 00 ss 00 ss 00 ss
		md = simde_mm_unpacklo_epi8( md, zero_ );// unpack 00 dd 00 dd 00 dd 00 dd
		ms = simde_mm_sub_epi16( ms, md );		// ms -= md
		ms = simde_mm_mullo_epi16( ms, opa_ );	// ms *= ma
		ms = simde_mm_srli_epi16( ms, 8 );		// ms >>= 8
		md = simde_mm_add_epi8( md, ms );		// md += ms : d + ((s-d)*sopa)>>8
		md = simde_mm_packus_epi16( md, zero_ );	// pack
		return simde_mm_cvtsi128_si32( md );		// store
	}
	inline simde__m128i operator()( simde__m128i md1, simde__m128i ms1 ) const {
		simde__m128i ms2 = ms1;
		simde__m128i md2 = md1;

		ms1 = simde_mm_unpacklo_epi8( ms1, zero_ );
		md1 = simde_mm_unpacklo_epi8( md1, zero_ );
		ms1 = simde_mm_sub_epi16( ms1, md1 );	// s -= d
		ms1 = simde_mm_mullo_epi16( ms1, opa_ );	// s *= a
		ms1 = simde_mm_srli_epi16( ms1, 8 );		// s >>= 8
		md1 = simde_mm_add_epi8( md1, ms1 );		// d += s

		ms2 = simde_mm_unpackhi_epi8( ms2, zero_ );
		md2 = simde_mm_unpackhi_epi8( md2, zero_ );
		ms2 = simde_mm_sub_epi16( ms2, md2 );		// s -= d
		ms2 = simde_mm_mullo_epi16( ms2, opa_ );		// s *= a
		ms2 = simde_mm_srli_epi16( ms2, 8 );			// s >>= 8
		md2 = simde_mm_add_epi8( md2, ms2 );		// d += s
		return simde_mm_packus_epi16( md1, md2 );
	}
	// 2pixel版でunpack済み(16bit単位)
	inline simde__m128i two( simde__m128i md1, simde__m128i ms1 ) const {
		ms1 = simde_mm_sub_epi16( ms1, md1 );	// s -= d
		ms1 = simde_mm_mullo_epi16( ms1, opa_ );	// s *= a
		ms1 = simde_mm_srli_epi16( ms1, 8 );		// s >>= 8
		return simde_mm_add_epi8( md1, ms1 );	// d += s
	}
};
typedef sse2_variation_hda<sse2_const_alpha_blend_functor>	sse2_const_alpha_blend_hda_functor;

struct sse2_const_alpha_blend_d_functor {
	const tjs_int32 opa32_;
	const simde__m128i colormask_;
	const simde__m128i zero_;
	const simde__m128i opa_;
	inline sse2_const_alpha_blend_d_functor( tjs_int32 opa )
		: colormask_(simde_mm_set1_epi32(0x00ffffff)), opa32_(opa<<8), zero_(simde_mm_setzero_si128()), opa_(simde_mm_set1_epi32(opa<<8)) {}
	inline tjs_uint32 operator()( tjs_uint32 d, tjs_uint32 s ) const {
		tjs_uint32 addr = opa32_ + (d>>24);
		tjs_uint32 sopa = TVPOpacityOnOpacityTable[addr];
		simde__m128i ma = simde_mm_cvtsi32_si128( sopa );
		ma = simde_mm_shufflelo_epi16( ma, SIMDE_MM_SHUFFLE( 0, 0, 0, 0 )  );	// 0000000000000000 00oo00oo00oo00oo
		simde__m128i ms = simde_mm_cvtsi32_si128( s );
		simde__m128i md = simde_mm_cvtsi32_si128( d );
		ms = simde_mm_unpacklo_epi8( ms, zero_ );// 00 ss 00 ss 00 ss 00 ss
		md = simde_mm_unpacklo_epi8( md, zero_ );// 00 dd 00 dd 00 dd 00 dd
		ms = simde_mm_sub_epi16( ms, md );		// ms -= md
		ms = simde_mm_mullo_epi16( ms, ma );		// ms *= ma
		ms = simde_mm_srli_epi16( ms, 8 );		// ms >>= 8
		md = simde_mm_add_epi8( md, ms );		// md += ms : d + ((s-d)*sopa)>>8
		md = simde_mm_packus_epi16( md, zero_ );	// pack
		tjs_uint32 ret = simde_mm_cvtsi128_si32( md );		// store
		addr = TVPNegativeMulTable[addr] << 24;
		return (ret&0x00ffffff) | addr;
	}
	inline simde__m128i operator()( simde__m128i md1, simde__m128i ms1 ) const {
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

		simde__m128i ms2 = ms1;
		simde__m128i md2 = md1;
		ms1 = simde_mm_unpacklo_epi8( ms1, zero_ );
		md1 = simde_mm_unpacklo_epi8( md1, zero_ );
		ms1 = simde_mm_sub_epi16( ms1, md1 );	// s -= d
		ms1 = simde_mm_mullo_epi16( ms1, ma1 );	// s *= a
		ms1 = simde_mm_srli_epi16( ms1, 8 );		// s >>= 8
		md1 = simde_mm_add_epi8( md1, ms1 );		// d += s
		
		ma2 = simde_mm_unpackhi_epi16( ma2, ma2 );	// 2 2 2 2 3 3 3 3
		ms2 = simde_mm_unpackhi_epi8( ms2, zero_ );
		md2 = simde_mm_unpackhi_epi8( md2, zero_ );
		ms2 = simde_mm_sub_epi16( ms2, md2 );		// s -= d
		ms2 = simde_mm_mullo_epi16( ms2, ma2 );		// s *= a
		ms2 = simde_mm_srli_epi16( ms2, 8 );			// s >>= 8
		md2 = simde_mm_add_epi8( md2, ms2 );			// d += s
		md1 = simde_mm_packus_epi16( md1, md2 );

		/*
		addr = ((s>>16)&0xff00) + (d>>24);
		addr ^= 0xffff;  (a = 255-a, b = 255-b)
		da = addr&0xff;
		sa = addr>>8;
		addr = sa * da;
		addr = ~addr;	result = 255-result
		addr &= 0xff00;
		addr <<= 16;
		*/
		// addr = b << 8 | a : b = opa | sa, a = da
		// ( 255 - (255-a)*(255-b)/ 255 ); 
		// 257 = 65536 / 255 
		// 257 * 255 = 65535
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
		//md1 = simde_mm_slli_epi32( md1, 8 ); // アルファを落とす colormask_ を使わなくていいが……
		//md1 = simde_mm_srli_epi32( md1, 8 );
		return simde_mm_or_si128( md1, dopa );
	}
};
struct sse2_const_alpha_blend_a_functor {
	const tjs_uint32 opa32_;
	const simde__m128i opa_;
	const simde__m128i colormask_;
	const struct sse2_premul_alpha_blend_a_functor blend_;
	inline sse2_const_alpha_blend_a_functor( tjs_int32 opa ) : colormask_(simde_mm_set1_epi32(0x00ffffff)), opa32_(opa<<24), opa_(simde_mm_set1_epi32(opa<<24)) {}
	inline tjs_uint32 operator()( tjs_uint32 d, tjs_uint32 s ) const {
		return blend_( d, (s&0x00ffffff)|opa32_ );
	}
	inline simde__m128i operator()( simde__m128i md1, simde__m128i ms1 ) const {
		ms1 = simde_mm_and_si128( ms1, colormask_ );
		ms1 = simde_mm_or_si128( ms1, opa_ );
		return blend_( md1, ms1 );
	}
};

//																	sse2_const_alpha_blend_functor;
typedef sse2_const_alpha_blend_functor								sse2_const_alpha_blend_sd_functor;
// tjs_uint32 sse2_const_alpha_blend_functor::operator()( tjs_uint32 d, tjs_uint32 s )
// tjs_uint32 sse2_const_alpha_blend_sd_functor::operator()( tjs_uint32 s1, tjs_uint32 s2 )
// と引数は異なるが、処理内容は同じ
// const_alpha_blend は、dest と src1 を共有しているようなもの dest = dest * src
// const_alpha_blend_sd は、dest = src1 * src2

// sse2_const_copy_functor = TVPCopy はない、memcpy になってる
// sse2_color_copy_functor = TVPCopyColor / TVPLinTransColorCopy
// sse2_alpha_copy_functor = TVPCopyMask
// sse2_color_opaque_functor = TVPCopyOpaqueImage
// sse2_const_alpha_blend_functor = TVPConstAlphaBlend
// sse2_const_alpha_blend_hda_functor = TVPConstAlphaBlend_HDA
// sse2_const_alpha_blend_d_functor = TVPConstAlphaBlend_a
// sse2_const_alpha_blend_a_functor = TVPConstAlphaBlend_a

//--------------------------------------------------------------------
// ここまでアルファブレンド
// ここから加算合成など、psでないブレンドはソースアルファが考慮されない
//--------------------------------------------------------------------
struct sse2_add_blend_functor {
	inline tjs_uint32 operator()( tjs_uint32 d, tjs_uint32 s ) const {
		simde__m128i md = simde_mm_cvtsi32_si128( d );
		simde__m128i ms = simde_mm_cvtsi32_si128( s );
		md = simde_mm_adds_epu8( md, ms );	// dest += src
		return simde_mm_cvtsi128_si32( md );
	}
	inline simde__m128i operator()( simde__m128i d, simde__m128i s ) const {
		return simde_mm_adds_epu8( d, s );	// dest += src
	}
};

struct sse2_add_blend_hda_functor {
	const simde__m128i mask_;
	inline sse2_add_blend_hda_functor() : mask_(simde_mm_set1_epi32(0x00ffffff)) {}
	inline tjs_uint32 operator()( tjs_uint32 d, tjs_uint32 s ) const {
		simde__m128i md = simde_mm_cvtsi32_si128( d );
		simde__m128i ms = simde_mm_cvtsi32_si128( s );
		ms = simde_mm_and_si128( ms, mask_ );// mask
		md = simde_mm_adds_epu8( md, ms );	// dest += src
		return simde_mm_cvtsi128_si32( md );
	}
	inline simde__m128i operator()( simde__m128i d, simde__m128i s ) const {
		s = simde_mm_and_si128( s, mask_ );	// 1 mask
		return simde_mm_adds_epu8( s, d );	// 1 src += dest
	}
};

struct sse2_add_blend_o_functor {
	const simde__m128i zero_;
	const simde__m128i opa_;
	inline sse2_add_blend_o_functor( tjs_int opa ) : zero_(simde_mm_setzero_si128())
		, opa_(simde_mm_srli_epi64(simde_mm_set1_epi16((short)opa),16)) {}
		// opa_ = 000000oo00oo00oo
	inline tjs_uint32 operator()( tjs_uint32 d, tjs_uint32 s ) const {
		simde__m128i md = simde_mm_cvtsi32_si128( d );
		md = simde_mm_unpacklo_epi8( md, zero_ );	// md = 00dd00dd00dd00dd
		simde__m128i ms = simde_mm_cvtsi32_si128( s );
		ms = simde_mm_unpacklo_epi8( ms, zero_ );	// ms = 00ss00ss00ss00ss
		ms = simde_mm_mullo_epi16( ms, opa_ );	// ms *= opa
		ms = simde_mm_srli_epi16( ms, 8 );		// ms >>=8
		md = simde_mm_add_epi16( md, ms );		// md += ms
		md = simde_mm_packus_epi16( md, zero_ );	// pack
		return simde_mm_cvtsi128_si32( md );
	}
	inline simde__m128i operator()( simde__m128i md1, simde__m128i ms1 ) const {
		simde__m128i md2 = md1;
		md1 = simde_mm_unpacklo_epi8( md1, zero_ );	// 00dd00dd00dd00dd
		md2 = simde_mm_unpackhi_epi8( md2, zero_ );	// 00dd00dd00dd00dd
		simde__m128i ms2 = ms1;
		ms1 = simde_mm_unpacklo_epi8( ms1, zero_ );	// 00ss00ss00ss00ss
		ms2 = simde_mm_unpackhi_epi8( ms2, zero_ );	// 00ss00ss00ss00ss
		ms1 = simde_mm_mullo_epi16( ms1, opa_ );		// s *= opa
		ms2 = simde_mm_mullo_epi16( ms2, opa_ );		// s *= opa
		ms1 = simde_mm_srli_epi16( ms1, 8 );			// s >>=8
		ms2 = simde_mm_srli_epi16( ms2, 8 );			// s >>=8
		md1 = simde_mm_add_epi16( md1, ms1 );		// d += s
		md2 = simde_mm_add_epi16( md2, ms2 );		// d += s
		return simde_mm_packus_epi16( md1, md2 );
	}
};
// sse2_add_blend_functor = TVPAddBlend_sse2_a
// sse2_add_blend_hda_functor = TVPAddBlend_HDA_sse2_a
// sse2_add_blend_o_functor = TVPAddBlend_o_sse2_a
typedef sse2_add_blend_o_functor			sse2_add_blend_hda_o_functor;	// = TVPAddBlend_HDA_o_sse2_a
//--------------------------------------------------------------------
struct sse2_sub_blend_functor {
	const simde__m128i xor_;
	inline sse2_sub_blend_functor() : xor_(simde_mm_set1_epi32(0xffffffff)){}
	inline tjs_uint32 operator()( tjs_uint32 d, tjs_uint32 s ) const {
		simde__m128i md = simde_mm_cvtsi32_si128( d );
		simde__m128i ms = simde_mm_cvtsi32_si128( s );
		ms = simde_mm_xor_si128( ms, xor_ );	// xor = 1.0 - src
		md = simde_mm_subs_epu8( md, ms );	// dest -= src
		return simde_mm_cvtsi128_si32( md );
	}
	inline simde__m128i operator()( simde__m128i d, simde__m128i s ) const {
		s = simde_mm_xor_si128( s, xor_ );	// xor = 1.0 - src
		return simde_mm_subs_epu8( d, s );	// dest -= src
	}
};
struct sse2_sub_blend_hda_functor {
	const simde__m128i mask_;
	const simde__m128i xor_;
	inline sse2_sub_blend_hda_functor() : mask_(simde_mm_set1_epi32(0x00ffffff)), xor_(simde_mm_set1_epi32(0xffffffff)) {}
	inline tjs_uint32 operator()( tjs_uint32 d, tjs_uint32 s ) const {
		simde__m128i md = simde_mm_cvtsi32_si128( d );
		simde__m128i ms = simde_mm_cvtsi32_si128( s );
		ms = simde_mm_xor_si128( ms, xor_ );	// xor = 1.0 - src
		ms = simde_mm_and_si128( ms, mask_ );// mask
		md = simde_mm_subs_epu8( md, ms );	// dest -= src
		return simde_mm_cvtsi128_si32( md );
	}
	inline simde__m128i operator()( simde__m128i d, simde__m128i s ) const {
		s = simde_mm_xor_si128( s, xor_ );	// xor = 1.0 - src
		s = simde_mm_and_si128( s, mask_ );	// mask
		return simde_mm_subs_epu8( d, s );	// dest -= src
	}
};
struct sse2_sub_blend_o_functor {
	const simde__m128i zero_;
	const simde__m128i xor_;
	const simde__m128i opa_;
	inline sse2_sub_blend_o_functor( tjs_int opa ) : zero_(simde_mm_setzero_si128()), xor_(simde_mm_set1_epi32(0xffffffff))
		, opa_(simde_mm_srli_epi64(simde_mm_set1_epi16((short)opa),16)){}
		// opa_ = 000000oo00oo00oo
	inline tjs_uint32 operator()( tjs_uint32 d, tjs_uint32 s ) const {
		simde__m128i md = simde_mm_cvtsi32_si128( d );
		md = simde_mm_unpacklo_epi8( md, zero_ );// md = 00dd00dd00dd00dd
		simde__m128i ms = simde_mm_cvtsi32_si128( s );
		ms = simde_mm_xor_si128( ms, xor_ );		// not
		ms = simde_mm_unpacklo_epi8( ms, zero_ );// ms = 00ss00ss00ss00ss
		ms = simde_mm_mullo_epi16( ms, opa_ );	// ms *= opa
		ms = simde_mm_srli_epi16( ms, 8 );		// ms >>=8
		md = simde_mm_sub_epi16( md, ms );		// md -= ms
		md = simde_mm_packus_epi16( md, zero_ );	// pack
		return simde_mm_cvtsi128_si32( md );
	}
	inline simde__m128i operator()( simde__m128i md1, simde__m128i ms1 ) const {
		simde__m128i md2 = md1;
		md1 = simde_mm_unpacklo_epi8( md1, zero_ );	// 00dd00dd00dd00dd
		md2 = simde_mm_unpackhi_epi8( md2, zero_ );	// 00dd00dd00dd00dd
		ms1 = simde_mm_xor_si128( ms1, xor_ );		// not
		simde__m128i ms2 = ms1;
		ms1 = simde_mm_unpacklo_epi8( ms1, zero_ );	// 00ss00ss00ss00ss
		ms2 = simde_mm_unpackhi_epi8( ms2, zero_ );	// 00ss00ss00ss00ss
		ms1 = simde_mm_mullo_epi16( ms1, opa_ );	// s *= opa
		ms2 = simde_mm_mullo_epi16( ms2, opa_ );	// s *= opa
		ms1 = simde_mm_srli_epi16( ms1, 8 );		// s >>=8
		ms2 = simde_mm_srli_epi16( ms2, 8 );		// s >>=8
		md1 = simde_mm_sub_epi16( md1, ms1 );	// d -= s
		md2 = simde_mm_sub_epi16( md2, ms2 );	// d -= s
		return simde_mm_packus_epi16( md1, md2 );	// pack
	}
};
typedef sse2_sub_blend_o_functor			sse2_sub_blend_hda_o_functor;
//--------------------------------------------------------------------
struct sse2_mul_blend_functor {
	const simde__m128i zero_;
	inline sse2_mul_blend_functor() : zero_(simde_mm_setzero_si128()){}
	inline tjs_uint32 operator()( tjs_uint32 d, tjs_uint32 s ) const {
		simde__m128i md = simde_mm_cvtsi32_si128( d );
		md = simde_mm_unpacklo_epi8( md, zero_ );
		simde__m128i ms = simde_mm_cvtsi32_si128( s );
		ms = simde_mm_unpacklo_epi8( ms, zero_ );
		md = simde_mm_mullo_epi16( md, ms );	// multiply
		md = simde_mm_srli_epi16( md, 8 );	// >>= 8
		md = simde_mm_packus_epi16( md, zero_ );	// pack
		return simde_mm_cvtsi128_si32( md );	// store
	}
	inline simde__m128i operator()( simde__m128i md1, simde__m128i ms1 ) const {
		simde__m128i md2 = md1;
		md1 = simde_mm_unpacklo_epi8( md1, zero_ );	// lo
		md2 = simde_mm_unpackhi_epi8( md2, zero_ );	// hi
		simde__m128i ms2 = ms1;
		ms1 = simde_mm_unpacklo_epi8( ms1, zero_ );	// lo
		ms2 = simde_mm_unpackhi_epi8( ms2, zero_ );	// hi
		md1 = simde_mm_mullo_epi16( md1, ms1 );	// lo multiply
		md2 = simde_mm_mullo_epi16( md2, ms2 );	// hi multiply
		md1 = simde_mm_srli_epi16( md1, 8 );	// lo shift
		md2 = simde_mm_srli_epi16( md2, 8 );	// hi shift
		return simde_mm_packus_epi16( md1, md2 );	// 1 pack
	}
};
struct sse2_mul_blend_hda_functor {
	const simde__m128i zero_;
	const simde__m128i mulmask_;
	const simde__m128i bitmask_;
	inline sse2_mul_blend_hda_functor() : zero_(simde_mm_setzero_si128())
		, mulmask_(simde_mm_set_epi32(0x0000ffff,0xffffffff,0x0000ffff,0xffffffff)), bitmask_(simde_mm_set_epi32(0x01000000,0x00000000,0x01000000,0x00000000)) {}
		//mulmask_ = 0x0000ffffffffffffLL
		//bitmask_ = 0x0100000000000000LL
	inline tjs_uint32 operator()( tjs_uint32 d, tjs_uint32 s ) const {
		simde__m128i md = simde_mm_cvtsi32_si128( d );
		md = simde_mm_unpacklo_epi8( md, zero_ );	// 00dd00dd00dd00dd
		simde__m128i ms = simde_mm_cvtsi32_si128( s );
		ms = simde_mm_unpacklo_epi8( ms, zero_ );	// 00ss00ss00ss00ss
		ms = simde_mm_and_si128( ms, mulmask_ );	// src & 0x00ffffff (mask)
		ms = simde_mm_or_si128( ms, bitmask_ );	// src | 0x100ffffff (bit) かけた時にdstアルファが消えないように
		md = simde_mm_mullo_epi16( md, ms );	// d * s
		md = simde_mm_srli_epi16( md, 8 );	// d * s >> 8
		md = simde_mm_packus_epi16( md, zero_ );	// pack
		return simde_mm_cvtsi128_si32( md );	// store
	}
	inline simde__m128i operator()( simde__m128i md1, simde__m128i ms1 ) const {
		simde__m128i md2 = md1;
		md1 = simde_mm_unpacklo_epi8( md1, zero_ );	// lo 00dd00dd00dd00dd
		md2 = simde_mm_unpackhi_epi8( md2, zero_ );	// hi 00dd00dd00dd00dd
		simde__m128i ms2 = ms1;
		ms1 = simde_mm_unpacklo_epi8( ms1, zero_ );	// lo 00ss00ss00ss00ss
		ms2 = simde_mm_unpackhi_epi8( ms2, zero_ );	// hi 00ss00ss00ss00ss
		ms1 = simde_mm_and_si128( ms1, mulmask_ );	// src & 0x00ffffff ( mask )
		ms1 = simde_mm_or_si128( ms1, bitmask_ );		// src | 0x100ffffff (bit)
		ms2 = simde_mm_and_si128( ms2, mulmask_ );	// src & 0x00ffffff ( mask )
		ms2 = simde_mm_or_si128( ms2, bitmask_ );		// src | 0x100ffffff (bit)
		md1 = simde_mm_mullo_epi16( md1, ms1 );		// d * s
		md2 = simde_mm_mullo_epi16( md2, ms2 );		// d * s
		md1 = simde_mm_srli_epi16( md1, 8 );	// d >>= 8
		md2 = simde_mm_srli_epi16( md2, 8 );	// d >>= 8
		return simde_mm_packus_epi16( md1, md2 );	// pack
	}
};
struct sse2_mul_blend_o_functor {
	const simde__m128i zero_;
	const simde__m128i opa_;
	const simde__m128i mask_;
	inline sse2_mul_blend_o_functor( tjs_int opa ) : zero_(simde_mm_setzero_si128()), opa_(simde_mm_set1_epi16((short)opa)), mask_(simde_mm_set1_epi32(0xffffffff)) {}
	inline tjs_uint32 operator()( tjs_uint32 d, tjs_uint32 s ) const {
		simde__m128i md = simde_mm_cvtsi32_si128( d );
		md = simde_mm_unpacklo_epi8( md, zero_ );	// 00dd00dd00dd00dd
		simde__m128i ms = simde_mm_cvtsi32_si128( s );
		ms = simde_mm_xor_si128( ms, mask_ );		// not src = (1.0-src)
		ms = simde_mm_unpacklo_epi8( ms, zero_ );	// 00ss00ss00ss00ss
		ms = simde_mm_mullo_epi16( ms, opa_ );	// s * opa
		ms = simde_mm_xor_si128( ms, mask_ );		// not src = (1.0-src)
		ms = simde_mm_srli_epi16( ms, 8 );		// src >>= 8
		md = simde_mm_mullo_epi16( md, ms );		// d * s
		md = simde_mm_srli_epi16( md, 8 );		// d >>= 8
		md = simde_mm_packus_epi16( md, zero_ );	// pack
		return simde_mm_cvtsi128_si32( md );		// store
	}
	inline simde__m128i operator()( simde__m128i md1, simde__m128i ms1 ) const {
		simde__m128i md2 = md1;
		md1 = simde_mm_unpacklo_epi8( md1, zero_ );	// lo 00dd00dd00dd00dd
		md2 = simde_mm_unpackhi_epi8( md2, zero_ );	// hi 00dd00dd00dd00dd
		ms1 = simde_mm_xor_si128( ms1, mask_ );		// not src
		simde__m128i ms2 = ms1;
		ms1 = simde_mm_unpacklo_epi8( ms1, zero_ );	// lo 00ss00ss00ss00ss
		ms2 = simde_mm_unpackhi_epi8( ms2, zero_ );	// hi 00ss00ss00ss00ss
		ms1 = simde_mm_mullo_epi16( ms1, opa_ );	// s * opa
		ms2 = simde_mm_mullo_epi16( ms2, opa_ );	// s * opa
		ms1 = simde_mm_xor_si128( ms1, mask_ );	// not src
		ms2 = simde_mm_xor_si128( ms2, mask_ );	// not src
		ms1 = simde_mm_srli_epi16( ms1, 8 );		// s >>= 8
		ms2 = simde_mm_srli_epi16( ms2, 8 );		// s >>= 8
		md1 = simde_mm_mullo_epi16( md1, ms1 );	// d * s
		md2 = simde_mm_mullo_epi16( md2, ms2 );	// d * s
		md1 = simde_mm_srli_epi16( md1, 8 );		// d >>= 8
		md2 = simde_mm_srli_epi16( md2, 8 );		// d >>= 8
		return simde_mm_packus_epi16( md1, md2 );	// pack
	}
};

struct sse2_mul_blend_hda_o_functor {
	const simde__m128i zero_;
	const simde__m128i opa_;
	const simde__m128i mask_;
	const simde__m128i mulmask_;
	const simde__m128i bitmask_;
	inline sse2_mul_blend_hda_o_functor( tjs_int opa ) : zero_(simde_mm_setzero_si128()), opa_(simde_mm_set1_epi16((short)opa))
		, mulmask_(simde_mm_set_epi32(0x0000ffff,0xffffffff,0x0000ffff,0xffffffff)), bitmask_(simde_mm_set_epi32(0x01000000,0x00000000,0x01000000,0x00000000))
		, mask_(simde_mm_set1_epi32(0xffffffff)) {}
		// mulmask_ = 0x0000ffffffffffffLL
		// bitmask_ = 0x0100000000000000LL
		// mask_ = 0xffffffffffffffffLL
	inline tjs_uint32 operator()( tjs_uint32 d, tjs_uint32 s ) const {
		simde__m128i md = simde_mm_cvtsi32_si128( d );
		md = simde_mm_unpacklo_epi8( md, zero_ );	// 00dd00dd00dd00dd
		simde__m128i ms = simde_mm_cvtsi32_si128( s );
		ms = simde_mm_xor_si128( ms, mask_ );		// not src
		ms = simde_mm_unpacklo_epi8( ms, zero_ );	// 00ss00ss00ss00ss
		ms = simde_mm_mullo_epi16( ms, opa_ );	// s * opa
		ms = simde_mm_xor_si128( ms, mask_ );		// not src
		ms = simde_mm_srli_epi16( ms, 8 );		// s >> 8
		ms = simde_mm_and_si128( ms, mulmask_ );	// s & 0x00ffffff
		ms = simde_mm_or_si128( ms, bitmask_ );	// s | 0x100000000
		md = simde_mm_mullo_epi16( md, ms );		// d * s
		md = simde_mm_srli_epi16( md, 8 );		// d >>= 8
		md = simde_mm_packus_epi16( md, zero_ );	// pack
		return simde_mm_cvtsi128_si32( md );
	}
	inline simde__m128i operator()( simde__m128i md1, simde__m128i ms1 ) const {
		simde__m128i md2 = md1;
		md1 = simde_mm_unpacklo_epi8( md1, zero_ );	// 00dd00dd00dd00dd
		md2 = simde_mm_unpackhi_epi8( md2, zero_ );	// 00dd00dd00dd00dd
		ms1 = simde_mm_xor_si128( ms1, mask_ );		// not src
		simde__m128i ms2 = ms1;
		ms1 = simde_mm_unpacklo_epi8( ms1, zero_ );	// 00ss00ss00ss00ss
		ms2 = simde_mm_unpackhi_epi8( ms2, zero_ );	// 00ss00ss00ss00ss
		ms1 = simde_mm_mullo_epi16( ms1, opa_ );	// s * opa
		ms2 = simde_mm_mullo_epi16( ms2, opa_ );	// s * opa
		ms1 = simde_mm_xor_si128( ms1, mask_ );	// not src
		ms2 = simde_mm_xor_si128( ms2, mask_ );	// not src
		ms1 = simde_mm_srli_epi16( ms1, 8 );		// s >>= 8
		ms2 = simde_mm_srli_epi16( ms2, 8 );		// s >>= 8
		ms1 = simde_mm_and_si128( ms1, mulmask_ );// s & 0x00ffffff
		ms1 = simde_mm_or_si128( ms1, bitmask_ );	// s | 0x100000000
		ms2 = simde_mm_and_si128( ms2, mulmask_ );// s & 0x00ffffff
		ms2 = simde_mm_or_si128( ms2, bitmask_ );	// s | 0x100000000
		md1 = simde_mm_mullo_epi16( md1, ms1 );	// d * s
		md2 = simde_mm_mullo_epi16( md2, ms2 );	// d * s
		md1 = simde_mm_srli_epi16( md1, 8 );		// d >>= 8
		md2 = simde_mm_srli_epi16( md2, 8 );		// d >>= 8
		return simde_mm_packus_epi16( md1, md2 );	// pack
	}
};
//--------------------------------------------------------------------
// lighten : normal/HDA
// The basic algorithm is:
// dest = b + sat(a-b)
// where sat(x) = x<0?0:x (zero lower saturation)
struct sse2_lighten_blend_functor {
	inline tjs_uint32 operator()( tjs_uint32 d, tjs_uint32 s ) const {
		simde__m128i ms = simde_mm_cvtsi32_si128( s );
		simde__m128i md = simde_mm_cvtsi32_si128( d );
		ms = simde_mm_subs_epu8( ms, md );
		ms = simde_mm_add_epi8( ms, md );
		return simde_mm_cvtsi128_si32( ms );
	}
	inline simde__m128i operator()( simde__m128i d, simde__m128i s ) const {
		s = simde_mm_subs_epu8( s, d );
		return simde_mm_add_epi8( s, d );
	}
};
struct sse2_lighten_blend_hda_functor {
	const simde__m128i mask_;
	inline sse2_lighten_blend_hda_functor() : mask_(simde_mm_set1_epi32(0x00ffffff)) {}
	inline tjs_uint32 operator()( tjs_uint32 d, tjs_uint32 s ) const {
		simde__m128i ms = simde_mm_cvtsi32_si128( s );
		simde__m128i md = simde_mm_cvtsi32_si128( d );
		ms = simde_mm_and_si128( ms, mask_ );
		ms = simde_mm_subs_epu8( ms, md );
		ms = simde_mm_add_epi8( ms, md );
		return simde_mm_cvtsi128_si32( ms );
	}
	inline simde__m128i operator()( simde__m128i d, simde__m128i s ) const {
		s = simde_mm_and_si128( s, mask_ );
		s = simde_mm_subs_epu8( s, d );
		return simde_mm_add_epi8( s, d );
	}
};
//--------------------------------------------------------------------
// darken : normal/HDA
// dest = b - sat(b-a)
// where sat(x) = x<0?0:x (zero lower saturation)
struct sse2_darken_blend_functor {
	inline tjs_uint32 operator()( tjs_uint32 d, tjs_uint32 s ) const {
		simde__m128i md = simde_mm_cvtsi32_si128( d );
		simde__m128i md2 = md;
		simde__m128i ms = simde_mm_cvtsi32_si128( s );
		md2 = simde_mm_subs_epu8( md2, ms );
		md = simde_mm_sub_epi8( md, md2 );
		return simde_mm_cvtsi128_si32( md );
	}
	inline simde__m128i operator()( simde__m128i d, simde__m128i s ) const {
		simde__m128i d2 = d;
		d2 = simde_mm_subs_epu8( d2, s );
		return simde_mm_sub_epi8( d, d2 );
	}
};
struct sse2_darken_blend_hda_functor {
	const simde__m128i mask_;
	inline sse2_darken_blend_hda_functor() : mask_(simde_mm_set1_epi32(0x00ffffff)) {}
	inline tjs_uint32 operator()( tjs_uint32 d, tjs_uint32 s ) const {
		simde__m128i md = simde_mm_cvtsi32_si128( d );
		//md = simde_mm_or_si128( md, mask_ ); // オリジナルにあるこれ何？
		simde__m128i md2 = md;
		simde__m128i ms = simde_mm_cvtsi32_si128( s );
		md2 = simde_mm_subs_epu8( md2, ms );
		md2 = simde_mm_and_si128( md2, mask_ );
		md = simde_mm_sub_epi8( md, md2 );
		return simde_mm_cvtsi128_si32( md );
	}
	inline simde__m128i operator()( simde__m128i d, simde__m128i s ) const {
		simde__m128i d2 = d;
		d2 = simde_mm_subs_epu8( d2, s );
		d2 = simde_mm_and_si128( d2, mask_ );
		return simde_mm_sub_epi8( d, d2 );
	}
};

//--------------------------------------------------------------------
// screen : normal/HDA/o/HDA o
struct sse2_screen_blend_functor {
	const simde__m128i mask_;
	const simde__m128i zero_;
	inline sse2_screen_blend_functor() : zero_(simde_mm_setzero_si128()), mask_(simde_mm_set1_epi32(0xffffffff)) {}
	inline tjs_uint32 operator()( tjs_uint32 d, tjs_uint32 s ) const {
		simde__m128i md = simde_mm_cvtsi32_si128( d );
		md = simde_mm_xor_si128( md, mask_ );	// not dest
		md = simde_mm_unpacklo_epi8( md, zero_ );
		simde__m128i ms = simde_mm_cvtsi32_si128( s );
		ms = simde_mm_xor_si128( ms, mask_ );	// not src
		ms = simde_mm_unpacklo_epi8( ms, zero_ );
		md = simde_mm_mullo_epi16( md, ms );	// d * s
		md = simde_mm_srli_epi16( md, 8 );	// (d * s) >> 8
		md = simde_mm_packus_epi16( md, zero_ );	// pack
		md = simde_mm_xor_si128( md, mask_ );	// not result
		return simde_mm_cvtsi128_si32( md );
	}
	inline simde__m128i operator()( simde__m128i md1, simde__m128i ms1 ) const {
		md1 = simde_mm_xor_si128( md1, mask_ );	// not dest
		simde__m128i md2 = md1;
		md1 = simde_mm_unpacklo_epi8( md1, zero_ );	// lo 00dd00dd00dd00dd
		md2 = simde_mm_unpackhi_epi8( md2, zero_ );	// hi 00dd00dd00dd00dd
		ms1 = simde_mm_xor_si128( ms1, mask_ );	// not src
		simde__m128i ms2 = ms1;
		ms1 = simde_mm_unpacklo_epi8( ms1, zero_ );	// lo 00ss00ss00ss00ss
		ms2 = simde_mm_unpackhi_epi8( ms2, zero_ );	// hi 00ss00ss00ss00ss
		md1 = simde_mm_mullo_epi16( md1, ms1 );	// d * s
		md2 = simde_mm_mullo_epi16( md2, ms2 );	// d * s
		md1 = simde_mm_srli_epi16( md1, 8 );	// d >> 8
		md2 = simde_mm_srli_epi16( md2, 8 );	// d >> 8
		md1 = simde_mm_packus_epi16( md1, md2 );	// pack
		return simde_mm_xor_si128( md1, mask_ );	// not result
	}
};
struct sse2_screen_blend_hda_functor {
	const simde__m128i mulmask_;
	const simde__m128i mul100_;
	const simde__m128i zero_;
	inline sse2_screen_blend_hda_functor() : zero_(simde_mm_setzero_si128())
		, mulmask_(simde_mm_set1_epi32(0x00ffffff)), mul100_(simde_mm_set_epi32(0x01000000,0x00000000,0x01000000,0x00000000)){}
	inline tjs_uint32 operator()( tjs_uint32 d, tjs_uint32 s ) const {
		simde__m128i md = simde_mm_cvtsi32_si128( d );
		md = simde_mm_xor_si128( md, mulmask_ );	// not dest
		md = simde_mm_unpacklo_epi8( md, zero_ );
		simde__m128i ms = simde_mm_cvtsi32_si128( s );
		ms = simde_mm_xor_si128( ms, mulmask_ );	// not src
		ms = simde_mm_and_si128( ms, mulmask_ );	// mask src オリジナルでは抜けてる？
		ms = simde_mm_unpacklo_epi8( ms, zero_ );
		ms = simde_mm_or_si128( ms, mul100_ );	// 1 or-1
		md = simde_mm_mullo_epi16( md, ms );
		md = simde_mm_srli_epi16( md, 8 );
		md = simde_mm_packus_epi16( md, zero_ );
		md = simde_mm_xor_si128( md, mulmask_ );	// not result
		return simde_mm_cvtsi128_si32( md );
	}
	inline simde__m128i operator()( simde__m128i md1, simde__m128i ms1 ) const {
		md1 = simde_mm_xor_si128( md1, mulmask_ );	// not dest
		simde__m128i md2 = md1;
		md1 = simde_mm_unpacklo_epi8( md1, zero_ );
		md2 = simde_mm_unpackhi_epi8( md2, zero_ );
		ms1 = simde_mm_xor_si128( ms1, mulmask_ );	// not src
		ms1 = simde_mm_and_si128( ms1, mulmask_ );	// mask src
		simde__m128i ms2 = ms1;
		ms1 = simde_mm_unpacklo_epi8( ms1, zero_ );
		ms2 = simde_mm_unpackhi_epi8( ms2, zero_ );
		ms1 = simde_mm_or_si128( ms1, mul100_ );	// or-1
		ms2 = simde_mm_or_si128( ms2, mul100_ );	// or-1
		md1 = simde_mm_mullo_epi16( md1, ms1 );
		md2 = simde_mm_mullo_epi16( md2, ms2 );
		md1 = simde_mm_srli_epi16( md1, 8 );
		md2 = simde_mm_srli_epi16( md2, 8 );
		md1 = simde_mm_packus_epi16( md1, md2 );
		return simde_mm_xor_si128( md1, mulmask_ );	// not result
	}
};
struct sse2_screen_blend_o_functor {
	const simde__m128i mask_;
	const simde__m128i zero_;
	const simde__m128i opa_;
	inline sse2_screen_blend_o_functor( tjs_int opa ) : zero_(simde_mm_setzero_si128()), opa_(simde_mm_set1_epi16((short)opa)), mask_(simde_mm_set1_epi32(0xffffffff)) {}
	inline tjs_uint32 operator()( tjs_uint32 d, tjs_uint32 s ) const {
		simde__m128i md = simde_mm_cvtsi32_si128( d );
		md = simde_mm_xor_si128( md, mask_ );		// not dest
		md = simde_mm_unpacklo_epi8( md, zero_ );
		simde__m128i ms = simde_mm_cvtsi32_si128( s );
		ms = simde_mm_unpacklo_epi8( ms, zero_ );
		ms = simde_mm_mullo_epi16( ms, opa_ );// opa multiply
		ms = simde_mm_xor_si128( ms, mask_ );	// not src
		ms = simde_mm_srli_epi16( ms, 8 );	// opa shift
		md = simde_mm_mullo_epi16( md, ms );	// multiply
		md = simde_mm_srli_epi16( md, 8 );	// shift
		md = simde_mm_packus_epi16( md, zero_ );// pack
		md = simde_mm_xor_si128( md, mask_ );	// not result
		return simde_mm_cvtsi128_si32( md );
	}
	inline simde__m128i operator()( simde__m128i md1, simde__m128i ms1 ) const {
		md1 = simde_mm_xor_si128( md1, mask_ );	// not dest
		simde__m128i md2 = md1;
		md1 = simde_mm_unpacklo_epi8( md1, zero_ );
		md2 = simde_mm_unpackhi_epi8( md2, zero_ );
		simde__m128i ms2 = ms1;
		ms1 = simde_mm_unpacklo_epi8( ms1, zero_ );
		ms2 = simde_mm_unpackhi_epi8( ms2, zero_ );
		ms1 = simde_mm_mullo_epi16( ms1, opa_ );	// opa multiply
		ms2 = simde_mm_mullo_epi16( ms2, opa_ );	// opa multiply
		ms1 = simde_mm_xor_si128( ms1, mask_ );	// not src
		ms2 = simde_mm_xor_si128( ms2, mask_ );	// not src
		ms1 = simde_mm_srli_epi16( ms1, 8 );		// opa shift
		ms2 = simde_mm_srli_epi16( ms2, 8 );		// opa shift
		md1 = simde_mm_mullo_epi16( md1, ms1 );	// multiply
		md2 = simde_mm_mullo_epi16( md2, ms2 );	// multiply
		md1 = simde_mm_srli_epi16( md1, 8 );		// shift
		md2 = simde_mm_srli_epi16( md2, 8 );		// shift
		md1 = simde_mm_packus_epi16( md1, md2 );	// pack
		return simde_mm_xor_si128( md1, mask_ );	// not result
	}
};
struct sse2_screen_blend_hda_o_functor {
	const simde__m128i zero_;
	const simde__m128i opa_;
	const simde__m128i mask_;
	const simde__m128i mulmask_;
	const simde__m128i alphamask_;
	inline sse2_screen_blend_hda_o_functor( tjs_int opa ) : zero_(simde_mm_setzero_si128()), opa_(simde_mm_set1_epi16((short)opa)), mask_(simde_mm_set1_epi32(0xffffffff))
		, mulmask_(simde_mm_set1_epi32(0x00ffffff)), alphamask_(simde_mm_set1_epi32(0xff000000)){}
	inline tjs_uint32 operator()( tjs_uint32 d, tjs_uint32 s ) const {
		simde__m128i md = simde_mm_cvtsi32_si128( d );
		md = simde_mm_xor_si128( md, mulmask_ );	// not dest
		md = simde_mm_unpacklo_epi8( md, zero_ );	// 00dd00dd00dd00dd
		simde__m128i ms = simde_mm_cvtsi32_si128( s );
		ms = simde_mm_unpacklo_epi8( ms, zero_ );	// 00ss00ss00ss00ss
		ms = simde_mm_mullo_epi16( ms, opa_ );	// s * opa
		ms = simde_mm_xor_si128( ms, mask_ );		// not src
		ms = simde_mm_srli_epi16( ms, 8 );		// s >> 8
		//ms = simde_mm_and_si128( ms, alphamask_ );// src mask
		//ms = simde_mm_or_si128( ms, mulmask_ );	// src alpha ???
		md = simde_mm_mullo_epi16( md, ms );		// d * s
		md = simde_mm_srli_epi16( md, 8 );		// d >> 8
		simde__m128i md2 = simde_mm_cvtsi32_si128( d );
		md = simde_mm_packus_epi16( md, zero_ );
		md2 = simde_mm_and_si128( md2, alphamask_ );	// dest mask
		md = simde_mm_xor_si128( md, mulmask_ );	// not result
		md = simde_mm_and_si128( md, mulmask_ );	// drop result alpha
		md = simde_mm_or_si128( md, md2 );	// restore dest alpha
		return simde_mm_cvtsi128_si32( md );
	}
	inline simde__m128i operator()( simde__m128i md1, simde__m128i ms1 ) const {
		simde__m128i mdr = md1;
		mdr = simde_mm_and_si128( mdr, alphamask_ );	// mask
		md1 = simde_mm_xor_si128( md1, mulmask_ );	// not dest
		simde__m128i md2 = md1;
		md1 = simde_mm_unpacklo_epi8( md1, zero_ );
		md2 = simde_mm_unpackhi_epi8( md2, zero_ );
		simde__m128i ms2 = ms1;
		ms1 = simde_mm_unpacklo_epi8( ms1, zero_ );
		ms2 = simde_mm_unpackhi_epi8( ms2, zero_ );
		ms1 = simde_mm_mullo_epi16( ms1, opa_ );	// opa multiply
		ms2 = simde_mm_mullo_epi16( ms2, opa_ );	// opa multiply
		ms1 = simde_mm_xor_si128( ms1, mask_ );	// not src
		ms2 = simde_mm_xor_si128( ms2, mask_ );	// not src
		ms1 = simde_mm_srli_epi16( ms1, 8 );		// opa shift
		ms2 = simde_mm_srli_epi16( ms2, 8 );		// opa shift
		md1 = simde_mm_mullo_epi16( md1, ms1 );	// multiply
		md2 = simde_mm_mullo_epi16( md2, ms2 );	// multiply
		md1 = simde_mm_srli_epi16( md1, 8 );		// shift
		md2 = simde_mm_srli_epi16( md2, 8 );		// shift
		md1 = simde_mm_packus_epi16( md1, md2 );	// pack
		md1 = simde_mm_xor_si128( md1, mulmask_ );	// not result
		md1 = simde_mm_and_si128( md1, mulmask_ );	// drop result alpha
		return simde_mm_or_si128( md1, mdr );			// restore dest alpha
	}
};


#endif // __BLEND_FUNCTOR_SSE2_H__
