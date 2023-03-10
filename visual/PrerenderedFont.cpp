
#include "PrerenderedFont.h"
#include "BinaryStream.h"
#include "MsgIntf.h"

extern tTJSHashTable<ttstr, tTVPPrerenderedFont *> TVPPrerenderedFonts;

tTVPPrerenderedFont::tTVPPrerenderedFont(const ttstr &storage) :
	LocalStorage(storage)
{
	RefCount = 1;
	Storage = storage;

	tTJSBinaryStream* stream = TVPCreateBinaryStreamForRead( storage, TJS_W("") );
	if( stream == NULL ) {
		TVPThrowExceptionMessage(TVPCannotOpenStorage, storage);
	}

	Image = NULL;
	try {
		FileLength = stream->GetSize();
		if( FileLength == 0 ) {
			TVPThrowExceptionMessage( TVPPrerenderedFontMappingFailed, (const tjs_char*)TVPFileSizeIsZero );
		}
		Image = new tjs_uint8[(tjs_uint)FileLength];
		if( Image == NULL ) {
			TVPThrowExceptionMessage( TVPPrerenderedFontMappingFailed, (const tjs_char*)TVPMemoryAllocationError );
		}
		tjs_uint readsize = stream->Read( const_cast<tjs_uint8*>(Image), (tjs_uint)FileLength );
		if( readsize != FileLength ) {
			TVPThrowExceptionMessage( TVPPrerenderedFontMappingFailed, (const tjs_char*)TVPFileReadError );
		}
		delete stream;
		stream = NULL;

		// check header
		if( memcmp("TVP pre-rendered font\x1a", Image, 22) ) {
			TVPThrowExceptionMessage(TVPPrerenderedFontMappingFailed, (const tjs_char*)TVPInvalidPrerenderedFontFile );
		}

		if( Image[23] != 2 ) {
			TVPThrowExceptionMessage(TVPPrerenderedFontMappingFailed, (const tjs_char*)TVPNot16BitUnicodeFontFile );
		}

		Version = Image[22];
		if( Version != 0 && Version != 1 ) {
			TVPThrowExceptionMessage(TVPPrerenderedFontMappingFailed, (const tjs_char*)TVPInvalidHeaderVersion );
		}

		// read index offset
		IndexCount = *(const tjs_uint32*)(Image + 24);
#if TJS_HOST_IS_BIG_ENDIAN
		IndexCount = __builtin_bswap32(IndexCount);
#endif
		tjs_uint32 ChIndexOffset = *(const tjs_uint32*)(Image + 28);
#if TJS_HOST_IS_BIG_ENDIAN
		ChIndexOffset = __builtin_bswap32(ChIndexOffset);
#endif
		tjs_char* ChIndexTmp = (tjs_char*)(Image + ChIndexOffset);
#if TJS_HOST_IS_BIG_ENDIAN
		// re-order input
		for(tjs_uint i = 0; i<IndexCount; i++)
		{
			tjs_char ch = ChIndexTmp[i];
			ChIndexTmp[i] = ((ch >> 8) & 0xff) + ((ch & 0xff) << 8);
		}
#endif
		ChIndex = ChIndexTmp;
		tjs_uint32 ImageOffset = *(const tjs_uint32*)(Image + 32);
#if TJS_HOST_IS_BIG_ENDIAN
		ImageOffset = __builtin_bswap32(ImageOffset);
#endif
		tTVPPrerenderedCharacterItem* IndexTmp = (tTVPPrerenderedCharacterItem*)(Image + ImageOffset);
#if TJS_HOST_IS_BIG_ENDIAN
		for (tjs_int i = ImageOffset; i < FileLength; i += sizeof(tTVPPrerenderedCharacterItem))
		{
			tTVPPrerenderedCharacterItem* IndexItem = (tTVPPrerenderedCharacterItem*)(Image + i);
			IndexItem->Offset = __builtin_bswap32(IndexItem->Offset);
			IndexItem->Width = __builtin_bswap16(IndexItem->Width);
			IndexItem->Height = __builtin_bswap16(IndexItem->Height);
			IndexItem->OriginX = __builtin_bswap16(IndexItem->OriginX);
			IndexItem->OriginY = __builtin_bswap16(IndexItem->OriginY);
			IndexItem->IncX = __builtin_bswap16(IndexItem->IncX);
			IndexItem->IncY = __builtin_bswap16(IndexItem->IncY);
			IndexItem->Inc = __builtin_bswap16(IndexItem->Inc);
			IndexItem->Reserved = __builtin_bswap16(IndexItem->Reserved);
		}
#endif
		Index = IndexTmp;
	} catch(...) {
		if( stream ) delete stream;
		if( Image ) delete[] Image;
		throw;
	}
	TVPPrerenderedFonts.Add(storage, this);
}
//---------------------------------------------------------------------------
tTVPPrerenderedFont::~tTVPPrerenderedFont()
{
	if( Image ) delete[] Image;

	TVPPrerenderedFonts.Delete(Storage);
}
//---------------------------------------------------------------------------
void tTVPPrerenderedFont::AddRef()
{
	RefCount ++;
}
//---------------------------------------------------------------------------
void tTVPPrerenderedFont::Release()
{
	if(RefCount == 1)
		delete this;
	else
		RefCount --;
}
//---------------------------------------------------------------------------
const tTVPPrerenderedCharacterItem *
		tTVPPrerenderedFont::Find(tjs_char ch)
{
	// search through ChIndex
	tjs_uint s = 0;
	tjs_uint e = IndexCount;
	const tjs_char *chindex = ChIndex;
	while(true)
	{
		tjs_int d = e-s;
		if(d <= 1)
		{
			if(chindex[s] == ch)
				return Index + s;
			else
				return NULL;
		}
		tjs_uint m = s + (d>>1);
		if(chindex[m] > ch) e = m; else s = m;
	}
}
//---------------------------------------------------------------------------
void tTVPPrerenderedFont::Retrieve(const tTVPPrerenderedCharacterItem * item,
	tjs_uint8 *buffer, tjs_int bufferpitch)
{
	// retrieve font data and store to buffer
	// bufferpitch must be larger then or equal to item->Width
	if(item->Width == 0 || item->Height == 0) return;

	const tjs_uint8 *ptr = item->Offset + Image;
	tjs_uint8 *dest = buffer;
	tjs_uint8 *destlim = dest + item->Width * item->Height;

	// expand compressed character bitmap data
	if(Version == 0)
	{
		// version 0 decompressor
		while(dest < destlim)
		{
			if(*ptr == 0x41) // running
			{
				ptr++;
				tjs_uint8 last = dest[-1];
				tjs_int len = *ptr;
				ptr++;
				while(len--) *(dest++) = (tjs_uint8)last;
			}
			else
			{
				*(dest++) = *(ptr++);
			}
		}
	}
	else if(Version >= 1)
	{
		// version 1+ decompressor
		while(dest < destlim)
		{
			if(*ptr >= 0x41) // running
			{
				tjs_int len = *ptr - 0x40;
				ptr++;
				tjs_uint8 last = dest[-1];
				while(len--) *(dest++) = (tjs_uint8)last;
			}
			else
			{
				*(dest++) = *(ptr++);
			}
		}
	}

	// expand to each pitch
	ptr = destlim - item->Width;
	dest = buffer + bufferpitch * item->Height - bufferpitch;
	while(buffer <= dest)
	{
		if(dest != ptr)
			memmove(dest, ptr, item->Width);
		dest -= bufferpitch;
		ptr -= item->Width;
	}
}
//---------------------------------------------------------------------------


