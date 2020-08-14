/**
 * @file      CoreTextFontRasterizer.h
 * @author    Hikaru Terazono (3c1u) <3c1u@vulpesgames.tokyo>
 * @brief     Core Text-based font rasterizer.
 * @date      2020-08-14
 *
 * @copyright Copyright (c) 2020 Hikaru Terazono. All rights reserved.
 *
 */

#if defined(__APPLE__)
#pragma once

#include "tjsCommHead.h"

#include "CharacterData.h"
#include "FontRasterizer.h"

class CoreTextFontRasterizer : public FontRasterizer {
public:
  CoreTextFontRasterizer();
  virtual ~CoreTextFontRasterizer();
  void    AddRef() override;
  void    Release() override;
  void    ApplyFont(class tTVPNativeBaseBitmap *bmp, bool force) override;
  void    ApplyFont(const struct tTVPFont &font) override;
  void    GetTextExtent(tjs_char ch, tjs_int &w, tjs_int &h) override;
  tjs_int GetAscentHeight() override;
  tTVPCharacterData *GetBitmap(const tTVPFontAndCharacterData &font,
                               tjs_int aofsx, tjs_int aofsy) override;
  void GetGlyphDrawRect(const ttstr &text, struct tTVPRect &area) override;
  bool AddFont(const ttstr &storage, std::vector<tjs_string> *faces) override;
  void GetFontList(std::vector<ttstr> &list, tjs_uint32 flags,
                   const struct tTVPFont &font) override;

private:
  size_t                      m_refCount;
  class tTVPNativeBaseBitmap *m_lastBitmap;
  class CoreTextFontFace *    m_fontFace;
};

#define KRKRZ_CORETEXT_SUPPORT

#endif // __APPLE__
