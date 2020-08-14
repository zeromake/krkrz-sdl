/**
 * @file      CoreTextFontRasterizer.cc
 * @author    Hikaru Terazono (3c1u) <3c1u@vulpesgames.tokyo>
 * @brief     Core Text-based font rasterizer.
 * @date      2020-08-14
 *
 * @copyright Copyright (c) 2020 Hikaru Terazono. All rights reserved.
 *
 */

/* FIXME: FUCK SURROGATE PAIRS! WE DON'T TREAT EM! */

#include "CoreTextFontRasterizer.h"
#include "CoreTextFontStorage.h"

#include <cmath>
#include <vector>

#include <CoreFoundation/CoreFoundation.h>
#include <CoreText/CoreText.h>

#include "DebugIntf.h"
#include "FontSystem.h"
#include "LayerBitmapIntf.h"
#include "MsgIntf.h"
#include "StringUtil.h"

// ---------------------------------------------
extern FontSystem *TVPFontSystem;
// ---------------------------------------------

CoreTextFontRasterizer::CoreTextFontRasterizer()
    : m_refCount(1), m_lastBitmap(nullptr), m_fontFace(nullptr) {}

CoreTextFontRasterizer::~CoreTextFontRasterizer() {
  if (m_fontFace) {
    delete m_fontFace;
    m_fontFace = nullptr;
  }
}

void CoreTextFontRasterizer::AddRef() { ++m_refCount; }

void CoreTextFontRasterizer::Release() {
  // invalidate the bitmap cache
  m_lastBitmap = nullptr;

  if (--m_refCount == 0) {
    delete this;
  }
}

void CoreTextFontRasterizer::ApplyFont(class tTVPNativeBaseBitmap *bmp,
                                       bool                        force) {
  // surpress redraw?
  if (m_lastBitmap == bmp && !force)
    return;

  ApplyFont(bmp->GetFont());
  m_lastBitmap = bmp;
}

void CoreTextFontRasterizer::ApplyFont(const struct tTVPFont &font) {
  // TODO: vertical text is not supported yet

  std::vector<tjs_string> faces;
  tjs_string              face = font.Face.AsStdString();

  if (face[0] == TJS_W(',')) {
    tjs_string stdname = TVPFontSystem->GetBeingFont(face);
    faces.push_back(stdname);
  } else {
    split(face, tjs_string(TJS_W(",")), faces);
    for (auto i = faces.begin(); i != faces.end();) {
      tjs_string &x = *i;
      x             = Trim(x);
      if (TVPFontSystem->FontExists(x) == false) {
        i = faces.erase(i);
      } else {
        i++;
      }
    }
    if (faces.empty()) {
      faces.push_back(tjs_string(TVPFontSystem->GetDefaultFontName()));
    }
  }

  // sanitize options
  tjs_uint32 opt = 0;
  opt |= (font.Flags & TVP_TF_ITALIC) ? TVP_TF_ITALIC : 0;
  opt |= (font.Flags & TVP_TF_BOLD) ? TVP_TF_BOLD : 0;
  opt |= (font.Flags & TVP_TF_UNDERLINE) ? TVP_TF_UNDERLINE : 0;
  opt |= (font.Flags & TVP_TF_STRIKEOUT) ? TVP_TF_STRIKEOUT : 0;
  opt |= (font.Flags & TVP_TF_FONTFILE) ? TVP_TF_FONTFILE : 0;

  // create CTFontCollection from faces

  auto storage = CoreTextFontStorage::getInstance();

  if (m_fontFace && (m_fontFace->getFaceName() != faces[0])) {
    // invalidate font face
    delete m_fontFace;
    m_fontFace = nullptr;
  }

  if (!m_fontFace && !(m_fontFace = storage->createFontFace(faces, opt))) {
    TVPThrowExceptionMessage(TJS_W("failed to create a font face: %1"),
                             faces[0]);
    // use system fallback
  }

  // ensure font options
  m_fontFace->createFontWithHeight(font.Height);
  m_fontFace->ensureOptions(opt);

  // why?
  m_lastBitmap = nullptr;
}

void CoreTextFontRasterizer::GetTextExtent(tjs_char ch, tjs_int &w,
                                           tjs_int &h) {
  auto    fontRef   = m_fontFace->getFont();
  UniChar character = static_cast<UniChar>(ch);
  CGGlyph glyph     = 0;

  if (!CTFontGetGlyphsForCharacters(fontRef, &character, &glyph, 1)) {
    w = h = this->m_fontFace->getHeight();
    return;
  }

  CGSize advances{0};
  CTFontGetAdvancesForGlyphs(fontRef, kCTFontOrientationDefault, &glyph,
                             &advances, 1);

  w = advances.width;
  h = advances.height;
}

tjs_int CoreTextFontRasterizer::GetAscentHeight() {
  return CTFontGetAscent(m_fontFace->getFont());
}

tTVPCharacterData *
CoreTextFontRasterizer::GetBitmap(const tTVPFontAndCharacterData &font,
                                  tjs_int aofsx, tjs_int aofsy) {
  // for better results
  constexpr auto kFontRenderingMargin = 5;

  // ensure font
  m_fontFace->createFontWithHeight(font.Font.Height);

  // get a glyph from a character code
  auto    fontRef   = m_fontFace->getFont();
  UniChar character = static_cast<UniChar>(font.Character);
  CGGlyph glyph     = 0;

  if (!CTFontGetGlyphsForCharacters(fontRef, &character, &glyph, 1)) {
    TVPThrowExceptionMessage(
        TJS_W("failed to create a glyph from a character: %1"), font.Character);
  }

  CGRect rect;
  CTFontGetBoundingRectsForGlyphs(fontRef, kCTFontOrientationDefault, &glyph,
                                  &rect, 1);
  int width  = rect.size.width + (kFontRenderingMargin * 2);
  int height = CTFontGetAscent(fontRef) + (kFontRenderingMargin * 2);

  // render a glyph into a buffer
  tjs_uint8 *buffer = new tjs_uint8[width * height];
  memset(buffer, 0, width * height);

  auto colorSpace = CGColorSpaceCreateDeviceGray();
  auto ctx =
      CGBitmapContextCreate(buffer, width, height, 8, width, colorSpace, 0);

  CGPoint offset =
      CGPointMake(-rect.origin.x + kFontRenderingMargin, kFontRenderingMargin);

  CGContextSetTextDrawingMode(ctx, kCGTextFill);
  CGContextSetGrayFillColor(ctx, 1.0, 1.0);
  CGContextSetGrayStrokeColor(ctx, 1.0, 0.0);
  CTFontDrawGlyphs(fontRef, &glyph, &offset, 1, ctx);

  CGContextRelease(ctx);
  CGColorSpaceRelease(colorSpace);

  tGlyphMetrics metrics{0};

  CGSize advances{0};
  CTFontGetAdvancesForGlyphs(fontRef, kCTFontOrientationDefault, &glyph,
                             &advances, 1);

  metrics.CellIncX = advances.width;
  metrics.CellIncY = advances.height;

  auto data =
      new tTVPCharacterData(buffer, width, std::round(-offset.x),
                            - kFontRenderingMargin,
                            width, height, metrics);

  data->Gray = 256;

  delete[] buffer;

  int cx = data->Metrics.CellIncX;
  int cy = data->Metrics.CellIncY;

  if (font.Font.Angle == 0) {
    data->Metrics.CellIncX = cx;
    data->Metrics.CellIncY = 0;
  } else if (font.Font.Angle == 2700) {
    data->Metrics.CellIncX = 0;
    data->Metrics.CellIncY = cx;
  } else {
    double angle           = font.Font.Angle * (M_PI / 1800);
    data->Metrics.CellIncX = static_cast<tjs_int>(std::cos(angle) * cx);
    data->Metrics.CellIncY = static_cast<tjs_int>(-std::sin(angle) * cx);
  }

  data->Antialiased = font.Antialiased;
  data->FullColored = false;
  data->Blured      = font.Blured;
  data->BlurWidth   = font.BlurWidth;
  data->BlurLevel   = font.BlurLevel;
  data->OriginX += aofsx;

  if (font.Blured)
    data->Blur();

  return data;
}

void CoreTextFontRasterizer::GetGlyphDrawRect(const ttstr &    text,
                                              struct tTVPRect &area) {
  area.left = area.top = area.right = area.bottom = 0;
  tjs_int  offsetx                                = 0;
  tjs_int  offsety                                = 0;
  tjs_uint len                                    = text.length();
  for (tjs_uint i = 0; i < len; i++) {
    tjs_char ch = text[i];
    tjs_int  ax, ay;
    tTVPRect rt(0, 0, 0, 0);
    bool     result = m_fontFace->getGlyphRectFromCharcode(rt, ch, ax, ay);
    if (!result) {
      m_fontFace->getGlyphRectFromCharcode(rt, '?', ax, ay);
    }
    if (result) {
      rt.add_offsets(offsetx, offsety);
      if (i != 0) {
        area.do_union(rt);
      } else {
        area = rt;
      }
    }
    offsetx += ax;
    offsety = 0;
  }
}

bool CoreTextFontRasterizer::AddFont(const ttstr &            storage,
                                     std::vector<tjs_string> *faces) {
  auto instance = CoreTextFontStorage::getInstance();
  return instance->loadFont(storage, faces);
}

void CoreTextFontRasterizer::GetFontList(std::vector<ttstr> &   list,
                                         tjs_uint32             flags,
                                         const struct tTVPFont &font) {
  auto instance = CoreTextFontStorage::getInstance();
  return instance->populateFontList(list, flags, font);
}
