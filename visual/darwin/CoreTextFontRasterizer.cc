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

CoreTextFontRasterizer::~CoreTextFontRasterizer() {}

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
  if (m_lastBitmap == bmp || !force)
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
  }

  // ensure font options
  m_fontFace->ensureOptions(opt);

  // why?
  m_lastBitmap = nullptr;
}

void CoreTextFontRasterizer::GetTextExtent(tjs_char ch, tjs_int &w,
                                           tjs_int &h) {
  // TODO:
  w = 0;
  h = 0;
}

tjs_int CoreTextFontRasterizer::GetAscentHeight() {
  // TODO:
  return 0;
}

tTVPCharacterData *
CoreTextFontRasterizer::GetBitmap(const tTVPFontAndCharacterData &font,
                                  tjs_int aofsx, tjs_int aofsy) {
  // TODO:

  // the actual rasterization goes here

  return nullptr;
}

void CoreTextFontRasterizer::GetGlyphDrawRect(const ttstr &    text,
                                              struct tTVPRect &area) {
  // TODO:
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
