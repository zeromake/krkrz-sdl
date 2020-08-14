/**
 * @file      CoreTextFontStorage.h
 * @author    Hikaru Terazono (3c1u) <3c1u@vulpesgames.tokyo>
 * @brief     The storage for cached `CTFontDescriptor`s.
 * @date      2020-08-14
 *
 * @copyright Copyright (c) 2020 Hikaru Terazono. All rights reserved.
 *
 */

#pragma once

#include "tjsCommHead.h"

#include <CoreFoundation/CoreFoundation.h>
#include <CoreText/CoreText.h>

class CoreTextFontFace {
public:
  CoreTextFontFace(CTFontDescriptorRef descriptor, tjs_string const &faceName,
                   tjs_uint32 options);
  ~CoreTextFontFace();

  // no copy
  CoreTextFontFace(CoreTextFontFace const &) = delete;
  CoreTextFontFace &operator=(CoreTextFontFace const &) = delete;

  tjs_string const &getFaceName() const;
  void              ensureOptions(tjs_uint32 options);

private:
  CTFontDescriptorRef m_descriptor;
  tjs_string          m_faceName;
  tjs_uint32          m_options;
};

class CoreTextFontStorage {
public:
  static CoreTextFontStorage *getInstance();
  CoreTextFontStorage(CoreTextFontStorage const &) = delete;
  CoreTextFontStorage &operator=(CoreTextFontStorage const &) = delete;
  bool              loadFont(const ttstr &path, std::vector<tjs_string> *faces);
  void              populateFontList(std::vector<ttstr> &list, tjs_uint32 flags,
                                     const struct tTVPFont &_font);
  CoreTextFontFace *createFontFace(std::vector<tjs_string> const &faces,
                                   tjs_uint32                     flags);

private:
  CFDataRef getFontData(const ttstr &path);
  void      populateFontFaces(std::vector<tjs_string> *faces,
                              CFArrayRef               descriptors);
  bool      testFont(CTFontDescriptorRef descriptor, tjs_uint32 flags);
  CoreTextFontStorage();
  ~CoreTextFontStorage();

  CFMutableDictionaryRef m_fontMap; // CFMutableDictionaryRef<CFStringRef,
                                    // CFArrayRef<CTFontDescriptorRef>>
};
