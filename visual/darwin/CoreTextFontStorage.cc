/**
 * @file      CoreTextFontStorage.cc
 * @author    Hikaru Terazono (3c1u) <3c1u@vulpesgames.tokyo>
 * @brief     The storage for cached `CTFontDescriptor`s.
 * @date      2020-08-14
 *
 * @copyright Copyright (c) 2020 Hikaru Terazono. All rights reserved.
 *
 */

#include "CoreTextFontStorage.h"

#include "BinaryStream.h"
#include "DebugIntf.h"
#include "LayerBitmapIntf.h"
#include "MsgIntf.h"

CoreTextFontFace::CoreTextFontFace(CTFontDescriptorRef descriptor,
                                   tjs_string const &  faceName,
                                   tjs_uint32          options)
    : m_descriptor(descriptor), m_faceName(faceName), m_font(nullptr),
      m_height(12.0), m_options(options) {}

CoreTextFontFace::~CoreTextFontFace() { CFRelease(m_descriptor); }

tjs_string const &CoreTextFontFace::getFaceName() const { return m_faceName; }

tjs_uint32 CoreTextFontFace::getOptions() const { return m_options; }

CGFloat CoreTextFontFace::getHeight() const { return m_height; }

CTFontDescriptorRef CoreTextFontFace::getDescriptor() const {
  return m_descriptor;
}

CTFontRef CoreTextFontFace::getFont() const { return m_font; }

void CoreTextFontFace::ensureOptions(tjs_uint32 options) {
  if (m_options == options)
    return;
}

void CoreTextFontFace::createFontWithHeight(CGFloat height) {
  if (m_font && m_height == height) {
    return;
  }

  m_font = CTFontCreateWithFontDescriptor(m_descriptor, height, nullptr);
}

bool CoreTextFontFace::getGlyphRectFromCharcode(struct tTVPRect &rt,
                                                tjs_char         code,
                                                tjs_int &        advancex,
                                                tjs_int &        advancey) {
  UniChar character = static_cast<UniChar>(code);
  CGGlyph glyph     = 0;

  if (!CTFontGetGlyphsForCharacters(m_font, &character, &glyph, 1)) {
    return false;
  }

  CGSize advance;
  CGRect rect;

  CTFontGetBoundingRectsForGlyphs(m_font, kCTFontOrientationDefault, &glyph,
                                  &rect, 1);
  CTFontGetAdvancesForGlyphs(m_font, kCTFontOrientationDefault, &glyph,
                             &advance, 1);

  rt.set_size(rect.size.width, rect.size.height);
  rt.set_offsets(rect.origin.x, rect.origin.y);

  advancex = advance.width;
  advancey = advance.height;

  return true;
}

// ----------------------------------------------------------------------

CoreTextFontStorage *CoreTextFontStorage::getInstance() {
  static CoreTextFontStorage instance{};
  return &instance;
}

CoreTextFontStorage::CoreTextFontStorage() {
  m_fontMap = CFDictionaryCreateMutable(CFAllocatorGetDefault(), 0,
                                        &kCFTypeDictionaryKeyCallBacks,
                                        &kCFTypeDictionaryValueCallBacks);
  if (!m_fontMap) {
    TVPThrowExceptionMessage(TJS_W("failed to create a font cache"));
  }
}

CoreTextFontStorage::~CoreTextFontStorage() { CFRelease(m_fontMap); }

bool CoreTextFontStorage::loadFont(const ttstr &            path,
                                   std::vector<tjs_string> *faces) {
  TVPAddLog(TVPFormatMessage(TJS_W("loading font: '%1'"), path));

  // convert path to CFStringRef
  auto path_ = CFStringCreateWithBytes(
      CFAllocatorGetDefault(), reinterpret_cast<const uint8_t *>(path.c_str()),
      path.GetLen() << 1, kCFStringEncodingUTF16LE, false);

  if (!path_) {
    TVPThrowExceptionMessage(TJS_W("font name conversion failed: '%1'"), path);
  }

  auto fontDescriptors =
      reinterpret_cast<CFArrayRef>(CFDictionaryGetValue(m_fontMap, path_));

  if (fontDescriptors) {
    TVPAddLog(TVPFormatMessage(TJS_W("font already cached: '%1'"), path));

    populateFontFaces(faces, fontDescriptors);
    CFRelease(path_);

    return false;
  }

  // load font
  auto fontData = getFontData(path);
  if (!fontData) {
    TVPThrowExceptionMessage(TVPCannotOpenFontFile, path);
  }

  TVPAddLog(TVPFormatMessage(TJS_W("font loaded into the memory: '%1'"), path));

  fontDescriptors = CTFontManagerCreateFontDescriptorsFromData(fontData);
  CFRelease(fontData);

  if (!fontDescriptors) {
    TVPThrowExceptionMessage(TVPCannotOpenFontFile, path);
  }

  // insert into m_fontMap
  CFDictionaryAddValue(m_fontMap, path_, fontDescriptors);
  TVPAddLog(TVPFormatMessage(TJS_W("font successfully cahed: '%1'"), path));

  populateFontFaces(faces, fontDescriptors);
  CFRelease(path_);

  return true;
}

void CoreTextFontStorage::populateFontList(std::vector<ttstr> &   list,
                                           tjs_uint32             flags,
                                           const struct tTVPFont &_font) {
  // unused parameter _font (why?)
  auto        count = CFDictionaryGetCount(m_fontMap);
  const void *values[count];

  CFDictionaryGetKeysAndValues(m_fontMap, nullptr, values);

  for (CFIndex j = 0; j < count; j++) {
    CFArrayRef descriptors = reinterpret_cast<CFArrayRef>(values[j]);
    auto       len         = CFArrayGetCount(descriptors);

    for (CFIndex i = 0; i < len; i++) {
      auto d = reinterpret_cast<CTFontDescriptorRef>(
          CFArrayGetValueAtIndex(descriptors, i));
      if (!d) {
        continue;
      }

      if (!testFont(d, flags))
        continue;

      auto familyName = reinterpret_cast<CFStringRef>(
          CTFontDescriptorCopyAttribute(d, kCTFontDisplayNameAttribute));

      if (!familyName) {
        // family name not present
        TVPAddLog(TJS_W("family name not present"));
        continue;
      }

      auto bufSize = CFStringGetLength(familyName) << 2;
      auto cStr    = new char[bufSize];

      // clear the buffer
      memset(cStr, 0, bufSize);

      if (!CFStringGetCString(familyName, cStr, bufSize,
                              kCFStringEncodingUTF16LE)) {
        TVPThrowExceptionMessage(TJS_W("font family name conversion failed"));
      }

      auto faceName = tjs_string(reinterpret_cast<tjs_char const *>(cStr));

      delete[] cStr;

      TVPAddLog(
          TVPFormatMessage(TJS_W("populating family name: '%1'"), faceName));

      list.push_back(std::move(faceName));

      CFRelease(familyName);
    }
  }
}

CFDataRef CoreTextFontStorage::getFontData(const ttstr &path) {
  auto in = TVPCreateBinaryStreamForRead(path, TJS_W(""));

  if (!in) {
    return nullptr;
  }

  // FIXME: this loads the entire font into the memory; not ideal
  auto size    = in->GetSize();
  auto dataBuf = new uint8_t[size];
  in->ReadBuffer(dataBuf, size);

  auto data = CFDataCreate(CFAllocatorGetDefault(), dataBuf, size);

  delete[] dataBuf;
  delete in;

  return data;
}

void CoreTextFontStorage::populateFontFaces(std::vector<tjs_string> *faces,
                                            CFArrayRef descriptors) {
  TVPAddLog(TJS_W("populating faces..."));

  auto len = CFArrayGetCount(descriptors);

  for (CFIndex i = 0; i < len; i++) {
    auto d = reinterpret_cast<CTFontDescriptorRef>(
        CFArrayGetValueAtIndex(descriptors, i));
    if (!d) {
      continue;
    }

    // don't use vertical fonts
    auto orientation = reinterpret_cast<CFNumberRef>(
        CTFontDescriptorCopyAttribute(d, kCTFontOrientationAttribute));

    if (orientation) {
      uint32_t orientationVal = 0;
      CFNumberGetValue(orientation, kCFNumberLongType, &orientationVal);
      CFRelease(orientation);

      if (orientationVal != kCTFontOrientationHorizontal) {
        continue;
      }
    }

    auto familyName = reinterpret_cast<CFStringRef>(
        CTFontDescriptorCopyAttribute(d, kCTFontDisplayNameAttribute));

    if (!familyName) {
      // family name not present
      TVPAddLog(TJS_W("family name not present"));
      continue;
    }

    auto bufSize = CFStringGetLength(familyName) << 2;
    auto cStr    = new char[bufSize];

    // clear the buffer
    memset(cStr, 0, bufSize);

    if (!CFStringGetCString(familyName, cStr, bufSize,
                            kCFStringEncodingUTF16LE)) {
      TVPThrowExceptionMessage(TJS_W("font family name conversion failed"));
    }

    auto faceName = tjs_string(reinterpret_cast<tjs_char const *>(cStr));

    delete[] cStr;

    TVPAddLog(
        TVPFormatMessage(TJS_W("populating family name: '%1'"), faceName));

    faces->push_back(std::move(faceName));

    CFRelease(familyName);
  }
}

bool CoreTextFontStorage::testFont(CTFontDescriptorRef descriptor,
                                   tjs_uint32          flags) {
  // TODO: implement font test
  return true;
}

CoreTextFontFace *
CoreTextFontStorage::createFontFace(std::vector<tjs_string> const &faces,
                                    tjs_uint32                     flags) {
  // unused parameter _font (why?)
  auto        count = CFDictionaryGetCount(m_fontMap);
  const void *values[count];

  CFDictionaryGetKeysAndValues(m_fontMap, nullptr, values);

  for (auto const &face : faces) {
    // convert face into CFString
    auto faceStr = CFStringCreateWithBytesNoCopy(
        CFAllocatorGetDefault(),
        reinterpret_cast<uint8_t const *>(face.c_str()), face.size() << 1,
        kCFStringEncodingUTF16LE, false, kCFAllocatorNull);
    if (!faceStr) {
      TVPThrowExceptionMessage(
          TJS_W("failed to convert the face string into CFString: %1"), face);
    }

    for (CFIndex j = 0; j < count; j++) {
      CFArrayRef descriptors = reinterpret_cast<CFArrayRef>(values[j]);
      auto       len         = CFArrayGetCount(descriptors);

      for (CFIndex k = 0; k < len; k++) {
        auto d = reinterpret_cast<CTFontDescriptorRef>(
            CFArrayGetValueAtIndex(descriptors, k));
        if (!d) {
          continue;
        }

        auto displayName = reinterpret_cast<CFStringRef>(
            CTFontDescriptorCopyAttribute(d, kCTFontDisplayNameAttribute));
        if (!displayName) {
          continue;
        }

        if (CFStringCompare(displayName, faceStr, 0) == kCFCompareEqualTo) {
          CFRelease(displayName);
          CFRelease(faceStr);
          CFRetain(d);

          return new CoreTextFontFace(d, face, flags);
        }

        // try localized names
        CFStringRef lang = nullptr;
        displayName      = reinterpret_cast<CFStringRef>(
            CTFontDescriptorCopyLocalizedAttribute(
                d, kCTFontDisplayNameAttribute, &lang));
        if (!displayName) {
          continue;
        }

        if (lang) {
          CFRelease(lang);
        }

        if (CFStringCompare(displayName, faceStr, 0) == kCFCompareEqualTo) {
          CFRelease(displayName);
          CFRelease(faceStr);
          CFRetain(d);

          return new CoreTextFontFace(d, face, flags);
        }

        CFRelease(displayName);
      }
    }

    CFRelease(faceStr);
  }

  return nullptr;
}
