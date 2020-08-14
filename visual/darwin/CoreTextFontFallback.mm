/**
 * @file      CoreTextFontFallback.mm
 * @author    Hikaru Terazono (3c1u) <3c1u@vulpesgames.tokyo>
 * @brief     Core Text-based font rasterizer.
 * @date      2020-08-14
 *
 * @copyright Copyright (c) 2020 Hikaru Terazono. All rights reserved.
 *
 */

#import <Cocoa/Cocoa.h>
#import <CoreFoundation/CoreFoundation.h>

CFArrayRef CoreTextFont_GetCurrentSystemLanguages() {
  return (__bridge CFArrayRef)
      [[NSUserDefaults standardUserDefaults] arrayForKey:@"AppleLanguages"];
}
